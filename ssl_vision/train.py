"""
Main training script for self-supervised learning (DINO)
"""
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import create_dataloader, get_transforms
from models import (
    create_dino_model,
    update_teacher,
    DINOLoss,
)
from utils import (
    get_cosine_schedule_with_warmup,
    AverageMeter,
)


def create_models(cfg: DictConfig, device: torch.device):
    """Create student and teacher models."""
    model_name = cfg.model.name

    if model_name in ["dino_v2", "dino_v3"]:
        student, teacher = create_dino_model(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    student = student.to(device)
    teacher = teacher.to(device)
    return student, teacher


def create_dataloaders(
    cfg: DictConfig,
    distributed: bool,
    rank: int,
    world_size: int,
):
    """Create training dataloader (optionally distributed)."""
    # Simple fast transforms - NO augmentations
    transform = get_transforms(cfg)

    # Decide whether to use local files or HuggingFace datasets
    use_local_files = getattr(cfg.data, "use_local_files", False)
    data_dir = getattr(cfg.data, "data_dir", None)

    train_loader = create_dataloader(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        transform=transform,
        cache_dir=cfg.data.cache_dir,
        shuffle=True,
        pin_memory=cfg.training.pin_memory,
        image_key=cfg.data.image_key,
        prefetch_factor=cfg.training.get("prefetch_factor", 4),
        persistent_workers=cfg.training.get("persistent_workers", True),
        data_dir=data_dir,
        use_local_files=use_local_files,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    return train_loader


def create_optimizer_and_scheduler(cfg: DictConfig, student: nn.Module, train_loader):
    """Create optimizer and learning rate scheduler."""
    params = student.parameters()

    if cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

    num_training_steps = len(train_loader) * cfg.training.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.scheduler.warmup_epochs * len(train_loader),
        num_training_steps=num_training_steps,
        min_lr=cfg.scheduler.min_lr,
    )

    return optimizer, scheduler


def create_loss(cfg: DictConfig, device: torch.device):
    """Create DINO loss."""
    model_cfg = cfg.model.dino

    criterion = DINOLoss(
        out_dim=model_cfg.out_dim,
        warmup_teacher_temp=model_cfg.warmup_teacher_temp,
        teacher_temp=model_cfg.teacher_temp,
        warmup_teacher_temp_epochs=model_cfg.warmup_teacher_temp_epochs,
        nepochs=cfg.training.num_epochs,
        student_temp=model_cfg.student_temp,
    ).to(device)

    return criterion


def setup_logging(cfg: DictConfig, checkpoint_dir: Path, resume_info: Optional[dict] = None):
    """Setup logging to console (captured by SLURM)."""
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Log directory: {cfg.logging.log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    if resume_info:
        print(f"\n{'=' * 80}")
        print("📦 RESUMING FROM PREVIOUS CHECKPOINT")
        print(f"{'=' * 80}")
        print(f"Previous experiment: {resume_info['prev_experiment']}")
        print(f"Checkpoint path: {resume_info['checkpoint_path']}")
        print(f"Previous epoch: {resume_info['prev_epoch']}")
        print(f"Previous global step: {resume_info['prev_global_step']}")
        print(f"{'=' * 80}\n")


def save_checkpoint(
    checkpoint_dir: Path,
    filename: str,
    epoch: int,
    global_step: int,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: DictConfig,
):
    """Save training checkpoint."""
    # Handle both plain nn.Module and DDP-wrapped student
    student_module = student.module if hasattr(student, "module") else student

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "student_state_dict": student_module.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    save_path = checkpoint_dir / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
):
    """Load training checkpoint and return (next_epoch, global_step, metadata)."""
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    student.load_state_dict(checkpoint["student_state_dict"])
    teacher.load_state_dict(checkpoint["teacher_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    next_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    metadata = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "global_step": checkpoint.get("global_step", "unknown"),
        "checkpoint_path": str(checkpoint_file),
    }

    print(f"Checkpoint loaded from {checkpoint_file}")
    return next_epoch, global_step, metadata


def handle_resume(
    cfg: DictConfig,
    checkpoint_dir: Path,
    device: torch.device,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    start_epoch: int,
    start_global_step: int,
    is_main_process: bool,
):
    """Handle resume logic. Returns (epoch, global_step, resume_info)."""
    resume_path = cfg.checkpoint.resume_from
    resume_info = None

    # Case 1: Explicit checkpoint path provided (from previous experiment)
    if resume_path:
        checkpoint_path = Path(resume_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path}'. "
                "Ensure the path is correct or unset checkpoint.resume_from."
            )

        if is_main_process:
            print(f"📂 Loading checkpoint from previous experiment: {checkpoint_path}")
        next_epoch, global_step, metadata = load_checkpoint(
            str(checkpoint_path),
            device=device,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        resume_info = {
            "prev_experiment": checkpoint_path.parent.name,
            "checkpoint_path": metadata["checkpoint_path"],
            "prev_epoch": metadata["epoch"],
            "prev_global_step": metadata["global_step"],
        }

        return next_epoch, global_step, resume_info

    # Case 2: Starting fresh
    if is_main_process:
        print(f"🆕 Starting new experiment: {cfg.experiment_name}")
        print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
    return start_epoch, start_global_step, None


def train_one_epoch(
    cfg: DictConfig,
    epoch: int,
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    global_step: int,
    is_main_process: bool,
):
    """Train for one epoch and return (avg_loss, global_step)."""
    student.train()
    teacher.eval()

    loss_meter = AverageMeter()

    # Ensure each distributed sampler gets a different seed per epoch
    if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not is_main_process,
    )

    for batch_idx, (images, _) in enumerate(pbar):
        # Images is already a list of 2 views from collate_fn
        images = [img.to(device, non_blocking=True) for img in images]

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision, dtype=torch.bfloat16):
            student_output = student(images)    # forward all views
            teacher_output = teacher(images[:2]) # forward only 2 global views

            # Compute loss
            loss = criterion(student_output, teacher_output, epoch)

        # Backward pass
        optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        if cfg.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                cfg.training.gradient_clip,
            )

        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Update teacher
        momentum = cfg.model.dino.momentum_teacher
        with torch.no_grad():
            update_teacher(student, teacher, momentum)

        # Update metrics
        loss_meter.update(loss.item())

        # Logging
        pbar.set_postfix(
            {
                "loss": f"{loss_meter.avg:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
            }
        )

        if is_main_process and global_step % cfg.logging.log_frequency == 0:
            print(
                f"Step {global_step} - Loss: {loss_meter.avg:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        global_step += 1

    return loss_meter.avg, global_step


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main entry point (procedural training loop, no Trainer class)."""
    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    dist_cfg = getattr(cfg, "distributed", None) or {}
    use_distributed = bool(dist_cfg.get("enabled", False))
    world_size = 1
    rank = 0
    local_rank = 0

    if use_distributed and torch.cuda.is_available():
        # torchrun provides these environment variables
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            # Safety fallback: avoid hanging init_process_group if misconfigured
            print(
                "[WARN] cfg.distributed.enabled=True but RANK/WORLD_SIZE are not set. "
                "Run with torchrun or disable distributed training; falling back to single-process."
            )
            use_distributed = False
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank % max(world_size, 1)))

            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=dist_cfg.get("backend", "nccl"),
                init_method=dist_cfg.get("init_method", "env://"),
                world_size=world_size,
                rank=rank,
            )

    is_main_process = (rank == 0)

    if is_main_process:
        print("=" * 80)
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

    # Device and seeds
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.manual_seed_all(cfg.seed)
    else:
        device = torch.device("cpu")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Checkpoint directory
    base_checkpoint_dir = Path(cfg.checkpoint.save_dir)
    checkpoint_dir = base_checkpoint_dir / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create training dataloader
    train_loader = create_dataloaders(
        cfg=cfg,
        distributed=use_distributed and torch.cuda.is_available(),
        rank=rank,
        world_size=world_size,
    )

    student, teacher = create_models(cfg, device)

    # Resume (if needed)
    epoch = 0
    global_step = 0
    epoch, global_step, resume_info = handle_resume(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        device=device,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=epoch,
        start_global_step=global_step,
        is_main_process=is_main_process,
    )

    # Wrap student in DDP (after loading checkpoint so state_dict keys stay simple)
    if use_distributed and torch.cuda.is_available():
        ddp_student = DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.distributed.get(
                "find_unused_parameters",
                False,
            ),
        )

    # Compile student and teacher
    compiled_student = torch.compile(ddp_student)
    compiled_teacher = torch.compile(teacher)

    optimizer, scheduler = create_optimizer_and_scheduler(cfg, compiled_student, train_loader)
    criterion = create_loss(cfg, device)



    # Logging (after resume so we have resume_info)
    if is_main_process:
        setup_logging(cfg, checkpoint_dir, resume_info)

        print(f"\n{'=' * 80}")
        print("Training Configuration")
        print(f"{'=' * 80}")
        print(f"Experiment: {cfg.experiment_name}")
        print(f"Starting epoch: {epoch}")
        print(f"Target epochs: {cfg.training.num_epochs}")
        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Device: {device}")
        if use_distributed:
            print(f"World size: {world_size}")
        print(f"{'=' * 80}\n")

    # Training loop
    for ep in range(epoch, cfg.training.num_epochs):
        # Train for one epoch
        avg_loss, global_step = train_one_epoch(
            cfg=cfg,
            epoch=ep,
            student=compiled_student,
            teacher=compiled_teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            global_step=global_step,
            is_main_process=is_main_process,
        )

        # Save checkpoint (only on main process)
        if is_main_process and ep % cfg.checkpoint.save_frequency == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                filename=f"checkpoint_epoch_{ep}.pth",
                epoch=ep,
                global_step=global_step,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
            )

        if is_main_process:
            print(f"Epoch {ep}: Loss = {avg_loss:.4f}")

    # Save final checkpoint (only on main process)
    if is_main_process:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            filename="checkpoint_final.pth",
            epoch=cfg.training.num_epochs - 1,
            global_step=global_step,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
        )

        print("Training completed!")

    # Clean up distributed state
    if use_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

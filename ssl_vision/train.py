"""
Main training script for self-supervised learning (DINO)
"""
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Optional

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


def create_dataloaders(cfg: DictConfig):
    """Create training dataloader."""
    # Simple fast transforms - NO augmentations
    transform = get_transforms(cfg)

    train_loader = create_dataloader(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        transform=transform,
        cache_dir=cfg.data.cache_dir,
        streaming=cfg.data.streaming,
        pin_memory=cfg.training.pin_memory,
        image_key=cfg.data.image_key,
        prefetch_factor=cfg.training.get("prefetch_factor", 4),
        persistent_workers=cfg.training.get("persistent_workers", True),
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
        ncrops=2,  # Only 2 views (same image duplicated for speed)
        warmup_teacher_temp=model_cfg.warmup_teacher_temp,
        teacher_temp=model_cfg.teacher_temp,
        warmup_teacher_temp_epochs=model_cfg.warmup_teacher_temp_epochs,
        nepochs=cfg.training.num_epochs,
        student_temp=model_cfg.student_temp,
    ).to(device)

    return criterion


def setup_logging(cfg: DictConfig, checkpoint_dir: Path):
    """Setup logging to console (captured by SLURM)."""
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Log directory: {cfg.logging.log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """Find the latest checkpoint in the experiment directory."""
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None

    def get_epoch_num(path: Path):
        try:
            return int(path.stem.split("_")[-1])
        except Exception:
            return -1

    latest = max(checkpoint_files, key=get_epoch_num)
    return str(latest)


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
    scaler: Optional[torch.cuda.amp.GradScaler],
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

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
    scaler: Optional[torch.cuda.amp.GradScaler],
):
    """Load training checkpoint and return (next_epoch, global_step)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    student.load_state_dict(checkpoint["student_state_dict"])
    teacher.load_state_dict(checkpoint["teacher_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    next_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]

    print(f"Checkpoint loaded from {checkpoint_path}")
    return next_epoch, global_step


def handle_resume(
    cfg: DictConfig,
    checkpoint_dir: Path,
    device: torch.device,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    start_epoch: int,
    start_global_step: int,
):
    """Handle automatic resume logic. Returns (epoch, global_step)."""
    resume_path = cfg.checkpoint.resume_from

    # Case 1: Explicit checkpoint path provided
    if resume_path and resume_path != "auto" and Path(resume_path).exists():
        print(f"📂 Resuming from explicit checkpoint: {resume_path}")
        return load_checkpoint(
            resume_path,
            device=device,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    # Case 2: Auto-resume enabled (for SLURM requeue)
    if cfg.checkpoint.auto_resume or resume_path == "auto":
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"🔄 Auto-resuming from: {latest_checkpoint}")
            return load_checkpoint(
                latest_checkpoint,
                device=device,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

    # Case 3: Starting fresh
    print(f"🆕 Starting new experiment: {cfg.experiment_name}")
    print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
    return start_epoch, start_global_step


def train_one_epoch(
    cfg: DictConfig,
    epoch: int,
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    global_step: int,
):
    """Train for one epoch and return (avg_loss, global_step)."""
    student.train()
    teacher.eval()

    loss_meter = AverageMeter()

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
    )

    for batch_idx, (images, _) in enumerate(pbar):
        # Images is already a list of 2 views from collate_fn
        images = [img.to(device, non_blocking=True) for img in images]

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
            # Student forward (all views)
            student_output = student(images)

            # Teacher forward (all views - same as student since we only have 2)
            teacher_output = teacher(images)

            # Compute loss
            loss = criterion(student_output, teacher_output, epoch)

        # Backward pass
        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()

            # Gradient clipping
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    cfg.training.gradient_clip,
                )

            scaler.step(optimizer)
            scaler.update()
        else:
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

        if global_step % cfg.logging.log_frequency == 0:
            print(
                f"Step {global_step} - Loss: {loss_meter.avg:.4f}, "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        global_step += 1

    return loss_meter.avg, global_step


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main entry point (procedural training loop, no Trainer class)."""
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Device and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Checkpoint directory
    base_checkpoint_dir = Path(cfg.checkpoint.save_dir)
    checkpoint_dir = base_checkpoint_dir / cfg.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create components
    student, teacher = create_models(cfg, device)
    train_loader = create_dataloaders(cfg)
    optimizer, scheduler = create_optimizer_and_scheduler(cfg, student, train_loader)
    criterion = create_loss(cfg, device)
    scaler = torch.cuda.amp.GradScaler() if cfg.training.mixed_precision else None

    # Logging
    setup_logging(cfg, checkpoint_dir)

    # Resume (if needed)
    epoch = 0
    global_step = 0
    epoch, global_step = handle_resume(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        device=device,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        start_epoch=epoch,
        start_global_step=global_step,
    )

    # Training loop
    print(f"\n{'=' * 80}")
    print("Training Configuration")
    print(f"{'=' * 80}")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Starting epoch: {epoch}")
    print(f"Target epochs: {cfg.training.num_epochs}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Device: {device}")
    print(f"{'=' * 80}\n")

    for ep in range(epoch, cfg.training.num_epochs):
        # Train for one epoch
        avg_loss, global_step = train_one_epoch(
            cfg=cfg,
            epoch=ep,
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            global_step=global_step,
        )

        # Save checkpoint
        if ep % cfg.checkpoint.save_frequency == 0:
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
                scaler=scaler,
            )

        print(f"Epoch {ep}: Loss = {avg_loss:.4f}")

    # Save final checkpoint
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
        scaler=scaler,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()

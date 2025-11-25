"""
Main training script for self-supervised learning (DINO)
"""
import json
import math
import os
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from ssl_vision.data_loader import create_dataloader, get_transforms, SubmissionDataset, submission_collate_fn, get_eval_transforms
from ssl_vision.models import create_dino_model, update_teacher, DINOLoss
from ssl_vision.utils import get_cosine_schedule_with_warmup, AverageMeter, KNNClassifier, extract_features


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

    data_dir = getattr(cfg.data, "data_dir", None)

    train_loader = create_dataloader(
        dataset_name=cfg.data.dataset_name,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        transform=transform,
        cache_dir=cfg.data.cache_dir,
        shuffle=True,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.get("prefetch_factor", 4),
        persistent_workers=cfg.training.get("persistent_workers", True),
        data_dir=data_dir,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    return train_loader


def run_knn_evaluations(backbone, eval_loaders, device, k):
    """Run k-NN evaluation for each dataset and return accuracies."""
    results = {}
    if not eval_loaders:
        return results

    for dataset_name, (train_loader, val_loader) in eval_loaders.items():
        train_features, train_labels, _ = extract_features(backbone, train_loader, device)
        val_features, val_labels, _ = extract_features(backbone, val_loader, device)

        knn = KNNClassifier(k=k)
        knn.train(train_features, train_labels)
        results[dataset_name] = knn.evaluate(val_features, val_labels)

    return results


def update_best_checkpoint_symlinks(checkpoint_dir: Path, dataset_name: str, ranked_entries, top_k: int):
    """Create symlinks for top-k checkpoints per dataset."""
    if not ranked_entries:
        return

    best_dir = checkpoint_dir / f"best_{dataset_name}"
    best_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous links/files
    for existing in best_dir.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()

    for record in ranked_entries[:top_k]:
        target = Path(record["checkpoint"])
        if not target.exists():
            print(f"[WARN] Cannot create symlink for missing checkpoint: {target}")
            continue

        link_path = best_dir / target.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        os.symlink(target, link_path)

    print(f"  Symlinks updated in {best_dir}")


def finalize_training(
    *,
    checkpoint_dir: Path,
    final_epoch: int,
    final_checkpoint_filename: str,
    global_step: int,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    cfg: DictConfig,
    eval_history,
    top_k_models: int,
):
    """Persist the last checkpoint and summarize k-NN evaluations."""
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        filename=final_checkpoint_filename,
        epoch=final_epoch,
        global_step=global_step,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )

    if any(eval_history.values()):
        print("\nTop k-NN checkpoints per dataset:")
        summary = {}
        for dataset_name, entries in eval_history.items():
            if not entries:
                continue
            ranked = sorted(entries, key=lambda e: e["accuracy"], reverse=True)
            summary[dataset_name] = ranked
            print(f"{dataset_name}:")
            for rank_idx, record in enumerate(ranked[:top_k_models], start=1):
                acc = record["accuracy"] * 100
                print(f"  #{rank_idx} Epoch {record['epoch']} - {acc:.2f}% ({record['checkpoint']})")
            update_best_checkpoint_symlinks(checkpoint_dir, dataset_name, ranked, top_k_models)

        summary_path = checkpoint_dir / "knn_eval_history.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull k-NN history saved to: {summary_path}")

    print("Training completed!")


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


def build_teacher_momentum_schedule(cfg: DictConfig, total_steps: int):
    """
    Build a cosine schedule for the teacher momentum (EMA) coefficient.
    Matches the strategy used in DINO: start from a base momentum and
    asymptotically approach 1.0 over training.
    """
    model_cfg = cfg.model.dino
    base_momentum = getattr(model_cfg, "momentum_teacher", 0.996)
    final_momentum = getattr(model_cfg, "momentum_teacher_end", 1.0)

    if total_steps <= 0:
        return [final_momentum]

    denom = max(total_steps - 1, 1)
    schedule = []
    for step in range(total_steps):
        progress = step / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        momentum = final_momentum - (final_momentum - base_momentum) * cosine
        schedule.append(momentum)

    return schedule


def build_weight_decay_schedule(cfg: DictConfig, total_steps: int):
    """
    Cosine schedule for AdamW weight decay (student optimizer).
    """
    opt_cfg = cfg.optimizer
    start_wd = getattr(opt_cfg, "weight_decay", 0.0)
    end_wd = getattr(opt_cfg, "weight_decay_end", start_wd)

    if total_steps <= 0 or start_wd == end_wd:
        return [start_wd]

    denom = max(total_steps - 1, 1)
    schedule = []
    for step in range(total_steps):
        progress = step / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        weight_decay = end_wd + (start_wd - end_wd) * cosine
        schedule.append(weight_decay)

    return schedule


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
    teacher_momentum_schedule,
    weight_decay_schedule,
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
        desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}",
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

        # Update weight decay before optimizer step
        wd_idx = min(global_step, len(weight_decay_schedule) - 1)
        current_weight_decay = weight_decay_schedule[wd_idx]
        for param_group in optimizer.param_groups:
            param_group["weight_decay"] = current_weight_decay

        optimizer.step()

        # Update learning rate
        scheduler.step()

        # Update teacher with scheduled momentum
        momentum_idx = min(global_step, len(teacher_momentum_schedule) - 1)
        with torch.no_grad():
            update_teacher(student, teacher, teacher_momentum_schedule[momentum_idx])

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


@hydra.main(version_base=None, config_path="../configs", config_name="ssl_train")
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

    total_training_steps = len(train_loader) * cfg.training.num_epochs
    teacher_momentum_schedule = build_teacher_momentum_schedule(
        cfg=cfg,
        total_steps=total_training_steps,
    )
    weight_decay_schedule = build_weight_decay_schedule(
        cfg=cfg,
        total_steps=total_training_steps,
    )


    # create evaluation dataloaders for feature extraction for K-NN
    eval_cfg = cfg.evaluation
    eval_transform = get_eval_transforms()
    eval_dataloaders = {}

    eval_data_list = [eval_cfg.data[k] for k in sorted(eval_cfg.data.keys(), key=lambda x: int(x))]

    for idx, eval_data_cfg in enumerate(eval_data_list):
        dataset_name = eval_data_cfg.dataset_name
        print(f"Loading evaluation dataset [{idx}]: {dataset_name}")
        print(f"  Config: {eval_data_cfg}")
        train_split = getattr(eval_data_cfg, "train_split", "train")
        val_split = getattr(eval_data_cfg, "val_split", "val")

        eval_train_dataset = SubmissionDataset(
            root_dir=eval_data_cfg.data_dir,
            split=train_split,
            transform=eval_transform,
        )

        eval_val_dataset = SubmissionDataset(
            root_dir=eval_data_cfg.data_dir,
            split=val_split,
            transform=eval_transform,
        )

        eval_train_loader = DataLoader(
            eval_train_dataset,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            collate_fn=submission_collate_fn,
        )

        eval_val_loader = DataLoader(
            eval_val_dataset,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            collate_fn=submission_collate_fn,
        )

        eval_dataloaders[dataset_name] = (eval_train_loader, eval_val_loader)

    eval_history = {name: [] for name in eval_dataloaders.keys()}
    knn_eval_frequency = getattr(eval_cfg, "knn_eval_frequency", getattr(cfg.early_stopping, "eval_frequency", 10))
    top_k_models = getattr(eval_cfg, "top_k_models", 5)

    early_cfg = getattr(cfg, "early_stopping", {})
    early_patience = max(0, int(getattr(early_cfg, "patience", 0)))
    early_min_delta = float(getattr(early_cfg, "min_delta", 0.0))
    early_stop_datasets = {
        "cub200": "CUB-200",
        "mini_imagenet": "mini-ImageNet",
    }
    early_stop_enabled = early_patience > 0 and bool(eval_dataloaders)
    best_avg_knn = None
    epochs_without_improve = 0
    stopped_early = False
    early_stop_epoch = None



    student, teacher = create_models(cfg, device)

    optimizer, scheduler = create_optimizer_and_scheduler(cfg, student, train_loader)
    criterion = create_loss(cfg, device)

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
        compiled_student = torch.compile(ddp_student)
    else:
        compiled_student = torch.compile(student)

    compiled_teacher = torch.compile(teacher)


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
        early_stop_triggered_this_epoch = False

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
            teacher_momentum_schedule=teacher_momentum_schedule,
            weight_decay_schedule=weight_decay_schedule,
        )

        # Save checkpoint (only on main process)
        if is_main_process and (ep + 1) % cfg.checkpoint.save_frequency == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                filename=f"checkpoint_epoch_{ep + 1}.pth",
                epoch=ep,
                global_step=global_step,
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
            )

        if is_main_process:
            print(f"Epoch {ep + 1}/{cfg.training.num_epochs}: Loss = {avg_loss:.4f}")

        should_eval_knn = bool(eval_dataloaders) and ((ep + 1) % knn_eval_frequency == 0)
        if should_eval_knn and is_main_process:
            knn_k = eval_cfg.knn_k
            current_ckpt_name = f"checkpoint_epoch_{ep + 1}.pth"
            current_ckpt_path = checkpoint_dir / current_ckpt_name

            if not current_ckpt_path.exists():
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    filename=current_ckpt_name,
                    epoch=ep,
                    global_step=global_step,
                    student=student,
                    teacher=teacher,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                )

            print(f"\n[Eval] k-NN evaluation at epoch {ep + 1}")
            prev_student_mode = student.training
            student.eval()

            knn_results = run_knn_evaluations(student.backbone, eval_dataloaders, device, knn_k)
            for dataset_name, val_acc in knn_results.items():
                eval_history[dataset_name].append(
                    {
                        "epoch": ep + 1,
                        "accuracy": float(val_acc),
                        "checkpoint": str(current_ckpt_path),
                    }
                )
                print(f"  {dataset_name}: {val_acc * 100:.2f}% ({current_ckpt_path.name})")

            tracked_accs = []
            missing_datasets = []
            for dataset_key, friendly_name in early_stop_datasets.items():
                if dataset_key in knn_results:
                    tracked_accs.append(float(knn_results[dataset_key]))
                else:
                    missing_datasets.append(friendly_name)

            avg_knn_acc = None
            if tracked_accs and not missing_datasets:
                avg_knn_acc = float(np.mean(tracked_accs))
                print(f"  Avg (CUB-200 + mini-ImageNet): {avg_knn_acc * 100:.2f}%")
            elif early_stop_enabled and missing_datasets:
                missing_str = ", ".join(missing_datasets)
                print(f"[EarlyStopping] Missing datasets for average: {missing_str}. Skipping update.")

            if early_stop_enabled and avg_knn_acc is not None:
                if best_avg_knn is None or avg_knn_acc >= best_avg_knn + early_min_delta:
                    best_avg_knn = avg_knn_acc
                    epochs_without_improve = 0
                    print(f"[EarlyStopping] New best avg k-NN accuracy: {avg_knn_acc * 100:.2f}%")
                else:
                    epochs_without_improve += 1
                    delta = avg_knn_acc - best_avg_knn
                    print(
                        f"[EarlyStopping] No improvement for {epochs_without_improve}/{early_patience} "
                        f"evaluations (Δ={delta:+.4f})."
                    )
                    if epochs_without_improve >= early_patience:
                        early_stop_triggered_this_epoch = True
                        early_stop_epoch = ep
                        print(f"[EarlyStopping] Patience exhausted at epoch {ep + 1}. Triggering early stop.")

            if prev_student_mode:
                student.train()

        stop_signal = torch.tensor(1 if early_stop_triggered_this_epoch else 0, device=device)
        if use_distributed and dist.is_initialized():
            dist.broadcast(stop_signal, src=0)
        if stop_signal.item():
            stopped_early = True
            if early_stop_epoch is None:
                early_stop_epoch = ep
            break



    default_final_epoch = cfg.training.num_epochs - 1
    if stopped_early and early_stop_epoch is not None:
        final_epoch_idx = early_stop_epoch
        final_ckpt_name = f"checkpoint_epoch_{early_stop_epoch + 1}.pth"
    else:
        final_epoch_idx = default_final_epoch
        final_ckpt_name = f"checkpoint_epoch_{default_final_epoch}.pth"

    if is_main_process:
        if stopped_early and early_stop_epoch is not None:
            print(f"\n[EarlyStopping] Training stopped at epoch {early_stop_epoch + 1}.")
        finalize_training(
            checkpoint_dir=checkpoint_dir,
            final_epoch=final_epoch_idx,
            final_checkpoint_filename=final_ckpt_name,
            global_step=global_step,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            eval_history=eval_history,
            top_k_models=top_k_models,
        )

    # Clean up distributed state
    if use_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

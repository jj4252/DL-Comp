"""
Data loading utilities for self-supervised and evaluation pipelines.

Supports:
- HuggingFace datasets (streaming or cached)
- Local image folders (for faster iteration when data is stored on disk)
"""
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from datasets import load_dataset


class MultiCropTransform:
    """
    Multi-crop augmentation for self-supervised learning (DINO)
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float],
        local_crops_scale: Tuple[float, float],
        local_crops_number: int,
        global_crops_number: int = 2,
        color_jitter: float = 0.4,
        grayscale_prob: float = 0.2,
        gaussian_blur_prob: float = 0.5,
        solarization_prob: float = 0.2,
    ):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # Normalization (ImageNet stats)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Global crop transformation
        self.global_transfo = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter / 4
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob
            ),
            transforms.ToTensor(),
            normalize,
        ])

        # Add solarization to second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter / 4
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob
            ),
            transforms.RandomSolarize(threshold=128, p=solarization_prob),
            transforms.ToTensor(),
            normalize,
        ])

        # Local crop transformation
        self.local_transfo = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter / 4
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_prob),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                p=gaussian_blur_prob
            ),
            transforms.ToTensor(),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # Global crops
        crops.append(self.global_transfo(image))
        crops.append(self.global_transfo2(image))
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class HuggingFaceImageDataset(Dataset):
    """
    Wrapper for HuggingFace datasets with custom transformations
    """
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform=None,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        image_key: str = "image",
    ):
        self.transform = transform
        self.image_key = image_key

        print(f"Loading dataset: {dataset_name}, split: {split}")

        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        print(f"Dataset loaded with {len(self.dataset) if not streaming else 'streaming'} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image
        image = item[self.image_key]

        # Convert to PIL Image if necessary
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformation
        if self.transform:
            image = self.transform(image)

        # Get label if available (for evaluation), otherwise return index
        if 'label' in item:
            label = item['label']
        elif 'fine_label' in item:  # CIFAR-100 uses 'fine_label'
            label = item['fine_label']
        else:
            label = idx  # Use index as pseudo-label for unsupervised training

        return image, label


class LocalImageDataset(Dataset):
    """
    Simple dataset for reading images from a local directory.

    This is primarily intended for self-supervised training on large unlabeled
    collections of images (e.g., the 500k competition images stored locally).

    Behaviour:
    - Recursively scans `root_dir` for image files with common extensions.
    - Uses the file index as a pseudo-label (labels are not used in SSL loss).
    - Applies the provided transform (which can be a multi-crop transform).
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = str(root_dir)
        self.transform = transform

        root_path = Path(self.root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"[LocalImageDataset] root_dir does not exist: {self.root_dir}")

        # Collect all image paths recursively
        self.samples: List[str] = []
        for dirpath, _, filenames in os.walk(self.root_dir):
            for fname in filenames:
                if fname.lower().endswith(self.IMG_EXTENSIONS):
                    self.samples.append(os.path.join(dirpath, fname))

        if len(self.samples) == 0:
            raise RuntimeError(f"[LocalImageDataset] No images found in {self.root_dir}")

        print(f"[LocalImageDataset] Loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        try:
            # Try to load the original image
            with Image.open(path) as img:
                image = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            # If corrupted, move to the next index (wrap around at the end)
            print(f"[WARN] Skipping corrupted image at index {idx}: {path} ({e})")
            idx = (idx + 1) % len(self.samples)
            path = self.samples[idx]
            # Assume this one is fine
            with Image.open(path) as img:
                image = img.convert("RGB")

    # Apply transformation (may be MultiCropTransform)
        if self.transform:
            image = self.transform(image)

        # Use index as pseudo-label (not used for SSL, keeps API consistent)
        label = idx
        return image, label


def collate_fn(batch):
    """
    Custom collate function for multi-crop batches.
    Handles the case where each sample returns a list of crops.
    """
    try:
        # Check if the first image is a list of crops or a single tensor
        first_img = batch[0][0]

        if isinstance(first_img, list):
            # Multi-crop case: transpose to get list of crop batches
            # batch is a list of (crops_list, label) tuples
            # We want to convert it to (list_of_crop_batches, labels_batch)

            num_crops = len(first_img)
            crops_batched = []

            for i in range(num_crops):
                # Stack the i-th crop from all samples in the batch
                crop_batch = torch.stack([item[0][i] for item in batch])
                crops_batched.append(crop_batch)

            labels = torch.tensor([item[1] for item in batch])
            return crops_batched, labels
        else:
            # Single image case: wrap in list to maintain consistency
            # This ensures train.py always receives a list
            images = [torch.stack([item[0] for item in batch])]
            labels = torch.tensor([item[1] for item in batch])
            return images, labels

    except Exception as e:
        print(f"[ERROR COLLATE] Exception in collate_fn: {e}")
        print(f"[ERROR COLLATE] Batch type: {type(batch)}")
        print(f"[ERROR COLLATE] Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
        if len(batch) > 0:
            print(f"[ERROR COLLATE] First item: {batch[0]}")
        raise


def create_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    transform,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    pin_memory: bool = True,
    image_key: str = "image",
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    data_dir: Optional[str] = None,
    use_local_files: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for self-supervised learning with optimizations.

    Two modes are supported:
    - HuggingFace mode (default): uses `load_dataset(dataset_name, split=...)`
    - Local mode: if `use_local_files=True`, loads images from `data_dir`
      using `LocalImageDataset`. This is useful for large on-disk datasets
      like the 500k competition images.
    """
    if use_local_files:
        if data_dir is None:
            raise ValueError(
                "create_dataloader: `data_dir` must be provided when `use_local_files=True`"
            )
        print(f"[create_dataloader] Using LocalImageDataset from: {data_dir}")
        dataset = LocalImageDataset(
            root_dir=data_dir,
            transform=transform,
        )
    else:
        dataset = HuggingFaceImageDataset(
            dataset_name=dataset_name,
            split=split,
            transform=transform,
            cache_dir=cache_dir,
            streaming=streaming,
            image_key=image_key,
        )

    # Sampler / shuffling logic
    sampler = None
    shuffle = not streaming
    if distributed:
        if world_size is None:
            raise ValueError("world_size must be provided when distributed=True")
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,  # Use custom collate function
        prefetch_factor=prefetch_factor,  # Prefetch more batches
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        multiprocessing_context='spawn',
    )

    return dataloader


def get_transforms(cfg):
    """
    Create transforms based on model configuration
    """
    model_name = cfg.model.name

    if model_name in ['dino_v2', 'dino_v3']:
        # Multi-crop transformation for self-supervised learning
        transform = MultiCropTransform(
            global_crops_scale=tuple(cfg.model[cfg.model.name.replace('_v2', '').replace('_v3', '')].global_crops_scale),
            local_crops_scale=tuple(cfg.model[cfg.model.name.replace('_v2', '').replace('_v3', '')].local_crops_scale),
            local_crops_number=cfg.model[cfg.model.name.replace('_v2', '').replace('_v3', '')].local_crops_number,
            color_jitter=cfg.data.augmentation.color_jitter,
            grayscale_prob=cfg.data.augmentation.grayscale_prob,
            gaussian_blur_prob=cfg.data.augmentation.gaussian_blur_prob,
            solarization_prob=cfg.data.augmentation.solarization_prob,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return transform

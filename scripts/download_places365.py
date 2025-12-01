#!/usr/bin/env python

from torchvision.datasets import Places365
from pathlib import Path

root = Path("data/Places365")
root.mkdir(parents=True, exist_ok=True)

# Trigger the download
# - split="train-standard"  → training split
# - small=True              → 256x256 images
# - download=True           → actually downloads (~30GB)
print("Downloading Places365 train-standard (256x256)...")

_ = Places365(str(root), split="train-standard", small=True, download=True, transform=None)

print("Download complete.")

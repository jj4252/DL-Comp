#!/usr/bin/env python

import argparse
import os
from pathlib import Path

from tqdm import tqdm


def is_category_dir(dirpath: Path) -> bool:
    """
    Return True if this directory looks like a Places365 category directory,
    i.e., it contains at least one file named like 00000001.jpg (8 digits + .jpg).
    """
    for name in os.listdir(dirpath):
        if not name.lower().endswith(".jpg"):
            continue
        stem = Path(name).stem  # e.g. "00000001"
        if len(stem) == 8 and stem.isdigit():
            return True
    return False


def collect_category_dirs(input_dir: Path):
    """
    Recursively walk input_dir and collect all directories that look like
    final category dirs (contain 8-digit .jpg files).
    """
    category_dirs = []
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        # Skip if no files here
        if not files:
            continue
        # Check if this dir is a category_dir
        if is_category_dir(root_path):
            category_dirs.append(root_path)
    return sorted(category_dirs)


def create_symlinks(input_dir: Path, output_dir: Path, start_idx: int, end_idx: int) -> None:
    """
    Recursively find category directories under input_dir (any depth),
    then for each category dir, create symlinks for images [start_idx, end_idx]
    under output_dir, preserving the relative path structure.

    Example:
        input_dir  = data_dl/Places365/data_256
        category   = a/apartment_building/outdoor
        image file = 00000300.jpg

        output_dir/a/apartment_building/outdoor/00000300.jpg  -> symlink to
        data_dl/Places365/data_256/a/apartment_building/outdoor/00000300.jpg
    """
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursively collect all "final" category directories
    category_dirs = collect_category_dirs(input_dir)

    # Process categories with progress bar
    for category_dir in tqdm(category_dirs, desc="Processing categories"):
        rel_category_path = category_dir.relative_to(input_dir)

        for idx in range(start_idx, end_idx + 1):
            filename = f"{idx:08d}.jpg"  # e.g. 00000300.jpg

            src = category_dir / filename
            if not src.is_file():
                # This category may not have that many images; just skip this index
                continue

            # Preserve full relative path from input_dir
            dst = output_dir / rel_category_path / filename
            dst.parent.mkdir(parents=True, exist_ok=True)

            # If symlink or file already exists, skip
            if dst.exists():
                continue

            os.symlink(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create symbolic links for a subset of Places365 images, "
                    "preserving directory structure (arbitrary depth)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory of Places365 data_256 (e.g., data_dl/Places365/data_256)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where symlinked subset will be created.",
    )
    parser.add_argument(
        "--img_start_idx",
        type=int,
        required=True,
        help="Starting image index (e.g., 300 for 00000300.jpg).",
    )
    parser.add_argument(
        "--img_end_idx",
        type=int,
        required=True,
        help="Ending image index (e.g., 777 for 00000777.jpg).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    start_idx = args.img_start_idx
    end_idx = args.img_end_idx

    assert 1 <= start_idx <= 5000, "start_idx must be between 1 and 5000"
    assert 1 <= end_idx <= 5000, "end_idx must be between 1 and 5000"
    assert start_idx < end_idx, "start_idx must be less than end_idx"

    create_symlinks(input_dir, output_dir, start_idx, end_idx)


if __name__ == "__main__":
    main()
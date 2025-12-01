#!/usr/bin/env python

import argparse
import os
from pathlib import Path

from tqdm import tqdm


def create_symlinks(input_dir: Path, output_dir: Path, start_idx: int, end_idx: int) -> None:
    """
    For each category in Places365 (organized as input_dir/first_letter/category),
    create symlinks for images [start_idx, end_idx] under output_dir, preserving
    the relative path structure.

    Example:
        input_dir  = data_dl/Places365/data_256
        category   = a/airfield
        image file = 00000300.jpg

        output_dir/a/airfield/00000300.jpg  -> symlink to
        data_dl/Places365/data_256/a/airfield/00000300.jpg
    """

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all category directories
    category_dirs = []
    for first_level_dir in sorted(input_dir.iterdir()):
        if not first_level_dir.is_dir():
            continue

        for category_dir in sorted(first_level_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category_dirs.append(category_dir)

    # Process categories with progress bar
    for category_dir in tqdm(category_dirs, desc="Processing categories"):
        # Compute the relative path of this category w.r.t input_dir
        rel_category_path = category_dir.relative_to(input_dir)

        # For each index from start_idx to end_idx, create symlink if source exists
        for idx in range(start_idx, end_idx + 1):
            # Filenames are 8-digit zero-padded, e.g. 00000300.jpg
            filename = f"{idx:08d}.jpg"

            src = category_dir / filename
            if not src.is_file():
                # If the category doesn't have that many images, just skip
                continue

            # Destination path preserves relative path
            # Example:
            #   output_dir / "a/airfield" / "00000300.jpg"
            dst = output_dir / rel_category_path / filename
            dst.parent.mkdir(parents=True, exist_ok=True)

            # If symlink or file already exists, skip
            if dst.exists():
                continue

            os.symlink(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create symbolic links for a subset of Places365 images, "
                    "preserving directory structure."
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

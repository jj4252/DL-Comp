import os
from datasets import load_dataset
from tqdm import tqdm

# 1. Load the dataset
ds = load_dataset(
    "sasha/birdsnap",
    split="train",
    cache_dir="./.cache",
)

root = "data/Birdsnap/train"
os.makedirs(root, exist_ok=True)

n_total = len(ds)
n_saved = 0
n_skipped_exists = 0
n_skipped_broken_read = 0   # failed when reading from dataset
n_skipped_broken_save = 0   # failed when saving as JPEG

for idx in tqdm(range(n_total), total=n_total):
    # -----------------------------
    # Step 1: safely read example
    # -----------------------------
    try:
        example = ds[idx]
        # example is a dict-like:
        # {
        #   "image": PIL.Image,
        #   "label": str, ...
        # }
    except Exception as e:
        # This catches errors like:
        # - truncated image file during PIL load
        # - any decoding issue in HF datasets
        n_skipped_broken_read += 1
        # Optionally log:
        # print(f"Skipping idx {idx} due to read error: {e}")
        continue

    img = example["image"]    # PIL.Image
    label = example["label"]  # e.g. "Blue_Jay"

    # -----------------------------
    # Step 2: ensure RGB (3 channels)
    # -----------------------------
    # Before: img.mode could be "RGB" (~H×W×3), "RGBA" (~H×W×4), "P", ...
    if img.mode != "RGB":
        img = img.convert("RGB")
        # After: img.mode == "RGB" (~H×W×3)

    # -----------------------------
    # Step 3: build folder and path
    # -----------------------------
    species_dir = label.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(root, species_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{idx:05d}.jpg")

    # -----------------------------
    # Step 4: skip if file already exists
    # -----------------------------
    if os.path.exists(out_path):
        n_skipped_exists += 1
        continue

    # -----------------------------
    # Step 5: safely save JPEG
    # -----------------------------
    try:
        img.save(out_path, format="JPEG")
        n_saved += 1
    except Exception as e:
        n_skipped_broken_save += 1
        # Optionally:
        # print(f"Skipping idx {idx} ({label}) due to save error: {e}")
        continue

print("Done!")
print(f"Total examples in dataset      : {n_total}")
print(f"Saved new images               : {n_saved}")
print(f"Skipped (already existed)      : {n_skipped_exists}")
print(f"Skipped (broken on read)       : {n_skipped_broken_read}")
print(f"Skipped (broken on save)       : {n_skipped_broken_save}")

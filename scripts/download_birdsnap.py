import os
from datasets import load_dataset
from tqdm import tqdm

# 1. Load the dataset (already cached will be reused)
ds = load_dataset(
    "sasha/birdsnap",
    split="train",
    cache_dir="/gpfs/scratch/dl5635/.cache",
)

root = "data/Birdsnap/train"
os.makedirs(root, exist_ok=True)

n_total = len(ds)
n_saved = 0
n_skipped_exists = 0
n_skipped_broken = 0

for idx, example in tqdm(enumerate(ds), total=n_total):
    # -----------------------------
    # Step 1: Get image and label
    # -----------------------------
    img = example["image"]   # PIL.Image object
    label = example["label"] # e.g. "Blue_Jay"

    # img is something like (H, W, C) internally:
    # - if mode "RGB": C = 3
    # - if mode "RGBA": C = 4

    # -----------------------------
    # Step 2: Convert to RGB (3 channels)
    # -----------------------------
    if img.mode != "RGB":
        img = img.convert("RGB")
        # After this: img.mode == "RGB" → shape ~ (H, W, 3)

    # -----------------------------
    # Step 3: Build folder and path
    # -----------------------------
    species_dir = label.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(root, species_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{idx:05d}.jpg")

    # -----------------------------
    # Step 4: Skip if file already exists
    # -----------------------------
    if os.path.exists(out_path):
        n_skipped_exists += 1
        continue

    # -----------------------------
    # Step 5: Try saving; skip if broken
    # -----------------------------
    try:
        # This can fail if the image object is somehow corrupted
        img.save(out_path, format="JPEG")
        n_saved += 1
    except Exception as e:
        # If anything goes wrong (broken image, IO error), skip it
        n_skipped_broken += 1
        # Optionally log:
        # print(f"Skipping idx {idx} ({label}) due to error: {e}")
        continue

print("Done!")
print(f"Total examples in dataset  : {n_total}")
print(f"Saved new images           : {n_saved}")
print(f"Skipped (already existed)  : {n_skipped_exists}")
print(f"Skipped (broken / errors)  : {n_skipped_broken}")

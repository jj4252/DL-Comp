import os
import csv
import random
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path("/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition")
INAT_DIR = BASE_DIR / "data/iNat_birds"
SOURCE_DIR = INAT_DIR / "birds_train_small"
CSV_PATH = BASE_DIR / "aba_english_scientific.csv"

# Destination dirs
NA_1 = INAT_DIR / "NA_1"
NA_2 = INAT_DIR / "NA_2"
NON_NA_1 = INAT_DIR / "non_NA_1"
NON_NA_2 = INAT_DIR / "non_NA_2"

def setup_dirs():
    for d in [NA_1, NA_2, NON_NA_1, NON_NA_2]:
        if d.exists():
            print(f"Directory {d} already exists. Proceeding to populate (existing links may be overwritten).")
        d.mkdir(parents=True, exist_ok=True)

def load_na_species(csv_path):
    na_species = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sc_name = row.get('scientific_name', '')
            if sc_name:
                # Match wow.py: just lower()
                # We strip() to be safe against surrounding whitespace in CSV fields
                na_species.add(sc_name.strip().lower())
    return na_species

def get_species_name_from_folder(folder_name):
    # Match wow.py logic
    # Expected Format: XXXXX_..._Family_Genus_species
    parts = folder_name.split('_')
    if len(parts) < 2:
        return None
    # Scientific name is the last two parts joined by space, lowercased
    name = f"{parts[-2]} {parts[-1]}"
    return name.lower()

def process_split():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    na_species_set = load_na_species(CSV_PATH)
    print(f"Loaded {len(na_species_set)} NA species from ABA CSV.")

    # Iterate source directories
    # Filter for directories only
    species_dirs = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir()])

    count_na = 0
    count_non_na = 0
    total_images = 0

    # Set seed for reproducibility
    random.seed(42)

    print(f"Found {len(species_dirs)} species directories in {SOURCE_DIR}")

    for i, species_dir in enumerate(species_dirs):
        try:
            folder_name = species_dir.name
            sc_name = get_species_name_from_folder(folder_name)

            if not sc_name:
                print(f"Skipping invalid folder format: {folder_name}")
                continue

            is_na = sc_name in na_species_set

            if is_na:
                dest_1 = NA_1
                dest_2 = NA_2
                count_na += 1
            else:
                dest_1 = NON_NA_1
                dest_2 = NON_NA_2
                count_non_na += 1

            # List valid image files (ignoring hidden files)
            images = sorted([f for f in species_dir.iterdir() if f.is_file() and not f.name.startswith('.')])

            # Shuffle images for random split
            random.shuffle(images)

            mid_point = len(images) // 2
            set_1 = images[:mid_point]
            set_2 = images[mid_point:]

            total_images += len(images)

            # Create species sub-directories in destinations
            # Keeping the original folder name
            (dest_1 / folder_name).mkdir(parents=True, exist_ok=True)
            (dest_2 / folder_name).mkdir(parents=True, exist_ok=True)

            # Create symlinks
            # Note: symlink_to(target) creates a link pointing TO target.
            # If target is absolute (which it is from iterdir on absolute path), it works fine.

            for img in set_1:
                link_path = dest_1 / folder_name / img.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(img)

            for img in set_2:
                link_path = dest_2 / folder_name / img.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(img)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1} / {len(species_dirs)} folders...")
        except Exception as e:
            print(f"Error processing {species_dir.name}: {e}")
            continue


    print("-" * 40)
    print(f"Processing Complete.")
    print(f"Total Species Processed: {len(species_dirs)}")
    print(f"Classified as NA: {count_na}")
    print(f"Classified as Non-NA: {count_non_na}")
    print(f"Total Images Processed: {total_images}")
    print(f"Output Directories:")
    print(f"  - {NA_1}")
    print(f"  - {NA_2}")
    print(f"  - {NON_NA_1}")
    print(f"  - {NON_NA_2}")

if __name__ == "__main__":
    setup_dirs()
    process_split()


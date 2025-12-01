import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize(96),          # short edge -> 96
    transforms.CenterCrop(96),      # center crop to 96x96
])

def process_one(args):
    in_path, data_dir, out_root = args

    # Compute relative path: e.g. hello/world/bird.jpg
    rel_path = in_path.relative_to(data_dir)
    # Target path: out_root / hello/world/bird.jpg
    out_path = out_root / rel_path

    # Make sure the parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: skip if already done
    if out_path.exists():
        return str(in_path), "exists"

    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            # After this transform:
            # - Resize: short side = 96, long side scaled to keep aspect ratio
            # - CenterCrop: image becomes exactly 96 x 96 pixels
            img = transform(img)
            img.save(out_path, format="JPEG", quality=90)
        return str(in_path), "ok"
    except Exception as e:
        print(f"[WARN] Failed on {in_path}: {e}")
        return str(in_path), "error"


def preprocess_images(image_paths, data_dir, out_dir, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    # Each job now also carries data_dir and out_dir so we can compute relative paths
    jobs = [(img_path, data_dir, out_dir) for img_path in image_paths]

    print(f"Processing {len(jobs)} images with {num_workers} workers...")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one, jobs),
            total=len(jobs),
            desc="Resizing images"
        ):
            results.append(res)

    # Print summary
    ok_count = sum(1 for _, status in results if status == "ok")
    exists_count = sum(1 for _, status in results if status == "exists")
    error_count = sum(1 for _, status in results if status == "error")

    print(f"\nSummary:")
    print(f"  Processed: {ok_count}")
    print(f"  Already existed: {exists_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Num workers: {args.num_workers}")

    # Find all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_paths = []

    print(f"\nScanning for images in {data_dir}...")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if Path(file).suffix in image_extensions:
                image_paths.append(Path(root) / file)

    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    # NOTE: pass data_dir so we can keep relative paths
    preprocess_images(image_paths, data_dir, output_dir, args.num_workers)

    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nDone. Elapsed time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
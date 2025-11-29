import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser

from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(96),          # short edge = 96
    transforms.CenterCrop(96),      # 96x96 center crop
])

def process_one(args):
    in_path, out_path = args

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: skip if already done
    if out_path.exists():
        return str(in_path), "exists"

    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            img = transform(img)  # torchvision resize + center crop
            img.save(out_path, format="JPEG", quality=90)
        return str(in_path), "ok"
    except Exception as e:
        print(f"[WARN] Failed on {in_path}: {e}")
        return str(in_path), "error"


def preprocess_images(image_paths, out_dir, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    jobs = [(img_path, out_dir / img_path.name) for img_path in image_paths]

    print(f"Processing {len(jobs)} images with {num_workers} workers...")
    os.makedirs(out_dir, exist_ok=True)

    with Pool(processes=num_workers) as pool:
        results = list(pool.imap_unordered(process_one, jobs))

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
    preprocess_images(image_paths, output_dir, args.num_workers)

    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nDone. Elapsed time: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    main()


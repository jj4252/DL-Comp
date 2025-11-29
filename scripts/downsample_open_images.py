import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser

import pandas as pd
from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize(96),          # short edge = 96
    transforms.CenterCrop(96),      # 96x96 center crop
])

def process_one(args):
    data_dir, out_dir, image_id = args
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    in_path = data_dir / f"{image_id}.jpg"
    out_path = out_dir / f"{image_id}.jpg"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional: skip if already done
    if out_path.exists():
        return image_id, "exists"

    try:
        with Image.open(in_path) as img:
            img = img.convert("RGB")
            img = transform(img)  # torchvision resize + center crop
            img.save(out_path, format="JPEG", quality=90)
        return image_id, "ok"
    except Exception as e:
        print(f"[WARN] Failed on {in_path}: {e}")
        return image_id, "error"


def preprocess_images(image_ids, data_dir, out_dir, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    jobs = [(data_dir, out_dir, img_id) for img_id in image_ids]

    print(f"Processing {len(jobs)} images with {num_workers} workers...")
    os.makedirs(out_dir, exist_ok=True)

    with Pool(processes=num_workers) as pool:
        for _ in pool.imap_unordered(process_one, jobs):
            pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--image_list_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    print(f"image list file: {args.image_list_file}")
    print(f"data dir: {args.data_dir}")
    print(f"output dir: {args.output_dir}")
    print(f"num workers: {args.num_workers}")

    df = pd.read_csv(args.image_list_file, header=None, sep='/')
    df.columns = ['split', 'image_id']

    image_ids = df['image_id'].tolist()

    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    preprocess_images(image_ids, args.data_dir, args.output_dir, args.num_workers)

    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Done. Elapsed time: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    main()

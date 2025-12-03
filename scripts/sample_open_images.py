import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Sample Open Images dataset")

    parser.add_argument("--start", type=int, required=True, help="Start index of the sample")
    parser.add_argument("--end", type=int, required=True, help="End index of the sample")


    args = parser.parse_args()

    if args.start < 1:
        parser.error("--start must be non-negative")

    if args.end <= args.start:
        parser.error("--end must be greater than --start")

    OI_data_dir = Path("data/Open_Images")
    mini_only_parquet_file = OI_data_dir / "mini_only_train_list.parquet"
    non_only_parquet_file = OI_data_dir / "non_only_train_list.parquet"
    mixed_parquet_file = OI_data_dir / "mixed_train_list.parquet"

    print("Reading mini_only, non_only, and mixed train parquet files...")
    mini_only_train_df = pd.read_parquet(mini_only_parquet_file)
    non_only_train_df = pd.read_parquet(non_only_parquet_file)
    mixed_train_df = pd.read_parquet(mixed_parquet_file)

    print(f"len(mini_only_train_df): {len(mini_only_train_df)}")
    print(f"len(non_only_train_df): {len(non_only_train_df)}")
    print(f"len(mixed_train_df): {len(mixed_train_df)}")

    assert args.end <= len(mini_only_train_df), f"End index ({args.end}) exceeds mini_only dataset size ({len(mini_only_train_df)})"
    assert args.end <= len(non_only_train_df), f"End index ({args.end}) exceeds non_only dataset size ({len(non_only_train_df)})"
    assert args.end <= len(mixed_train_df), f"End index ({args.end}) exceeds mixed dataset size ({len(mixed_train_df)})"


    print(f"Selecting from {args.start}th to {args.end}th rows for each dataframe...")

    s_mini_only_df = mini_only_train_df.iloc[args.start-1:args.end].copy()
    s_non_only_df = non_only_train_df.iloc[args.start-1:args.end].copy()
    s_mixed_df = mixed_train_df.iloc[args.start-1:args.end].copy()

    s_mini_only_df.drop(columns=["Labels"], inplace=True)
    s_non_only_df.drop(columns=["Labels"], inplace=True)
    s_mixed_df.drop(columns=["Labels"], inplace=True)

    file_suffix = f"{args.start}-{args.end}"

    s_mini_only_csv_file = OI_data_dir / f"mini_only_{file_suffix}.csv"
    s_non_only_csv_file = OI_data_dir / f"non_only_{file_suffix}.csv"
    s_mixed_csv_file = OI_data_dir / f"mixed_{file_suffix}.csv"

    print(f"Saving to {s_mini_only_csv_file}...")
    s_mini_only_df.to_csv(s_mini_only_csv_file, index=False)
    print(f"Saving to {s_non_only_csv_file}...")
    s_non_only_df.to_csv(s_non_only_csv_file, index=False)
    print(f"Saving to {s_mixed_csv_file}...")
    s_mixed_df.to_csv(s_mixed_csv_file, index=False)

    # need these files for downloading actual images using scripts/download_open_images.py
    mini_only_image_list_file = OI_data_dir / f"mini_only_{file_suffix}.txt"
    non_only_image_list_file = OI_data_dir / f"non_only_{file_suffix}.txt"
    mixed_image_list_file = OI_data_dir / f"mixed_{file_suffix}.txt"

    print(f"Writing image list to {mini_only_image_list_file}...")
    with open(mini_only_image_list_file, "w") as f:
        for image_id in s_mini_only_df["ImageID"]:
            f.write(f"train/{image_id}\n")
    print(f"Writing image list to {non_only_image_list_file}...")
    with open(non_only_image_list_file, "w") as f:
        for image_id in s_non_only_df["ImageID"]:
            f.write(f"train/{image_id}\n")
    print(f"Writing image list to {mixed_image_list_file}...")
    with open(mixed_image_list_file, "w") as f:
        for image_id in s_mixed_df["ImageID"]:
            f.write(f"train/{image_id}\n")

    print("Done!")


if __name__ == "__main__":
    main()

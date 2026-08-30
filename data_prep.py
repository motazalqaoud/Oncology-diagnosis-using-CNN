"""
One-time setup helper: verifies a downloaded HAM10000 archive is laid out the way
dataset.py expects, and reports basic dataset statistics.

The dataset is NOT bundled in this repo (it's ~2.7GB). Download it yourself:

    1. Go to https://doi.org/10.7910/DVN/DBW86T (Harvard Dataverse)
    2. Download: HAM10000_metadata.csv, HAM10000_images_part_1.zip, HAM10000_images_part_2.zip
    3. Unzip both image archives into the same parent folder as the metadata CSV, e.g.:

        HAM10000/
          HAM10000_metadata.csv
          HAM10000_images_part_1/
          HAM10000_images_part_2/

    4. Run:  python data_prep.py --data_dir HAM10000

Usage:
    python data_prep.py --data_dir /path/to/HAM10000
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found at {metadata_path}. "
            "Download the dataset from https://doi.org/10.7910/DVN/DBW86T first."
        )

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} rows")
    print(f"Unique lesions:  {df['lesion_id'].nunique()}")
    print()
    print("Class distribution:")
    print(df["dx"].value_counts())
    print()

    # Spot-check that images are actually reachable before a multi-hour training run
    # discovers a missing folder at epoch 0.
    sample_ids = df["image_id"].sample(min(20, len(df)), random_state=0).tolist()
    missing = []
    for image_id in sample_ids:
        found = False
        for candidate in [
            os.path.join(args.data_dir, f"{image_id}.jpg"),
            os.path.join(args.data_dir, "HAM10000_images_part_1", f"{image_id}.jpg"),
            os.path.join(args.data_dir, "HAM10000_images_part_2", f"{image_id}.jpg"),
            os.path.join(args.data_dir, "images", f"{image_id}.jpg"),
        ]:
            if os.path.exists(candidate):
                found = True
                break
        if not found:
            missing.append(image_id)

    if missing:
        print(f"WARNING: {len(missing)}/{len(sample_ids)} sampled images not found. "
              f"Missing example: {missing[0]}.jpg")
        print("Check that both image archives were unzipped into --data_dir.")
    else:
        print(f"OK: all {len(sample_ids)} sampled images found. Dataset looks ready.")
        print(f"\nNext step: python src/train.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()

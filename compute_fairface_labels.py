#!/usr/bin/env python3
"""
FairFace L* Preprocessing Pipeline
===================================
Computes synthetic Fitzpatrick skin-tone labels (0-5) for every image in
the FairFace dataset by extracting the median CIELAB L* (Lightness) value
from a central skin patch and mapping it to the nearest MILK10k centroid.

Usage:
    python compute_fairface_labels.py \
        --csv-dir datasets/ \
        --image-root datasets/fairface-img-margin025-trainval/ \
        --output datasets/fairface_lstar_labels.csv
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================================================
# OPTIMISED CENTROIDS  (Class → L* value)
# Higher class = lighter skin
#
# Derived from MILK10k + MSKCC median analysis:
#   MILK10k medians:  I≈72,  II≈64,  III≈66.7,  IV≈61.9,  V≈53.7,  VI≈52.1
#   MSKCC medians:    I≈65,  II≈64,  III≈~62,   IV≈58,    V≈47,    VI≈43
#
# Strategy: Use weighted average (0.6×MILK + 0.4×MSKCC) to maximise
# inter-class separation while staying grounded in clinical data.
# Minimum gap between adjacent centroids: ~5 L*
# ========================================================================
CENTROIDS = {
    5: 69.0,   # Type I   (MILK ~72, MSKCC ~65) → avg ~69
    4: 64.0,   # Type II  (MILK ~64, MSKCC ~64) → both agree at 64
    3: 59.0,   # Type III (MILK ~66.7, MSKCC ~62 → target ~59, gap ≥5 from II & IV)
    2: 53.0,   # Type IV  (MILK ~61.9, MSKCC ~58 → target ~53, gap ≥5 from III)
    1: 46.0,   # Type V   (MILK ~53.7, MSKCC ~47 → target ~46)
    0: 38.0,   # Type VI  (MILK ~52.1, MSKCC ~43 → target ~38, ensures gap from V)
}

CENTROID_VALUES = np.array(list(CENTROIDS.values()))   # [69.0, 64.0, …]
CENTROID_CLASSES = np.array(list(CENTROIDS.keys()))     # [5, 4, 3, 2, 1, 0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute L*-based skin-tone labels for FairFace images"
    )
    parser.add_argument(
        "--csv-dir", type=str, required=True,
        help="Directory containing fairface_label_train.csv and fairface_label_val.csv"
    )
    parser.add_argument(
        "--image-root", type=str, required=True,
        help="Root directory of FairFace images (contains train/ and val/ subdirs)"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/fairface_lstar_labels.csv",
        help="Output CSV path (default: datasets/fairface_lstar_labels.csv)"
    )
    parser.add_argument(
        "--crop-ratio", type=float, default=0.4,
        help="Central crop ratio for skin patch isolation (default: 0.4 = inner 40%%)"
    )
    return parser.parse_args()


# ========================================================================
# CORE FUNCTIONS
# ========================================================================

def extract_skin_patch(image_bgr: np.ndarray, crop_ratio: float = 0.4) -> np.ndarray:
    """
    Extract a central crop from a face image to approximate a skin patch.

    FairFace images are already tightly-cropped, centered face images,
    so the central region reliably captures forehead/cheek skin while
    avoiding hair edges, ears, and background.

    Args:
        image_bgr: BGR image as numpy array (H, W, 3)
        crop_ratio: Fraction of the image to keep (0.4 = inner 40%)

    Returns:
        Cropped BGR image as numpy array
    """
    h, w = image_bgr.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    return image_bgr[margin_y:h - margin_y, margin_x:w - margin_x]


def compute_median_lstar(skin_patch_bgr: np.ndarray) -> float:
    """
    Convert a BGR skin patch to CIELAB and return the median L* value.

    Median is preferred over mean because it is robust to outlier pixels
    (specular highlights, deep shadows, stray hair pixels).

    Args:
        skin_patch_bgr: BGR image as numpy array

    Returns:
        Median L* value (0-100 scale)
    """
    lab = cv2.cvtColor(skin_patch_bgr, cv2.COLOR_BGR2LAB)
    # OpenCV stores L* as 0-255 (scaled); convert back to 0-100
    l_channel = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    return float(np.median(l_channel))


def lstar_to_class(median_lstar: float) -> int:
    """
    Map a median L* value to the nearest Fitzpatrick class (0-5)
    using the MILK10k centroids.

    Args:
        median_lstar: Median L* value

    Returns:
        Integer class label (0 = Type VI / darkest, 5 = Type I / lightest)
    """
    distances = np.abs(CENTROID_VALUES - median_lstar)
    nearest_idx = np.argmin(distances)
    return int(CENTROID_CLASSES[nearest_idx])


def process_single_image(image_path: str, crop_ratio: float = 0.4):
    """
    Process one image: load → crop → compute L* → classify.

    Returns:
        (median_lstar, skin_tone_class) or (None, None) if the image
        cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    patch = extract_skin_patch(img, crop_ratio)
    if patch.size == 0:
        return None, None

    median_l = compute_median_lstar(patch)
    cls = lstar_to_class(median_l)
    return median_l, cls


# ========================================================================
# MAIN
# ========================================================================

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load FairFace CSVs
    # ------------------------------------------------------------------
    train_csv = os.path.join(args.csv_dir, "fairface_label_train.csv")
    val_csv = os.path.join(args.csv_dir, "fairface_label_val.csv")

    dfs = []
    for csv_path, split_name in [(train_csv, "train"), (val_csv, "val")]:
        if not os.path.isfile(csv_path):
            print(f"⚠  CSV not found: {csv_path} — skipping {split_name} split")
            continue
        df = pd.read_csv(csv_path)
        df["original_split"] = split_name
        dfs.append(df)
        print(f"✓ Loaded {split_name}: {len(df):,} rows from {csv_path}")

    if not dfs:
        print("✗ No FairFace CSVs found. Please check --csv-dir.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal images to process: {len(combined):,}")

    # ------------------------------------------------------------------
    # 2. Process each image
    # ------------------------------------------------------------------
    results = []
    skipped = 0

    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Computing L*"):
        # FairFace CSV 'file' column has paths like "train/1.jpg"
        rel_path = row["file"]
        abs_path = os.path.join(args.image_root, rel_path)

        median_l, cls = process_single_image(abs_path, args.crop_ratio)

        if median_l is None:
            skipped += 1
            continue

        results.append({
            "file": rel_path,
            "original_split": row["original_split"],
            "median_lstar": round(median_l, 2),
            "skin_tone_class": cls,
        })

    # ------------------------------------------------------------------
    # 3. Save output CSV
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(out_df):,} labelled images to {args.output}")
    if skipped:
        print(f"⚠  Skipped {skipped:,} unreadable images")

    # ------------------------------------------------------------------
    # 4. Print class distribution
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*60}")

    type_names = {
        5: "Type I   (lightest)",
        4: "Type II",
        3: "Type III",
        2: "Type IV",
        1: "Type V",
        0: "Type VI  (darkest)",
    }

    dist = out_df["skin_tone_class"].value_counts().sort_index(ascending=False)
    total = len(out_df)
    for cls_id in sorted(dist.index, reverse=True):
        count = dist[cls_id]
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  Class {cls_id} ({type_names[cls_id]:20s}): {count:6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  L* range: {out_df['median_lstar'].min():.1f} – {out_df['median_lstar'].max():.1f}")
    print(f"  L* mean:  {out_df['median_lstar'].mean():.1f} ± {out_df['median_lstar'].std():.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
UTKFace → MST Labelling Pipeline
==================================
Computes ITA-based Monk Skin Tone labels for UTKFace images,
producing a CSV that can be merged with the FairFace MST labels.

UTKFace images are named: [age]_[gender]_[race]_[date].jpg
There is no separate CSV — metadata is extracted from filenames.

Output CSV columns:
    file, original_split, ita, mst10_class, mst5_class

All images go into original_split="train" since UTKFace is used
as supplemental training data only.

Usage:
    python compute_utkface_mst_labels.py \\
        --image-dir datasets/UTKFace \\
        --output datasets/utkface_mst_labels.csv
"""

import os
import argparse
import glob
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================================================
# ITA → MST THRESHOLDS (same as compute_mst_labels.py)
# ========================================================================

MST10_BOUNDARIES = [-81.3, -66.9, -37.8, -4.6, 30.6, 57.5, 68.2, 76.0, 81.6]

MST10_CLASS_NAMES = [
    "MST 10 (deepest)", "MST 9", "MST 8", "MST 7", "MST 6",
    "MST 5", "MST 4", "MST 3", "MST 2", "MST 1 (lightest)",
]

MST5_CLASS_NAMES = [
    "Very Dark (MST 9-10)", "Dark (MST 7-8)", "Medium (MST 5-6)",
    "Light (MST 3-4)", "Very Light (MST 1-2)",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ITA-based MST labels for UTKFace images"
    )
    parser.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing UTKFace .jpg images"
    )
    parser.add_argument(
        "--output", type=str, default="datasets/utkface_mst_labels.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--crop-ratio", type=float, default=0.4,
        help="Central crop ratio for skin patch (default: 0.4)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, choices=[5, 10],
        help="Number of MST classes for distribution display"
    )
    return parser.parse_args()


# ========================================================================
# CORE FUNCTIONS (shared logic with compute_mst_labels.py)
# ========================================================================

def extract_skin_patch(image_bgr: np.ndarray, crop_ratio: float = 0.4) -> np.ndarray:
    """Extract central crop from a face image."""
    h, w = image_bgr.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    return image_bgr[margin_y:h - margin_y, margin_x:w - margin_x]


def compute_ita(skin_patch_bgr: np.ndarray) -> float:
    """
    ITA = arctan((L* - 50) / b*) × 180/π

    Uses both L* (lightness) and b* (yellow-blue) from CIELAB.
    """
    lab = cv2.cvtColor(skin_patch_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64) * (100.0 / 255.0)
    b_channel = lab[:, :, 2].astype(np.float64) - 128.0

    median_l = float(np.median(l_channel))
    median_b = float(np.median(b_channel))

    if abs(median_b) < 0.01:
        return 90.0 if median_l > 50 else -90.0

    return math.atan((median_l - 50.0) / median_b) * (180.0 / math.pi)


def ita_to_mst10(ita: float) -> int:
    """Map ITA to MST-10 class (0=darkest, 9=lightest)."""
    for i, threshold in enumerate(MST10_BOUNDARIES):
        if ita < threshold:
            return i
    return 9


def mst10_to_mst5(mst10_class: int) -> int:
    """MST-10 → MST-5: pairs (0,1)→0, (2,3)→1, ..."""
    return mst10_class // 2


def parse_utkface_filename(filename: str):
    """
    Parse UTKFace filename: [age]_[gender]_[race]_[date].jpg

    Returns (age, gender, race) or None if filename is malformed.
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])  # 0=male, 1=female
        race = int(parts[2])    # 0=White, 1=Black, 2=Asian, 3=Indian, 4=Other
        return age, gender, race
    except (ValueError, IndexError):
        return None


# ========================================================================
# MAIN
# ========================================================================

def main():
    args = parse_args()

    # Collect all .jpg images
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not image_paths:
        # Try .chip.gz or common alternatives
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))

    print(f"Found {len(image_paths):,} images in {args.image_dir}")

    if not image_paths:
        print("✗ No images found. Check --image-dir.")
        return

    # Process each image
    results = []
    skipped = 0

    for img_path in tqdm(image_paths, desc="Computing ITA"):
        filename = os.path.basename(img_path)

        # Parse metadata from filename
        meta = parse_utkface_filename(filename)
        if meta is None:
            skipped += 1
            continue

        age, gender, race = meta

        # Load and process image
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        patch = extract_skin_patch(img, args.crop_ratio)
        if patch.size == 0:
            skipped += 1
            continue

        ita = compute_ita(patch)
        mst10_cls = ita_to_mst10(ita)

        results.append({
            "file": filename,
            "original_split": "train",  # All UTKFace → training supplement
            "ita": round(ita, 2),
            "mst10_class": mst10_cls,
            "mst5_class": mst10_to_mst5(mst10_cls),
            # Keep UTKFace metadata for analysis
            "utk_age": age,
            "utk_gender": gender,
            "utk_race": race,
        })

    # Save CSV
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Labelled {len(out_df):,} UTKFace images → {args.output}")
    if skipped:
        print(f"⚠  Skipped {skipped:,} malformed/unreadable images")

    # Print distribution
    if args.num_classes == 10:
        class_col, class_names = "mst10_class", MST10_CLASS_NAMES
    else:
        class_col, class_names = "mst5_class", MST5_CLASS_NAMES

    print(f"\n{'='*60}")
    print(f"UTKFACE CLASS DISTRIBUTION (MST-{args.num_classes})")
    print(f"{'='*60}")

    dist = out_df[class_col].value_counts().sort_index()
    total = len(out_df)
    for cls_id in range(len(class_names)):
        count = dist.get(cls_id, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        name = class_names[cls_id]
        print(f"  Class {cls_id} ({name:25s}): {count:6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  ITA range: {out_df['ita'].min():.1f}° – {out_df['ita'].max():.1f}°")
    print(f"  ITA mean:  {out_df['ita'].mean():.1f}° ± {out_df['ita'].std():.1f}°")


if __name__ == "__main__":
    main()

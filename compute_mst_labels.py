#!/usr/bin/env python3
"""
MST (Monk Skin Tone) Labelling Pipeline
=========================================
Computes ITA-based skin tone labels for every image in the FairFace
dataset, mapping each image to one of 10 MST categories (or 5 grouped).

ITA (Individual Typology Angle) uses both L* (lightness) and b*
(yellow-blue) from CIELAB colour space, giving much better separation
in medium tones than L* alone.

    ITA = arctan((L* - 50) / b*) × 180/π

MST thresholds from published literature:
    Monk 1:  ITA > 81.6°   (very light)
    Monk 10: ITA < -81.3°  (deepest)

Usage:
    python compute_mst_labels.py \\
        --csv-dir datasets/ \\
        --image-root datasets/fairface-img-margin025-trainval/ \\
        --output datasets/fairface_mst_labels.csv

    # Verify distribution:
    python compute_mst_labels.py ... --num-classes 5   # MST-5 mode
    python compute_mst_labels.py ... --num-classes 10  # MST-10 mode
"""

import os
import argparse
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================================================
# ITA → MST-10 THRESHOLDS (Published)
#
# Source: "Beyond Fitzpatrick" and related literature.
# ITA = arctan((L* - 50) / b*) × 180/π
#
# Each threshold is the midpoint between adjacent MST reference tones.
# Class 0 = MST 10 (deepest), Class 9 = MST 1 (lightest)
# — stored dark-to-light for consistency with Fitzpatrick ordering.
# ========================================================================

# Boundaries between adjacent MST classes (ITA degrees, ascending)
# MST 10 | -81.3 | MST 9 | -66.9 | MST 8 | -37.8 | MST 7 | -4.6 | MST 6 | 30.6 | MST 5 | 57.5 | MST 4 | 68.2 | MST 3 | 76.0 | MST 2 | 81.6 | MST 1
MST10_BOUNDARIES = [-81.3, -66.9, -37.8, -4.6, 30.6, 57.5, 68.2, 76.0, 81.6]

# MST-5 boundaries: group pairs (1-2, 3-4, 5-6, 7-8, 9-10)
# Reuse the even-indexed boundaries from MST-10
MST5_BOUNDARIES = [-66.9, -4.6, 57.5, 76.0]

MST10_CLASS_NAMES = [
    "MST 10 (deepest)",    # class 0
    "MST 9",               # class 1
    "MST 8",               # class 2
    "MST 7",               # class 3
    "MST 6",               # class 4
    "MST 5",               # class 5
    "MST 4",               # class 6
    "MST 3",               # class 7
    "MST 2",               # class 8
    "MST 1 (lightest)",    # class 9
]

MST5_CLASS_NAMES = [
    "Very Dark (MST 9-10)",    # class 0
    "Dark (MST 7-8)",          # class 1
    "Medium (MST 5-6)",        # class 2
    "Light (MST 3-4)",         # class 3
    "Very Light (MST 1-2)",    # class 4
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ITA-based MST skin-tone labels for FairFace images"
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
        "--output", type=str, default="datasets/fairface_mst_labels.csv",
        help="Output CSV path (default: datasets/fairface_mst_labels.csv)"
    )
    parser.add_argument(
        "--crop-ratio", type=float, default=0.4,
        help="Central crop ratio for skin patch isolation (default: 0.4 = inner 40%%)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=10, choices=[5, 10],
        help="Number of MST classes for distribution display (labels CSV always stores MST-10)"
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
    """
    h, w = image_bgr.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    return image_bgr[margin_y:h - margin_y, margin_x:w - margin_x]


def compute_ita(skin_patch_bgr: np.ndarray) -> float:
    """
    Compute the Individual Typology Angle (ITA) from a BGR skin patch.

    ITA = arctan((L* - 50) / b*) × 180/π

    where L* is lightness (0-100) and b* is yellow-blue axis from CIELAB.

    ITA ranges from ~+90° (very light skin) to ~-90° (very dark skin).
    Unlike L* alone, ITA captures the yellow-blue dimension which varies
    significantly in medium skin tones, providing better separation.

    Args:
        skin_patch_bgr: BGR image as numpy array

    Returns:
        ITA value in degrees
    """
    # Convert to CIELAB
    lab = cv2.cvtColor(skin_patch_bgr, cv2.COLOR_BGR2LAB)

    # OpenCV stores L* as 0-255, a* and b* as 0-255 (centred at 128)
    l_channel = lab[:, :, 0].astype(np.float64) * (100.0 / 255.0)
    b_channel = lab[:, :, 2].astype(np.float64) - 128.0  # centre at 0

    # Use median of each channel (robust to outlier pixels)
    median_l = float(np.median(l_channel))
    median_b = float(np.median(b_channel))

    # Avoid division by zero when b* ≈ 0
    if abs(median_b) < 0.01:
        # When b* is near zero, ITA is ±90° depending on L*
        return 90.0 if median_l > 50 else -90.0

    ita = math.atan((median_l - 50.0) / median_b) * (180.0 / math.pi)
    return ita


def ita_to_mst10(ita: float) -> int:
    """
    Map an ITA value to MST-10 class (0-9).

    Class 0 = MST 10 (deepest), Class 9 = MST 1 (lightest).
    This dark-to-light ordering is consistent with our Fitzpatrick convention.
    """
    for i, threshold in enumerate(MST10_BOUNDARIES):
        if ita < threshold:
            return i
    return 9  # Above all thresholds → MST 1 (lightest)


def mst10_to_mst5(mst10_class: int) -> int:
    """
    Convert MST-10 class (0-9) to MST-5 class (0-4).

    Pairs: (0,1)→0, (2,3)→1, (4,5)→2, (6,7)→3, (8,9)→4
    """
    return mst10_class // 2


def process_single_image(image_path: str, crop_ratio: float = 0.4):
    """
    Process one image: load → crop → compute ITA → classify.

    Returns:
        (ita, mst10_class) or (None, None) if the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    patch = extract_skin_patch(img, crop_ratio)
    if patch.size == 0:
        return None, None

    ita = compute_ita(patch)
    cls = ita_to_mst10(ita)
    return ita, cls


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

    for _, row in tqdm(combined.iterrows(), total=len(combined), desc="Computing ITA"):
        rel_path = row["file"]
        abs_path = os.path.join(args.image_root, rel_path)

        ita, mst10_cls = process_single_image(abs_path, args.crop_ratio)

        if ita is None:
            skipped += 1
            continue

        results.append({
            "file": rel_path,
            "original_split": row["original_split"],
            "ita": round(ita, 2),
            "mst10_class": mst10_cls,      # Always store MST-10
            "mst5_class": mst10_to_mst5(mst10_cls),  # Pre-compute MST-5
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
    if args.num_classes == 10:
        class_col = "mst10_class"
        class_names = MST10_CLASS_NAMES
    else:
        class_col = "mst5_class"
        class_names = MST5_CLASS_NAMES

    num_classes = len(class_names)

    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION (MST-{args.num_classes})")
    print(f"{'='*60}")

    dist = out_df[class_col].value_counts().sort_index()
    total = len(out_df)
    for cls_id in range(num_classes):
        count = dist.get(cls_id, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        name = class_names[cls_id]
        print(f"  Class {cls_id} ({name:25s}): {count:6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  ITA range: {out_df['ita'].min():.1f}° – {out_df['ita'].max():.1f}°")
    print(f"  ITA mean:  {out_df['ita'].mean():.1f}° ± {out_df['ita'].std():.1f}°")


if __name__ == "__main__":
    main()

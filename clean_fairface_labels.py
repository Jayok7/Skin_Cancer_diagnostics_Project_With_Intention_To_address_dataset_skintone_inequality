#!/usr/bin/env python3
"""
FairFace Label Cleaning Script
===============================
Flags and removes training images with suspicious ITA labels that are
likely mislabelled due to pipeline artefacts.

Three detection criteria:
  1. Saturated ITA — ITA near ±90° (boundary artefact from b* ≈ 0)
  2. Monochrome — very low colour saturation in HSV
  3. ITA-vs-luminance mismatch — ITA class contradicts raw L*

Usage:
    python clean_fairface_labels.py \\
        --csv datasets/master_mst_labels.csv \\
        --image-root datasets/ \\
        --output datasets/master_mst_labels_cleaned.csv \\
        --flagged-output datasets/flagged_images.csv
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── MST-5 class names (dark → light) ──
MST5_NAMES = [
    "Very Dark (MST 9-10)",
    "Dark (MST 7-8)",
    "Medium (MST 5-6)",
    "Light (MST 3-4)",
    "Very Light (MST 1-2)",
]


def parse_args():
    p = argparse.ArgumentParser(description="Flag & clean noisy ITA labels")
    p.add_argument("--csv", type=str, default="datasets/master_mst_labels.csv",
                    help="Input CSV with ITA labels")
    p.add_argument("--image-root", type=str, default="datasets/",
                    help="Root directory for loading images")
    p.add_argument("--output", type=str,
                    default="datasets/master_mst_labels_cleaned.csv",
                    help="Output cleaned CSV (flagged rows removed)")
    p.add_argument("--flagged-output", type=str,
                    default="datasets/flagged_images.csv",
                    help="Output CSV containing only flagged images")
    p.add_argument("--ita-threshold", type=float, default=85.0,
                    help="Flag images with |ITA| > this value (default: 85°)")
    p.add_argument("--saturation-threshold", type=float, default=20.0,
                    help="Flag images with mean HSV saturation < this (default: 20)")
    p.add_argument("--dry-run", action="store_true",
                    help="Print summary without writing output files")
    return p.parse_args()


# ========================================================================
# DETECTION FUNCTIONS
# ========================================================================

def check_saturated_ita(row, ita_threshold=85.0):
    """
    Flag 1: Saturated ITA values near ±90°.
    
    When the b* channel (yellow-blue) is near zero, the ITA formula
    arctan((L*-50)/b*) produces extreme values that don't reflect
    actual skin tone. These boundary artefacts push images to the
    wrong extreme of the scale.
    """
    if abs(row["ita"]) > ita_threshold:
        return f"saturated_ita (ITA={row['ita']:.1f}°)"
    return None


def check_monochrome(image_bgr, saturation_threshold=20.0):
    """
    Flag 2: Monochrome / heavily colour-shifted images.
    
    Images with very low colour saturation (near greyscale, or strong
    blue/green/sepia tints) produce unreliable ITA values because the
    b* channel carries the colour signal that ITA depends on.
    
    We measure mean saturation in HSV: if < threshold, the image is
    effectively monochrome and ITA is unreliable.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat < saturation_threshold:
        return f"monochrome (saturation={mean_sat:.1f})"
    return None


def check_ita_luminance_mismatch(row, image_bgr, crop_ratio=0.4):
    """
    Flag 3: ITA class contradicts raw luminance.
    
    If the ITA pipeline says "Very Dark" but the central skin patch
    has L* > 60 (clearly not dark), or says "Very Light" but L* < 40,
    the label is likely wrong.
    
    This catches cases where the b* distortion pushed a mid-skin-tone
    image to an incorrect extreme class.
    """
    # Extract central skin patch (same as compute_mst_labels.py)
    h, w = image_bgr.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    patch = image_bgr[margin_y:h - margin_y, margin_x:w - margin_x]

    if patch.size == 0:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    median_l = float(np.median(lab[:, :, 0])) * (100.0 / 255.0)

    mst5 = int(row["mst5_class"])

    # Very Dark (class 0) but luminance clearly light
    if mst5 == 0 and median_l > 60:
        return f"ita_luminance_mismatch (VeryDark but L*={median_l:.1f})"

    # Very Light (class 4) but luminance clearly dark
    if mst5 == 4 and median_l < 40:
        return f"ita_luminance_mismatch (VeryLight but L*={median_l:.1f})"

    # Dark (class 1) but luminance very light
    if mst5 == 1 and median_l > 70:
        return f"ita_luminance_mismatch (Dark but L*={median_l:.1f})"

    # Light (class 3) but luminance very dark
    if mst5 == 3 and median_l < 30:
        return f"ita_luminance_mismatch (Light but L*={median_l:.1f})"

    return None


def resolve_image_path(file_col, image_root):
    """Try multiple path resolutions for both CSF and local layouts."""
    import re
    full = os.path.join(image_root, file_col)
    if os.path.isfile(full):
        return full
    stripped = re.sub(r'^fairface-img-margin025-trainval/', '', file_col)
    alt = os.path.join(image_root, stripped)
    if os.path.isfile(alt):
        return alt
    return None


# ========================================================================
# MAIN
# ========================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("FairFace Label Cleaning")
    print("=" * 60)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} images from {args.csv}")

    # Detect the source column — if missing, infer from file path
    if "source" not in df.columns:
        def infer_source(path):
            if "utkface" in path.lower() or "UTKFace" in path:
                return "utkface"
            return "fairface"
        df["source"] = df["file"].apply(infer_source)
        print("  ⚠ Inferred 'source' column from file paths")

    print(f"\n  Dataset sources:")
    for src, count in df["source"].value_counts().items():
        print(f"    {src}: {count:,}")

    print(f"\n  Class distribution (before cleaning):")
    for cls_id in range(5):
        count = (df["mst5_class"] == cls_id).sum()
        print(f"    {MST5_NAMES[cls_id]:25s}: {count:,}")

    # ── Run all checks ──
    flags = []
    images_checked = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
        reasons = []

        # Check 1: Saturated ITA (no image needed)
        flag = check_saturated_ita(row, args.ita_threshold)
        if flag:
            reasons.append(flag)

        # Checks 2 & 3 require loading the image
        img_path = resolve_image_path(row["file"], args.image_root)
        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                images_checked += 1

                flag = check_monochrome(img, args.saturation_threshold)
                if flag:
                    reasons.append(flag)

                flag = check_ita_luminance_mismatch(row, img)
                if flag:
                    reasons.append(flag)

        if reasons:
            flags.append({
                "index": idx,
                "file": row["file"],
                "source": row.get("source", "unknown"),
                "original_split": row["original_split"],
                "ita": row["ita"],
                "mst5_class": int(row["mst5_class"]),
                "mst5_name": MST5_NAMES[int(row["mst5_class"])],
                "flag_reasons": " | ".join(reasons),
            })

    # ── Report ──
    print(f"\n{'=' * 60}")
    print(f"CLEANING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total images scanned:  {len(df):,}")
    print(f"  Images loaded for QC:  {images_checked:,}")
    print(f"  Images flagged:        {len(flags):,} ({len(flags)/len(df)*100:.2f}%)")

    if flags:
        flagged_df = pd.DataFrame(flags)

        # Breakdown by reason
        print(f"\n  Breakdown by flag type:")
        for reason_type in ["saturated_ita", "monochrome", "ita_luminance_mismatch"]:
            count = flagged_df["flag_reasons"].str.contains(reason_type).sum()
            print(f"    {reason_type:30s}: {count:,}")

        # Breakdown by class
        print(f"\n  Breakdown by class:")
        for cls_id in range(5):
            count = (flagged_df["mst5_class"] == cls_id).sum()
            print(f"    {MST5_NAMES[cls_id]:25s}: {count:,}")

        # Breakdown by dataset source
        print(f"\n  Breakdown by dataset source:")
        for src, count in flagged_df["source"].value_counts().items():
            print(f"    {src:25s}: {count:,}")

        # Breakdown by split
        print(f"\n  Breakdown by split:")
        for split, count in flagged_df["original_split"].value_counts().items():
            print(f"    {split:25s}: {count:,}")

        if not args.dry_run:
            # Save flagged images
            flagged_df.to_csv(args.flagged_output, index=False)
            print(f"\n  ✓ Flagged images saved to {args.flagged_output}")

            # Create cleaned CSV
            flagged_indices = set(flagged_df["index"])
            cleaned_df = df[~df.index.isin(flagged_indices)].reset_index(drop=True)
            cleaned_df.to_csv(args.output, index=False)
            print(f"  ✓ Cleaned CSV saved to {args.output}")
            print(f"    {len(df):,} → {len(cleaned_df):,} images "
                  f"({len(flagged_indices):,} removed)")

            print(f"\n  Class distribution (after cleaning):")
            for cls_id in range(5):
                before = (df["mst5_class"] == cls_id).sum()
                after = (cleaned_df["mst5_class"] == cls_id).sum()
                diff = before - after
                print(f"    {MST5_NAMES[cls_id]:25s}: {before:,} → {after:,}  "
                      f"(-{diff})")
        else:
            print("\n  [DRY RUN] No files written.")
    else:
        print("  No images flagged — dataset is clean!")


if __name__ == "__main__":
    main()

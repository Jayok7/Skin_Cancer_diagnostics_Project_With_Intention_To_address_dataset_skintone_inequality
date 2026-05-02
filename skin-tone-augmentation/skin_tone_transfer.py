#!/usr/bin/env python3
"""
Skin Tone Transfer: Reinhard LAB Colour Transfer
====================================================
Transfers skin tone across dermoscopic/clinical images using Reinhard's
method in CIELAB colour space. Designed for dataset augmentation to
balance skin tone distributions.

The transfer shifts the entire image (skin + lesion) to match the colour
statistics of a target class. This is intentional: lesion appearance
varies with skin tone, so the lesion should transform too.


  1. Load a predictions CSV with skin tone labels per image
  2. Compute per-class colour statistics (mean/std in LAB space)
  3. Transfer images from majority classes → minority classes
  4. Output augmented images + updated metadata CSV

Usage:
    # Transfer Light images → Dark skin tone
    python skin_tone_transfer.py \
        --predictions datasets/isic2020_skin_tone_predictions.csv \
        --image-root datasets/ISIC-2020-images/ \
        --output-dir datasets/ISIC-2020-augmented/ \
        --source-class Light \
        --target-class Dark \
        --num-images 500

    # Auto-balance all classes to match the largest class
    python skin_tone_transfer.py \
        --predictions datasets/isic2020_skin_tone_predictions.csv \
        --image-root datasets/ISIC-2020-images/ \
        --output-dir datasets/ISIC-2020-augmented/ \
        --auto-balance

    # Preview mode: generate 10 sample transfers for visual QA
    python skin_tone_transfer.py \
        --predictions datasets/isic2020_skin_tone_predictions.csv \
        --image-root datasets/ISIC-2020-images/ \
        --output-dir datasets/ISIC-2020-augmented/ \
        --source-class Light \
        --target-class Dark \
        --preview 10
"""

import os
import sys
import argparse
import time
import random

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# CLASS DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# Must match the 3-class labels from classify_isic2020.py
CLASS_MAP = {
    "Dark":   0,  # Dark (MST 7-10)
    "Medium": 1,  # Medium (MST 3-6)
    "Light":  2,  # Light (MST 1-2)
}

CLASS_NAMES = {
    0: "Dark (MST 7-10)",
    1: "Medium (MST 3-6)",
    2: "Light (MST 1-2)",
}


# ═══════════════════════════════════════════════════════════════
# REINHARD COLOUR TRANSFER
# ═══════════════════════════════════════════════════════════════

def compute_lab_stats(image_bgr):
    """
    Compute per-channel mean and std in CIELAB space.

    Returns:
        dict with keys: L_mean, L_std, a_mean, a_std, b_mean, b_std
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return {
        "L_mean": lab[:, :, 0].mean(),
        "L_std":  lab[:, :, 0].std(),
        "a_mean": lab[:, :, 1].mean(),
        "a_std":  lab[:, :, 1].std(),
        "b_mean": lab[:, :, 2].mean(),
        "b_std":  lab[:, :, 2].std(),
    }


def reinhard_transfer(source_bgr, target_stats, source_stats=None):
    """
    Apply Reinhard colour transfer to shift source image towards
    the target colour distribution.

    Args:
        source_bgr:    Source image (BGR, uint8)
        target_stats:  Dict with L/a/b mean+std of target class
        source_stats:  Optional pre-computed source stats (saves time
                       when applying the same source to multiple targets).
                       If None, computed from source_bgr.

    Returns:
        Transferred image (BGR, uint8)
    """
    lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if source_stats is None:
        source_stats = compute_lab_stats(source_bgr)

    # Transfer each channel: shift mean, scale std
    for ch, ch_name in enumerate(["L", "a", "b"]):
        s_mean = source_stats[f"{ch_name}_mean"]
        s_std  = source_stats[f"{ch_name}_std"]
        t_mean = target_stats[f"{ch_name}_mean"]
        t_std  = target_stats[f"{ch_name}_std"]

        # Normalise → rescale → shift
        lab[:, :, ch] = (lab[:, :, ch] - s_mean) * (t_std / max(s_std, 1e-6)) + t_mean

    # Clip to valid LAB range and convert back
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result


def reinhard_transfer_multi_ref(source_bgr, reference_images, n_refs=5):
    """
    Transfer colour using averaged stats from multiple reference images.
    More robust than single-image transfer - averages out per-image noise.

    Args:
        source_bgr:       Source image (BGR, uint8)
        reference_images: List of reference image paths
        n_refs:           How many references to sample and average

    Returns:
        Transferred image (BGR, uint8)
    """
    # Sample references
    refs = random.sample(reference_images, min(n_refs, len(reference_images)))

    # Average their LAB stats
    all_stats = []
    for ref_path in refs:
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            all_stats.append(compute_lab_stats(ref_img))

    if not all_stats:
        return source_bgr  # Fallback: return unchanged

    avg_stats = {}
    for key in all_stats[0]:
        avg_stats[key] = np.mean([s[key] for s in all_stats])

    return reinhard_transfer(source_bgr, avg_stats)


# ═══════════════════════════════════════════════════════════════
# CLASS-LEVEL STATISTICS
# ═══════════════════════════════════════════════════════════════

def compute_class_stats(predictions_csv, image_root, image_ext=".jpg",
                        max_samples_per_class=200):
    """
    Compute average LAB statistics for each predicted class.

    Samples up to max_samples_per_class images per class to avoid
    loading the entire dataset into memory.

    Returns:
        dict: {class_id: {L_mean, L_std, a_mean, a_std, b_mean, b_std}}
    """
    df = pd.read_csv(predictions_csv)
    class_stats = {}

    for cls_id in sorted(df["predicted_class"].unique()):
        cls_df = df[df["predicted_class"] == cls_id]
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

        # Sample subset
        sample_df = cls_df.sample(
            n=min(max_samples_per_class, len(cls_df)),
            random_state=42,
        )

        stats_list = []
        for _, row in sample_df.iterrows():
            img_path = os.path.join(image_root, f"{row['isic_id']}{image_ext}")
            img = cv2.imread(img_path)
            if img is not None:
                stats_list.append(compute_lab_stats(img))

        if stats_list:
            avg = {}
            for key in stats_list[0]:
                avg[key] = np.mean([s[key] for s in stats_list])
            class_stats[cls_id] = avg
            print(f"  {cls_name}: L*={avg['L_mean']:.1f}±{avg['L_std']:.1f}, "
                  f"a*={avg['a_mean']:.1f}±{avg['a_std']:.1f}, "
                  f"b*={avg['b_mean']:.1f}±{avg['b_std']:.1f}  "
                  f"(n={len(stats_list)})")
        else:
            print(f"  {cls_name}: ⚠ No images found!")

    return class_stats


# ═══════════════════════════════════════════════════════════════
# TRANSFER ENGINE
# ═══════════════════════════════════════════════════════════════

def transfer_batch(predictions_csv, image_root, output_dir,
                   source_class_id, target_class_id, target_stats,
                   num_images=None, image_ext=".jpg",
                   min_confidence=0.0):
    """
    Transfer a batch of images from source_class to target_class tone.

    Args:
        predictions_csv:  CSV with isic_id, predicted_class, confidence
        image_root:       Directory containing source images
        output_dir:       Directory to save transferred images
        source_class_id:  Class ID to pull source images from
        target_class_id:  Class ID whose colour stats to apply
        target_stats:     Pre-computed LAB stats for target class
        num_images:       How many images to transfer (None = all)
        image_ext:        File extension
        min_confidence:   Only transfer images with confidence >= this

    Returns:
        List of dicts with metadata for the transferred images
    """
    df = pd.read_csv(predictions_csv)

    # Filter to source class with confidence threshold
    source_df = df[
        (df["predicted_class"] == source_class_id) &
        (df["confidence"] >= min_confidence)
    ]

    if num_images is not None:
        source_df = source_df.sample(
            n=min(num_images, len(source_df)),
            random_state=42,
        )

    os.makedirs(output_dir, exist_ok=True)

    src_name = CLASS_NAMES.get(source_class_id, f"Class {source_class_id}")
    tgt_name = CLASS_NAMES.get(target_class_id, f"Class {target_class_id}")
    print(f"\n  Transferring {len(source_df)} images: {src_name} → {tgt_name}")

    results = []
    t0 = time.time()

    for i, (_, row) in enumerate(source_df.iterrows()):
        isic_id = row["isic_id"]
        img_path = os.path.join(image_root, f"{isic_id}{image_ext}")

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Apply transfer
        transferred = reinhard_transfer(img, target_stats)

        # Save with clear naming: {original_id}_transferred_{target}.jpg
        tgt_short = tgt_name.split("(")[0].strip().lower().replace(" ", "")
        out_filename = f"{isic_id}_to_{tgt_short}{image_ext}"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, transferred)

        results.append({
            "isic_id": isic_id,
            "original_class": source_class_id,
            "original_label": src_name,
            "transferred_class": target_class_id,
            "transferred_label": tgt_name,
            "original_confidence": float(row["confidence"]),
            "augmented_file": out_filename,
            "is_augmented": True,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{len(source_df)}] {rate:.0f} img/s")

    elapsed = time.time() - t0
    print(f"  ✓ Transferred {len(results)} images in {elapsed:.1f}s "
          f"({len(results)/max(elapsed,1):.0f} img/s)")

    return results


# ═══════════════════════════════════════════════════════════════
# PREVIEW / QA
# ═══════════════════════════════════════════════════════════════

def generate_preview(predictions_csv, image_root, output_dir,
                     source_class_id, target_class_id, target_stats,
                     n_preview=10, image_ext=".jpg"):
    """
    Generate a side-by-side preview grid for visual quality assessment.
    Shows original → transferred pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(predictions_csv)
    source_df = df[df["predicted_class"] == source_class_id].sample(
        n=min(n_preview, len(df[df["predicted_class"] == source_class_id])),
        random_state=42,
    )

    src_name = CLASS_NAMES.get(source_class_id, f"Class {source_class_id}")
    tgt_name = CLASS_NAMES.get(target_class_id, f"Class {target_class_id}")

    n = len(source_df)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    fig.suptitle(f"Reinhard Colour Transfer Preview: {src_name} → {tgt_name}",
                 fontsize=16, fontweight="bold", y=1.01)

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (_, row) in enumerate(source_df.iterrows()):
        isic_id = row["isic_id"]
        img_path = os.path.join(image_root, f"{isic_id}{image_ext}")
        img = cv2.imread(img_path)
        if img is None:
            continue

        transferred = reinhard_transfer(img, target_stats)

        # Difference map (amplified for visibility)
        diff = cv2.absdiff(img, transferred)
        diff_amplified = np.clip(diff.astype(np.float32) * 3, 0, 255).astype(np.uint8)

        # Convert BGR → RGB for matplotlib
        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trans_rgb = cv2.cvtColor(transferred, cv2.COLOR_BGR2RGB)
        diff_rgb = cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB)

        # Original stats
        orig_stats = compute_lab_stats(img)
        trans_stats = compute_lab_stats(transferred)

        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(
            f"Original ({src_name})\n"
            f"L*={orig_stats['L_mean']:.0f} a*={orig_stats['a_mean']:.0f} "
            f"b*={orig_stats['b_mean']:.0f}",
            fontsize=9
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(trans_rgb)
        axes[i, 1].set_title(
            f"Transferred ({tgt_name})\n"
            f"L*={trans_stats['L_mean']:.0f} a*={trans_stats['a_mean']:.0f} "
            f"b*={trans_stats['b_mean']:.0f}",
            fontsize=9
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff_rgb)
        axes[i, 2].set_title("Difference (3× amplified)", fontsize=9)
        axes[i, 2].axis("off")

        if i == 0:
            axes[i, 0].set_ylabel("Sample", fontsize=11)

    plt.tight_layout()
    preview_path = os.path.join(output_dir, f"preview_{src_name.split('(')[0].strip().lower()}"
                                f"_to_{tgt_name.split('(')[0].strip().lower()}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Preview saved → {preview_path}")


# ═══════════════════════════════════════════════════════════════
# AUTO-BALANCE
# ═══════════════════════════════════════════════════════════════

def auto_balance(predictions_csv, image_root, output_dir,
                 class_stats, image_ext=".jpg", min_confidence=0.5):
    """
    Automatically transfer images to balance class distribution.

    Strategy: For each minority class, sample from the majority class(es)
    and transfer them to the minority's colour profile.

    Returns:
        Combined results list + saves metadata CSV
    """
    df = pd.read_csv(predictions_csv)
    counts = df["predicted_class"].value_counts().to_dict()

    print(f"\n  Current distribution:")
    for cls_id, count in sorted(counts.items()):
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        print(f"    {cls_name}: {count:,}")

    max_count = max(counts.values())
    all_results = []

    for cls_id, count in sorted(counts.items()):
        if count >= max_count:
            continue  # Already the largest class

        deficit = max_count - count
        cls_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")

        # Find best source class (largest that isn't the target)
        source_id = max(
            (c for c in counts if c != cls_id),
            key=lambda c: counts[c],
        )

        print(f"\n  Augmenting {cls_name}: need {deficit:,} more images")
        results = transfer_batch(
            predictions_csv, image_root, output_dir,
            source_class_id=source_id,
            target_class_id=cls_id,
            target_stats=class_stats[cls_id],
            num_images=deficit,
            image_ext=image_ext,
            min_confidence=min_confidence,
        )
        all_results.extend(results)

    return all_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Skin Tone Transfer via Reinhard LAB Colour Transfer"
    )
    parser.add_argument("--predictions", type=str, required=True,
                        help="CSV with isic_id, predicted_class, confidence columns")
    parser.add_argument("--image-root", type=str, required=True,
                        help="Directory containing source images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save transferred images")
    parser.add_argument("--image-ext", type=str, default=".jpg",
                        help="Image file extension (default: .jpg)")

    # Transfer mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto-balance", action="store_true",
                      help="Automatically balance all classes to match the largest")
    mode.add_argument("--source-class", type=str, choices=["Dark", "Medium", "Light"],
                      help="Source class to transfer FROM")

    parser.add_argument("--target-class", type=str, choices=["Dark", "Medium", "Light"],
                        help="Target class to transfer TO (required with --source-class)")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Number of images to transfer (default: all matching)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Only use source images with confidence >= this (default: 0.5)")
    parser.add_argument("--preview", type=int, default=0,
                        help="Generate N preview comparisons instead of full transfer")
    parser.add_argument("--stats-samples", type=int, default=200,
                        help="Max images per class for computing colour stats (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validation
    if args.source_class and not args.target_class:
        parser.error("--target-class is required when using --source-class")
    if args.source_class and args.source_class == args.target_class:
        parser.error("Source and target classes must be different")

    print(f"\n{'='*60}")
    print(f"Skin Tone Transfer - Reinhard LAB Method")
    print(f"{'='*60}")
    print(f"  Predictions: {args.predictions}")
    print(f"  Image root:  {args.image_root}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Mode:        {'auto-balance' if args.auto_balance else f'{args.source_class} → {args.target_class}'}")

    # ── Step 1: Compute per-class colour stats ──
    print(f"\n--- Computing per-class colour statistics ---")
    class_stats = compute_class_stats(
        args.predictions, args.image_root,
        image_ext=args.image_ext,
        max_samples_per_class=args.stats_samples,
    )

    if not class_stats:
        print("  ✗ No class stats computed! Check your data.")
        sys.exit(1)

    # ── Step 2: Run transfer ──
    if args.preview > 0:
        # Preview mode
        source_id = CLASS_MAP[args.source_class]
        target_id = CLASS_MAP[args.target_class]
        print(f"\n--- Generating {args.preview} preview comparisons ---")
        generate_preview(
            args.predictions, args.image_root, args.output_dir,
            source_id, target_id, class_stats[target_id],
            n_preview=args.preview, image_ext=args.image_ext,
        )

    elif args.auto_balance:
        # Auto-balance mode
        print(f"\n--- Auto-balancing class distribution ---")
        all_results = auto_balance(
            args.predictions, args.image_root, args.output_dir,
            class_stats, image_ext=args.image_ext,
            min_confidence=args.min_confidence,
        )

        # Save metadata
        if all_results:
            results_df = pd.DataFrame(all_results)
            meta_path = os.path.join(args.output_dir, "augmented_metadata.csv")
            results_df.to_csv(meta_path, index=False)
            print(f"\n  ✓ Metadata saved → {meta_path}")
            print(f"  ✓ Total augmented images: {len(all_results):,}")

    else:
        # Manual transfer mode
        source_id = CLASS_MAP[args.source_class]
        target_id = CLASS_MAP[args.target_class]

        print(f"\n--- Transferring {args.source_class} → {args.target_class} ---")
        results = transfer_batch(
            args.predictions, args.image_root, args.output_dir,
            source_class_id=source_id,
            target_class_id=target_id,
            target_stats=class_stats[target_id],
            num_images=args.num_images,
            image_ext=args.image_ext,
            min_confidence=args.min_confidence,
        )

        # Save metadata
        if results:
            results_df = pd.DataFrame(results)
            meta_path = os.path.join(args.output_dir, "augmented_metadata.csv")
            results_df.to_csv(meta_path, index=False)
            print(f"\n  ✓ Metadata saved → {meta_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"✓ Transfer complete!")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

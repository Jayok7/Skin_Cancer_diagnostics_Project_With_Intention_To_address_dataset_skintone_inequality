#!/usr/bin/env python3
"""
Skin Tone Augmentation Pipeline — Reinhard CIE-LAB + Deep Blending
====================================================================
Balances the skin-tone distribution of ISIC 2019 using two complementary
methods, controlled by a tunable λ (lambda) parameter:

  λ = 0.0  →  100% Deep Blending (Poisson-gradient based)
  λ = 0.5  →  50/50 mix
  λ = 1.0  →  100% Reinhard CIE-LAB colour transport

Methods:
  1. **Reinhard CIE-LAB Transfer** (Reinhard et al. 2001)
     - Converts source + reference to CIE-LAB
     - Transfers per-channel mean & std from reference to source
     - Simple, fast, preserves spatial structure

  2. **Deep Blending** (Wu et al. 2022, doi:10.2196/39143)
     - Based on Poisson Image Editing (Pérez et al. 2003)
     - Uses OpenCV's seamlessClone to transplant the lesion region
       from the source onto a different-skin-tone reference image
     - Preserves gradient structure of the lesion while adopting
       the colour/texture of the reference skin
     - Falls back to Reinhard if seamlessClone fails (e.g. mask issues)

Reference Pool:
  - MSKCC images (3-class labelled) provide the reference styles
  - References are selected at random from the target class to maintain
    diversity (per your design requirement)

Usage:
    python skin_tone_augmentation.py \
        --isic-csv  outputs/isic2019_skin_tone_labels.csv \
        --mskcc-csv outputs/mskcc_skin_tone_labels.csv \
        --isic-images  datasets/ISIC_2019_Training_Input \
        --mskcc-images datasets/MSKCC-images \
        --output-dir   datasets/ISIC_2019_Augmented \
        --lambda-ratio 0.7 \
        --target-per-class 8500 \
        --max-cycles 4
"""

import os
import sys
import argparse
import random
import time
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image


# ══════════════════════════════════════════════════════════════
#  1.  REINHARD CIE-LAB COLOUR TRANSFER
# ══════════════════════════════════════════════════════════════

def reinhard_transfer(source_bgr: np.ndarray, reference_bgr: np.ndarray,
                      clip: bool = True) -> np.ndarray:
    """
    Reinhard et al. (2001) colour transfer in CIE-LAB space.

    Transfers the mean and standard deviation of each LAB channel
    from the reference image to the source image. This shifts the
    overall colour palette while preserving spatial structure.

    Parameters
    ----------
    source_bgr : np.ndarray
        Source image (BGR, uint8) to be recoloured.
    reference_bgr : np.ndarray
        Reference image (BGR, uint8) whose colour statistics
        will be transferred. Selected randomly from the MSKCC pool.
    clip : bool
        If True, clip LAB values to valid ranges before conversion.

    Returns
    -------
    np.ndarray
        Recoloured image (BGR, uint8).
    """
    # Convert to LAB (float32 for arithmetic)
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Per-channel statistics
    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1))
    ref_mean, ref_std = ref_lab.mean(axis=(0, 1)), ref_lab.std(axis=(0, 1))

    # Avoid division by zero
    src_std = np.where(src_std < 1e-6, 1.0, src_std)

    # Transfer: normalise source, scale by reference, shift to reference mean
    result_lab = (src_lab - src_mean) * (ref_std / src_std) + ref_mean

    if clip:
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)

    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result_bgr


# ══════════════════════════════════════════════════════════════
#  2.  DEEP BLENDING (Poisson Image Editing)
# ══════════════════════════════════════════════════════════════

def create_lesion_mask(image_bgr: np.ndarray,
                       threshold: int = 40) -> np.ndarray:
    """
    Create a rough binary mask of the lesion region.

    Uses colour-distance from the border mean to identify the lesion.
    The border (outer 10% ring) is assumed to be perilesional skin.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image (BGR, uint8).
    threshold : int
        Distance threshold in LAB space for lesion detection.

    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255).
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample border pixels (outer 10%)
    margin_h, margin_w = max(1, h // 10), max(1, w // 10)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin_h, :] = True
    border_mask[-margin_h:, :] = True
    border_mask[:, :margin_w] = True
    border_mask[:, -margin_w:] = True

    border_pixels = lab[border_mask]
    skin_mean = border_pixels.mean(axis=0)

    # Distance from skin mean
    diff = np.sqrt(((lab - skin_mean) ** 2).sum(axis=2))
    mask = (diff > threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Keep only the largest contour (the lesion)
        largest = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest], -1, 255, -1)
        mask = mask_filled

    return mask


def deep_blend(source_bgr: np.ndarray, reference_bgr: np.ndarray,
               mask: np.ndarray = None) -> np.ndarray:
    """
    Deep Blending via Poisson Image Editing (Pérez et al. 2003).

    Transplants the lesion (defined by `mask`) from `source_bgr` onto
    `reference_bgr` using OpenCV's seamlessClone. This preserves the
    gradient structure of the lesion while adopting the skin colour
    of the reference image.

    Based on the methodology from Wu et al. (2022):
    "Improving Skin Color Diversity in Cancer Detection: Deep Learning
    Approach" (doi:10.2196/39143).

    Parameters
    ----------
    source_bgr : np.ndarray
        Source lesion image (BGR, uint8).
    reference_bgr : np.ndarray
        Reference skin image (BGR, uint8) of the target skin tone.
    mask : np.ndarray or None
        Binary mask of the lesion. If None, auto-generated.

    Returns
    -------
    np.ndarray
        Blended image (BGR, uint8).
    """
    if mask is None:
        mask = create_lesion_mask(source_bgr)

    # Ensure mask has content
    if mask.sum() < 100:
        # Fallback: use Reinhard if mask is too small
        return reinhard_transfer(source_bgr, reference_bgr)

    # Resize reference to match source dimensions
    h, w = source_bgr.shape[:2]
    ref_resized = cv2.resize(reference_bgr, (w, h),
                              interpolation=cv2.INTER_LANCZOS4)

    # Center of the lesion for seamlessClone
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return reinhard_transfer(source_bgr, reference_bgr)

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Ensure center is valid
    cx = max(1, min(cx, w - 2))
    cy = max(1, min(cy, h - 2))

    try:
        # MIXED_CLONE preserves lesion texture better than NORMAL_CLONE
        result = cv2.seamlessClone(
            source_bgr, ref_resized, mask, (cx, cy),
            cv2.MIXED_CLONE
        )
        return result
    except cv2.error:
        # Fallback to Reinhard if Poisson solver fails
        return reinhard_transfer(source_bgr, reference_bgr)


# ══════════════════════════════════════════════════════════════
#  3.  HYBRID AUGMENTATION (λ-controlled)
# ══════════════════════════════════════════════════════════════

def augment_image(source_bgr: np.ndarray, reference_bgr: np.ndarray,
                  lambda_ratio: float = 0.7,
                  mask: np.ndarray = None) -> tuple:
    """
    Apply hybrid augmentation controlled by λ.

    For a given image, randomly decides which method to use based
    on the lambda_ratio. This ensures the augmented dataset has
    a mix of both approaches for diversity.

    Parameters
    ----------
    source_bgr : np.ndarray
        Source image to augment.
    reference_bgr : np.ndarray
        Reference image from the MSKCC pool.
    lambda_ratio : float
        Probability of using Reinhard (1.0 = all Reinhard,
        0.0 = all deep blending).
    mask : np.ndarray or None
        Optional precomputed lesion mask.

    Returns
    -------
    tuple of (np.ndarray, str)
        (augmented_image, method_used)
    """
    if random.random() < lambda_ratio:
        result = reinhard_transfer(source_bgr, reference_bgr)
        method = "reinhard"
    else:
        result = deep_blend(source_bgr, reference_bgr, mask=mask)
        method = "deep_blend"

    return result, method


# ══════════════════════════════════════════════════════════════
#  3b. QUALITY FILTER — reject unrealistic augmented images
# ══════════════════════════════════════════════════════════════

def quality_check(image_bgr: np.ndarray) -> tuple:
    """
    Fast realism check for augmented dermoscopic images.

    Analyses BOTH the perilesional skin (image borders) AND the
    central lesion region to detect colour casts, extreme saturation,
    and artifacts from failed colour transfers.

    Checks (border region):
      1. Skin hue plausibility (HSV) — rejects green/blue casts
      2. Saturation bounds — rejects neon/oversaturated images
      3. LAB a*/b* range — rejects extreme colour shifts
      4. Colour variance — rejects patchy multi-colour artifacts

    Checks (central region):
      5. Central hue plausibility — catches lesion-area colour casts
      6. Central oversaturation — catches neon lesion artifacts

    Returns
    -------
    tuple of (bool, str)
        (passed, reason) — True if realistic, reason string if rejected.
    """
    h, w = image_bgr.shape[:2]

    # ── Region masks ──
    # Border: outer 12% ring (perilesional skin)
    margin_h = max(2, h // 8)
    margin_w = max(2, w // 8)
    border = np.zeros((h, w), dtype=bool)
    border[:margin_h, :] = True
    border[-margin_h:, :] = True
    border[:, :margin_w] = True
    border[:, -margin_w:] = True

    # Center: inner 30% box (lesion region)
    ch, cw = h // 2, w // 2
    rh, rw = max(2, int(h * 0.15)), max(2, int(w * 0.15))
    center = np.zeros((h, w), dtype=bool)
    center[ch - rh:ch + rh, cw - rw:cw + rw] = True

    # ── HSV analysis (border) ──
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    border_h = hsv[border, 0].astype(float)
    border_s = hsv[border, 1].astype(float)
    border_v = hsv[border, 2].astype(float)

    median_hue = np.median(border_h)
    mean_sat = np.mean(border_s)
    mean_val = np.mean(border_v)

    # 1. Skin hue check (border)
    #    Skin tones: 0-25 and 170-180. Green=35-85, Blue=85-130.
    if 30 < median_hue < 160:
        return False, f"border_non_skin_hue={median_hue:.0f}"

    # 2. Oversaturation (border)
    if mean_sat > 170:
        return False, f"border_oversaturated={mean_sat:.0f}"

    # 3. Too dark / underexposed
    if mean_val < 25:
        return False, f"too_dark={mean_val:.0f}"

    # ── HSV analysis (center) ──
    center_h = hsv[center, 0].astype(float)
    center_s = hsv[center, 1].astype(float)
    center_median_hue = np.median(center_h)
    center_mean_sat = np.mean(center_s)

    # 5. Central hue check — slightly more lenient than border
    #    since lesions can have unusual hues, but green is never valid
    if 40 < center_median_hue < 150:
        return False, f"center_non_skin_hue={center_median_hue:.0f}"

    # 6. Central oversaturation
    if center_mean_sat > 200:
        return False, f"center_oversaturated={center_mean_sat:.0f}"

    # ── LAB analysis (border) ──
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    border_lab = lab[border].astype(float)

    mean_a = border_lab[:, 1].mean()
    mean_b = border_lab[:, 2].mean()
    std_a = border_lab[:, 1].std()
    std_b = border_lab[:, 2].std()

    # 4a. Extreme a* (green↔magenta)
    if mean_a < 105:
        return False, f"green_cast_a={mean_a:.0f}"
    if mean_a > 180:
        return False, f"magenta_cast_a={mean_a:.0f}"

    # 4b. Extreme b* (blue↔yellow)
    if mean_b < 95:
        return False, f"blue_cast_b={mean_b:.0f}"
    if mean_b > 200:
        return False, f"extreme_yellow_b={mean_b:.0f}"

    # 4c. Patchy colour variance in border
    if std_a > 35:
        return False, f"patchy_a_std={std_a:.0f}"
    if std_b > 35:
        return False, f"patchy_b_std={std_b:.0f}"

    # ── LAB analysis (center) — catch central artifacts ──
    center_lab = lab[center].astype(float)
    center_a = center_lab[:, 1].mean()
    center_b = center_lab[:, 2].mean()

    # Slightly wider range for lesion centre (lesions can be unusual)
    if center_a < 90:
        return False, f"center_green_a={center_a:.0f}"
    if center_b < 80:
        return False, f"center_blue_b={center_b:.0f}"

    return True, "ok"


# ══════════════════════════════════════════════════════════════
#  4.  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_labels_csv(csv_path: str) -> dict:
    """
    Load a skin tone labels CSV into a dict grouped by class.

    CSV format: image_path, image_id, predicted_label, confidence

    Returns
    -------
    dict
        {class_name: [{'image_path': ..., 'image_id': ...,
                        'predicted_label': ..., 'confidence': ...}, ...]}
    """
    by_class = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row['predicted_label']
            by_class[cls].append(row)
    return dict(by_class)


def resolve_image_path(row: dict, image_dir: str) -> str:
    """
    Resolve the actual image path on this machine.

    The CSV may contain CSF absolute paths; we re-root to the
    local image directory using the filename.
    """
    image_id = row['image_id']
    original_path = row['image_path']

    # Extract the filename from the original path
    filename = os.path.basename(original_path)

    # Try the local directory
    local_path = os.path.join(image_dir, filename)
    if os.path.exists(local_path):
        return local_path

    # Try with just the image_id + .jpg
    alt_path = os.path.join(image_dir, f"{image_id}.jpg")
    if os.path.exists(alt_path):
        return alt_path

    # Fallback: return the original path (for CSF execution)
    return original_path


# ══════════════════════════════════════════════════════════════
#  5.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Skin Tone Augmentation: Reinhard CIE-LAB + Deep Blending"
    )
    parser.add_argument('--isic-csv', required=True,
                        help='ISIC 2019 skin tone labels CSV')
    parser.add_argument('--mskcc-csv', required=True,
                        help='MSKCC skin tone labels CSV (reference pool)')
    parser.add_argument('--isic-images', required=True,
                        help='Directory containing ISIC 2019 images')
    parser.add_argument('--mskcc-images', required=True,
                        help='Directory containing MSKCC images')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for augmented images')
    parser.add_argument('--lambda-ratio', type=float, default=0.7,
                        help='λ: ratio of Reinhard (1.0) vs Deep Blending (0.0). '
                             'Default: 0.7 (70%% Reinhard, 30%% Deep Blending)')
    parser.add_argument('--target-per-class', type=int, default=8500,
                        help='Target number of images per class after augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-cycles', type=int, default=4,
                        help='Max generate→filter→retry cycles (default: 4)')
    parser.add_argument('--preview', type=int, default=0,
                        help='Generate N preview comparisons and exit')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Print config ──
    print(f"\n{'='*65}")
    print(f"  Skin Tone Augmentation Pipeline")
    print(f"{'='*65}")
    print(f"  ISIC CSV:        {args.isic_csv}")
    print(f"  MSKCC CSV:       {args.mskcc_csv}")
    print(f"  ISIC images:     {args.isic_images}")
    print(f"  MSKCC images:    {args.mskcc_images}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  λ (lambda):      {args.lambda_ratio:.2f}")
    print(f"    → Reinhard:    {args.lambda_ratio*100:.0f}%")
    print(f"    → Deep Blend:  {(1-args.lambda_ratio)*100:.0f}%")
    print(f"  Target/class:    {args.target_per_class}")
    print(f"  Max cycles:      {args.max_cycles}")
    print(f"  Seed:            {args.seed}")
    print(f"{'='*65}\n")

    # ── Load labels ──
    print("Loading label CSVs...")
    isic_by_class = load_labels_csv(args.isic_csv)
    mskcc_by_class = load_labels_csv(args.mskcc_csv)

    all_classes = sorted(set(list(isic_by_class.keys()) +
                             list(mskcc_by_class.keys())))

    print(f"\n  ISIC 2019 Distribution:")
    for cls in all_classes:
        count = len(isic_by_class.get(cls, []))
        print(f"    {cls:8s}: {count:6d}")

    print(f"\n  MSKCC Reference Pool:")
    for cls in all_classes:
        count = len(mskcc_by_class.get(cls, []))
        print(f"    {cls:8s}: {count:6d}")

    # ── Calculate augmentation budget ──
    print(f"\n  Augmentation Budget (target = {args.target_per_class}/class):")
    augmentation_plan = {}
    for cls in all_classes:
        current = len(isic_by_class.get(cls, []))
        needed = max(0, args.target_per_class - current)
        augmentation_plan[cls] = needed
        status = "✓ sufficient" if needed == 0 else f"need {needed} more"
        print(f"    {cls:8s}: {current:6d} → {args.target_per_class:6d}  ({status})")

    total_to_generate = sum(augmentation_plan.values())
    if total_to_generate == 0:
        print("\n  All classes meet the target. Nothing to augment.")
        return

    print(f"\n  Total synthetic images to generate: {total_to_generate}")

    # ── Create output directories ──
    os.makedirs(args.output_dir, exist_ok=True)
    for cls in all_classes:
        os.makedirs(os.path.join(args.output_dir, cls), exist_ok=True)

    # ── Preview mode ──
    if args.preview > 0:
        run_preview(args, isic_by_class, mskcc_by_class, all_classes)
        return

    # ── Run augmentation with iterative quality filtering ──
    MAX_CYCLES = args.max_cycles
    OVERSAMPLE = 1.5   # try 50% more per cycle to compensate for rejections

    print(f"\n{'='*65}")
    print(f"  Generating augmented images (max {MAX_CYCLES} cycles)...")
    print(f"{'='*65}\n")

    manifest = []  # Track all accepted images
    t0 = time.time()
    total_generated = 0
    total_rejected = 0
    total_errors = 0
    reject_reasons = defaultdict(int)
    img_counter = 0  # global counter for unique filenames

    for cls in all_classes:
        n_target = augmentation_plan[cls]
        if n_target == 0:
            continue

        # Source pool: over-represented classes (we recolour them)
        source_classes = [c for c in all_classes
                          if c != cls and len(isic_by_class.get(c, [])) > 0]
        source_pool = []
        for sc in source_classes:
            source_pool.extend(isic_by_class[sc])

        if not source_pool:
            print(f"    ⚠ No source images for class {cls}")
            continue

        # Reference pool: MSKCC images of the TARGET skin tone
        ref_pool = mskcc_by_class.get(cls, [])
        if not ref_pool:
            ref_pool = isic_by_class.get(cls, [])
        if not ref_pool:
            print(f"    ⚠ No reference images for class {cls}")
            continue

        cls_generated = 0

        # ── λ-conserving method quotas ──
        # Pre-compute how many of each method this class needs
        # so the final output exactly matches the requested λ ratio.
        n_reinhard_target = int(round(n_target * args.lambda_ratio))
        n_blend_target = n_target - n_reinhard_target
        reinhard_accepted = 0
        blend_accepted = 0

        for cycle in range(1, MAX_CYCLES + 1):
            remaining = n_target - cls_generated
            if remaining <= 0:
                break

            attempts = int(remaining * OVERSAMPLE) if cycle < MAX_CYCLES else remaining * 3
            cycle_accepted = 0
            cycle_rejected = 0

            random.shuffle(source_pool)

            print(f"\n  [{cls}] Cycle {cycle}/{MAX_CYCLES}: "
                  f"need {remaining} more (R:{n_reinhard_target-reinhard_accepted}/"
                  f"B:{n_blend_target-blend_accepted}), "
                  f"attempting {attempts}...")

            for i in range(attempts):
                if cls_generated >= n_target:
                    break

                source_row = source_pool[i % len(source_pool)]
                ref_row = random.choice(ref_pool)

                src_path = resolve_image_path(source_row, args.isic_images)
                ref_path = resolve_image_path(ref_row, args.mskcc_images)

                try:
                    src_img = cv2.imread(src_path)
                    ref_img = cv2.imread(ref_path)

                    if src_img is None or ref_img is None:
                        total_errors += 1
                        continue

                    # ── Deterministic method selection (λ conservation) ──
                    # Pick whichever method still has quota remaining.
                    # If both have quota, alternate to maintain ratio.
                    r_need = n_reinhard_target - reinhard_accepted
                    b_need = n_blend_target - blend_accepted

                    if r_need > 0 and b_need > 0:
                        # Both need more: use ratio to decide
                        if r_need / max(r_need + b_need, 1) >= random.random():
                            force_method = "reinhard"
                        else:
                            force_method = "deep_blend"
                    elif r_need > 0:
                        force_method = "reinhard"
                    elif b_need > 0:
                        force_method = "deep_blend"
                    else:
                        break  # both quotas met

                    if force_method == "reinhard":
                        result = reinhard_transfer(src_img, ref_img)
                        method = "reinhard"
                    else:
                        result = deep_blend(src_img, ref_img)
                        method = "deep_blend"

                    # ── Quality gate ──
                    passed, reason = quality_check(result)
                    if not passed:
                        cycle_rejected += 1
                        total_rejected += 1
                        reject_reasons[reason.split('=')[0]] += 1
                        continue

                    # ── Save accepted image ──
                    src_id = source_row['image_id']
                    ref_id = ref_row['image_id']
                    out_name = f"{src_id}_to_{cls}_{method}_{img_counter:05d}.jpg"
                    out_path = os.path.join(args.output_dir, cls, out_name)

                    cv2.imwrite(out_path, result,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])

                    manifest.append({
                        'output_path': out_path,
                        'output_name': out_name,
                        'source_id': src_id,
                        'source_class': source_row['predicted_label'],
                        'reference_id': ref_id,
                        'reference_class': cls,
                        'method': method,
                        'target_class': cls,
                    })

                    cls_generated += 1
                    total_generated += 1
                    cycle_accepted += 1
                    img_counter += 1
                    if method == "reinhard":
                        reinhard_accepted += 1
                    else:
                        blend_accepted += 1

                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"    ⚠ Error: {e}")
                    continue

                # Progress every 500 accepted
                if cycle_accepted % 500 == 0:
                    elapsed = time.time() - t0
                    rate = total_generated / max(elapsed, 1)
                    print(f"    ... {cls_generated}/{n_target} accepted "
                          f"(R:{reinhard_accepted} B:{blend_accepted}, "
                          f"{rate:.0f} img/s)")

            pass_rate = cycle_accepted / max(cycle_accepted + cycle_rejected, 1)
            print(f"    Cycle {cycle}: accepted={cycle_accepted}, "
                  f"rejected={cycle_rejected}, "
                  f"pass_rate={pass_rate:.0%}")

        shortfall = n_target - cls_generated
        status = "✓" if shortfall == 0 else f"⚠ short by {shortfall}"
        print(f"  {status} {cls}: {cls_generated}/{n_target} "
              f"(reinhard={reinhard_accepted}, blend={blend_accepted})")

        # Warn if λ drifted significantly
        actual_lambda = reinhard_accepted / max(cls_generated, 1)
        if abs(actual_lambda - args.lambda_ratio) > 0.05 and cls_generated > 0:
            print(f"    ⚠ λ drift: target={args.lambda_ratio:.2f}, "
                  f"actual={actual_lambda:.2f}")

    # ── Save manifest ──
    manifest_path = os.path.join(args.output_dir, "augmentation_manifest.csv")
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'output_name', 'source_id', 'source_class',
            'reference_id', 'reference_class', 'method', 'target_class',
        ])
        writer.writeheader()
        for entry in manifest:
            writer.writerow({k: v for k, v in entry.items()
                             if k != 'output_path'})

    # ── Final report ──
    total_time = time.time() - t0
    reinhard_count = sum(1 for m in manifest if m['method'] == 'reinhard')
    blend_count = sum(1 for m in manifest if m['method'] == 'deep_blend')

    print(f"\n{'='*65}")
    print(f"  Augmentation Complete")
    print(f"{'='*65}")
    print(f"  Accepted:      {total_generated}")
    print(f"  Rejected (QC): {total_rejected}")
    print(f"  Errors:        {total_errors}")
    print(f"  Pass rate:     {total_generated/max(total_generated+total_rejected,1):.1%}")
    print(f"  Time:          {total_time/60:.1f} minutes")
    print(f"  Rate:          {total_generated/max(total_time,1):.0f} img/s")
    print(f"  Reinhard:      {reinhard_count} ({100*reinhard_count/max(total_generated,1):.0f}%)")
    print(f"  Deep Blend:    {blend_count} ({100*blend_count/max(total_generated,1):.0f}%)")
    print(f"  Manifest:      {manifest_path}")

    if reject_reasons:
        print(f"\n  Rejection Breakdown:")
        for reason, count in sorted(reject_reasons.items(),
                                     key=lambda x: -x[1]):
            print(f"    {reason:25s}: {count:5d}")

    print(f"\n  Final Distribution (original + augmented):")
    for cls in all_classes:
        original = len(isic_by_class.get(cls, []))
        augmented = sum(1 for m in manifest if m['target_class'] == cls)
        total = original + augmented
        bar = "█" * (total // 500)
        print(f"    {cls:8s}: {total:6d}  "
              f"(orig={original}, aug={augmented})  {bar}")

    print(f"{'='*65}\n")


def run_preview(args, isic_by_class, mskcc_by_class, all_classes):
    """Generate side-by-side preview comparisons."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    preview_dir = os.path.join(args.output_dir, "_preview")
    os.makedirs(preview_dir, exist_ok=True)

    print(f"\n  Generating {args.preview} preview comparisons → {preview_dir}")

    # Pick a random under-represented class
    target_cls = "Dark"  # Usually the most under-represented
    ref_pool = mskcc_by_class.get(target_cls, [])
    source_pool = isic_by_class.get("Light", []) + isic_by_class.get("Medium", [])

    if not ref_pool or not source_pool:
        print("  ⚠ Not enough images for preview")
        return

    for i in range(min(args.preview, len(source_pool), 10)):
        src_row = source_pool[i]
        ref_row = random.choice(ref_pool)

        src_path = resolve_image_path(src_row, args.isic_images)
        ref_path = resolve_image_path(ref_row, args.mskcc_images)

        src_img = cv2.imread(src_path)
        ref_img = cv2.imread(ref_path)

        if src_img is None or ref_img is None:
            continue

        reinhard_result = reinhard_transfer(src_img, ref_img)
        blend_result = deep_blend(src_img, ref_img)

        # Plot comparison
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ['Source (Light/Medium)', f'Reference ({target_cls})',
                   'Reinhard CIE-LAB', 'Deep Blend (Poisson)']
        images = [src_img, cv2.resize(ref_img, (src_img.shape[1], src_img.shape[0])),
                  reinhard_result, blend_result]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11)
            ax.axis('off')

        fig.suptitle(f"Preview {i+1}: {src_row['image_id']} → {target_cls} "
                     f"(ref: {ref_row['image_id']})", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(preview_dir, f"preview_{i+1:02d}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved preview_{i+1:02d}.png")

    print(f"  ✓ Previews saved to {preview_dir}")


if __name__ == '__main__':
    main()

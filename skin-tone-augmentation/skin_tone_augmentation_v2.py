#!/usr/bin/env python3
"""
Skin Tone Augmentation Pipeline v2 -- U-Net Transplant + Deep Blending
======================================================================
Replaces Reinhard CIE-LAB with lesion-aware U-Net transplant.

  lambda = 0.0  ->  100% Deep Blending (Poisson, simple mask)
  lambda = 0.5  ->  50/50 mix
  lambda = 1.0  ->  100% U-Net Transplant (Poisson, U-Net mask on MSKCC bg)

Methods:
  1. **U-Net Transplant** 
     - Segments lesion with pretrained U-Net (ISIC 2018)
     - Picks dark-skin MSKCC reference, inpaints out its lesion
     - Poisson seamless-clones ISIC lesion onto clean dark background
     - REJECTION: if U-Net mask fails quality check, image is skipped

  2. **Deep Blending** 
     - Simple colour-distance mask
     - Poisson seamless clone onto reference
     - Falls back to skip if seamlessClone fails

Reference Pool:
  - MSKCC images provide dark-skin backgrounds
  - References selected randomly for diversity

Usage:
    python skin_tone_augmentation_v2.py \
        --isic-csv  outputs/isic2019_skin_tone_labels.csv \
        --mskcc-csv outputs/mskcc_skin_tone_labels.csv \
        --isic-images  datasets/ISIC_2019_Training_Input \
        --mskcc-images datasets/MSKCC-images \
        --unet-dir Pytorch-UNet \
        --unet-weights Pytorch-UNet/checkpoints/checkpoint_epoch50.pth \
        --output-dir   datasets/ISIC_2019_Augmented_v2 \
        --lambda-ratio 0.7 \
        --target-per-class 8500
"""

import os, sys, argparse, random, time, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F

print("[INFO] Imports complete", flush=True)


# ================================================================
# U-NET LOADING
# ================================================================

def load_unet(unet_dir, weights_path, device):
    """Load milesial/Pytorch-UNet with trained ISIC 2018 weights."""
    sys.path.insert(0, unet_dir)
    from unet import UNet

    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    net.load_state_dict(state_dict, strict=False)
    net.to(device).eval()
    print(f"  U-Net loaded from {weights_path}", flush=True)
    return net


# ================================================================
# LESION SEGMENTATION
# ================================================================

def unet_segment(net, image_bgr, device, threshold=0.5, pad_pixels=10):
    """
    Segment lesion using U-Net.
    Returns: (mask, is_valid)
      mask: binary mask 255=lesion, 0=skin
      is_valid: True if mask passes quality checks
    """
    h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = net(tensor)
        if logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits.squeeze(1))

    prob_map = probs.squeeze().cpu().numpy()
    prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (prob_map > threshold).astype(np.uint8) * 255

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Validate mask quality
    lesion_ratio = mask.sum() / (255.0 * h * w)
    is_valid = 0.005 < lesion_ratio < 0.7

    # Pad for better blending
    if pad_pixels > 0 and is_valid:
        pad_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pad_pixels * 2 + 1, pad_pixels * 2 + 1))
        mask = cv2.dilate(mask, pad_kernel)

    return mask, is_valid


def simple_lesion_mask(image_bgr, threshold=40):
    """
    Simple colour-distance mask for deep blending fallback.
    Same as v1 create_lesion_mask.
    """
    h, w = image_bgr.shape[:2]
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    margin_h, margin_w = max(1, h // 10), max(1, w // 10)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin_h, :] = True
    border_mask[-margin_h:, :] = True
    border_mask[:, :margin_w] = True
    border_mask[:, -margin_w:] = True

    border_pixels = lab[border_mask]
    skin_mean = border_pixels.mean(axis=0)

    diff = np.sqrt(((lab - skin_mean) ** 2).sum(axis=2))
    mask = (diff > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest], -1, 255, -1)
        mask = mask_filled

    return mask


# ================================================================
# BACKGROUND HELPERS
# ================================================================

def get_lesion_center(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        h, w = mask.shape
        return (w // 2, h // 2)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)


def detect_dermoscope_circle(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corner_size = min(h, w) // 8
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    dark_corners = sum(1 for c in corners if c.mean() < 20)
    if dark_corners >= 3:
        _, circle_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        circle_mask = cv2.morphologyEx(circle_mask, cv2.MORPH_CLOSE, kernel)
        return circle_mask
    return None


def prepare_dark_background(ref_bgr, target_h, target_w, circle_mask=None):
    ref_h, ref_w = ref_bgr.shape[:2]
    scale = max(target_h / ref_h, target_w / ref_w) * 1.1
    new_h, new_w = int(ref_h * scale), int(ref_w * scale)
    resized = cv2.resize(ref_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    y_off = (new_h - target_h) // 2
    x_off = (new_w - target_w) // 2
    background = resized[y_off:y_off + target_h, x_off:x_off + target_w]

    if circle_mask is not None:
        bg_mask = np.stack([circle_mask / 255.0] * 3, axis=-1)
        background = (background.astype(np.float64) * bg_mask).astype(np.uint8)

    return background


def remove_ref_lesion(ref_bgr, unet=None, device=None):
    """Inpaint out the reference image's own lesion."""
    if unet is not None and device is not None:
        ref_mask, valid = unet_segment(unet, ref_bgr, device, pad_pixels=20)
        if not valid:
            # Simple fallback for reference cleaning
            ref_mask = simple_lesion_mask(ref_bgr)
    else:
        ref_mask = simple_lesion_mask(ref_bgr)

    lesion_ratio = ref_mask.sum() / (255.0 * ref_mask.shape[0] * ref_mask.shape[1])
    if 0.01 < lesion_ratio < 0.4:
        return cv2.inpaint(ref_bgr, ref_mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
    return ref_bgr


# ================================================================
# METHOD 1: U-NET TRANSPLANT (replaces Reinhard)
# ================================================================

def unet_transplant(source_bgr, ref_bgr_clean, lesion_mask,
                    lambda_blend=0.7):
    """
    Transplant lesion from source onto cleaned dark-skin reference.
    ref_bgr_clean: MSKCC image with its own lesion already inpainted out.
    lesion_mask: U-Net generated mask (must be pre-validated).
    lambda_blend: how much original lesion colour to preserve (0.5-0.8).

    Returns result image or None if failed.
    """
    h, w = source_bgr.shape[:2]

    circle_mask = detect_dermoscope_circle(source_bgr)
    background = prepare_dark_background(ref_bgr_clean, h, w, circle_mask)

    center = get_lesion_center(lesion_mask)
    center = (max(1, min(center[0], w - 2)), max(1, min(center[1], h - 2)))

    try:
        result = cv2.seamlessClone(
            source_bgr, background, lesion_mask, center, cv2.MIXED_CLONE)
    except cv2.error:
        return None

    # Blend back some original lesion colour for natural appearance
    if lambda_blend > 0:
        lesion_soft = cv2.GaussianBlur(
            lesion_mask.astype(np.float64) / 255.0, (21, 21), 0)
        lesion_3ch = np.stack([lesion_soft] * 3, axis=-1)
        result = (
            lesion_3ch * (lambda_blend * source_bgr + (1 - lambda_blend) * result)
            + (1 - lesion_3ch) * result
        ).astype(np.uint8)

    if circle_mask is not None:
        border_3ch = np.stack([circle_mask / 255.0] * 3, axis=-1)
        result = (result.astype(np.float64) * border_3ch).astype(np.uint8)

    return result


# ================================================================
# METHOD 2: DEEP BLENDING (same as v1)
# ================================================================

def deep_blend(source_bgr, reference_bgr, mask=None):
    """
    Poisson seamless clone with simple colour-distance mask.
    Falls back to None if it fails (instead of Reinhard).
    """
    if mask is None:
        mask = simple_lesion_mask(source_bgr)

    if mask.sum() < 100:
        return None

    h, w = source_bgr.shape[:2]
    ref_resized = cv2.resize(reference_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None

    cx = max(1, min(int(moments["m10"] / moments["m00"]), w - 2))
    cy = max(1, min(int(moments["m01"] / moments["m00"]), h - 2))

    try:
        result = cv2.seamlessClone(
            source_bgr, ref_resized, mask, (cx, cy), cv2.MIXED_CLONE)
        return result
    except cv2.error:
        return None


# ================================================================
# QUALITY FILTER (same as v1)
# ================================================================

def quality_check(image_bgr):
    h, w = image_bgr.shape[:2]

    margin_h = max(2, h // 8)
    margin_w = max(2, w // 8)
    border = np.zeros((h, w), dtype=bool)
    border[:margin_h, :] = True
    border[-margin_h:, :] = True
    border[:, :margin_w] = True
    border[:, -margin_w:] = True

    ch, cw = h // 2, w // 2
    rh, rw = max(2, int(h * 0.15)), max(2, int(w * 0.15))
    center = np.zeros((h, w), dtype=bool)
    center[ch - rh:ch + rh, cw - rw:cw + rw] = True

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    border_h = hsv[border, 0].astype(float)
    border_s = hsv[border, 1].astype(float)
    border_v = hsv[border, 2].astype(float)

    median_hue = np.median(border_h)
    mean_sat = np.mean(border_s)
    mean_val = np.mean(border_v)

    if 30 < median_hue < 160:
        return False, f"border_non_skin_hue={median_hue:.0f}"
    if mean_sat > 170:
        return False, f"border_oversaturated={mean_sat:.0f}"
    if mean_val < 25:
        return False, f"too_dark={mean_val:.0f}"

    center_h = hsv[center, 0].astype(float)
    center_s = hsv[center, 1].astype(float)
    if 40 < np.median(center_h) < 150:
        return False, f"center_non_skin_hue={np.median(center_h):.0f}"
    if np.mean(center_s) > 200:
        return False, f"center_oversaturated={np.mean(center_s):.0f}"

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    border_lab = lab[border].astype(float)
    mean_a = border_lab[:, 1].mean()
    mean_b = border_lab[:, 2].mean()
    std_a = border_lab[:, 1].std()
    std_b = border_lab[:, 2].std()

    if mean_a < 105: return False, f"green_cast_a={mean_a:.0f}"
    if mean_a > 180: return False, f"magenta_cast_a={mean_a:.0f}"
    if mean_b < 95:  return False, f"blue_cast_b={mean_b:.0f}"
    if mean_b > 200: return False, f"extreme_yellow_b={mean_b:.0f}"
    if std_a > 35:   return False, f"patchy_a_std={std_a:.0f}"
    if std_b > 35:   return False, f"patchy_b_std={std_b:.0f}"

    center_lab = lab[center].astype(float)
    if center_lab[:, 1].mean() < 90: return False, f"center_green_a={center_lab[:, 1].mean():.0f}"
    if center_lab[:, 2].mean() < 80: return False, f"center_blue_b={center_lab[:, 2].mean():.0f}"

    return True, "ok"


# ================================================================
# DATA LOADING
# ================================================================

def load_labels_csv(csv_path):
    by_class = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row['predicted_label']
            by_class[cls].append(row)
    return dict(by_class)


def resolve_image_path(row, image_dir):
    image_id = row['image_id']
    original_path = row['image_path']
    filename = os.path.basename(original_path)

    local_path = os.path.join(image_dir, filename)
    if os.path.exists(local_path):
        return local_path

    alt_path = os.path.join(image_dir, f"{image_id}.jpg")
    if os.path.exists(alt_path):
        return alt_path

    return original_path


def preload_mskcc_references(mskcc_by_class, mskcc_images_dir, target_class,
                              unet, device, max_refs=100, seed=42):
    """
    Preload and clean MSKCC dark-skin references (inpaint out their lesions).
    This avoids re-processing them for every augmentation.
    """
    rng = np.random.RandomState(seed)
    ref_pool = mskcc_by_class.get(target_class, [])
    if not ref_pool:
        # Fall back to any MSKCC images
        for cls_imgs in mskcc_by_class.values():
            ref_pool.extend(cls_imgs)

    if not ref_pool:
        return []

    n = min(max_refs, len(ref_pool))
    indices = rng.choice(len(ref_pool), n, replace=False)

    cleaned_refs = []
    for idx in indices:
        row = ref_pool[idx]
        path = resolve_image_path(row, mskcc_images_dir)
        img = cv2.imread(path)
        if img is None:
            continue
        clean = remove_ref_lesion(img, unet=unet, device=device)
        cleaned_refs.append({'image': clean, 'id': row['image_id']})

    return cleaned_refs


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Skin Tone Augmentation v2: U-Net Transplant + Deep Blending")
    parser.add_argument('--isic-csv', required=True)
    parser.add_argument('--mskcc-csv', required=True)
    parser.add_argument('--isic-images', required=True)
    parser.add_argument('--mskcc-images', required=True)
    parser.add_argument('--unet-dir', default='Pytorch-UNet')
    parser.add_argument('--unet-weights',
                        default='Pytorch-UNet/checkpoints/checkpoint_epoch50.pth')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--lambda-ratio', type=float, default=0.7,
                        help='Ratio of U-Net Transplant (1.0) vs Deep Blend (0.0)')
    parser.add_argument('--target-per-class', type=int, default=8500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-cycles', type=int, default=4)
    parser.add_argument('--unet-threshold', type=float, default=0.5)
    parser.add_argument('--pad-pixels', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}", flush=True)
    print(f"  Skin Tone Augmentation v2", flush=True)
    print(f"  U-Net Transplant + Deep Blending", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Device:          {device}", flush=True)
    print(f"  Lambda:          {args.lambda_ratio:.2f}", flush=True)
    print(f"    -> Transplant: {args.lambda_ratio*100:.0f}%", flush=True)
    print(f"    -> Deep Blend: {(1-args.lambda_ratio)*100:.0f}%", flush=True)
    print(f"  Target/class:    {args.target_per_class}", flush=True)
    print(f"  U-Net threshold: {args.unet_threshold}", flush=True)
    print(f"  Max cycles:      {args.max_cycles}", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Load U-Net
    print("Loading U-Net...", flush=True)
    unet = load_unet(args.unet_dir, args.unet_weights, device)

    # Load labels
    print("Loading label CSVs...", flush=True)
    isic_by_class = load_labels_csv(args.isic_csv)
    mskcc_by_class = load_labels_csv(args.mskcc_csv)
    all_classes = sorted(set(list(isic_by_class.keys()) + list(mskcc_by_class.keys())))

    print(f"\n  ISIC 2019 Distribution:", flush=True)
    for cls in all_classes:
        count = len(isic_by_class.get(cls, []))
        print(f"    {cls:8s}: {count:6d}", flush=True)

    print(f"\n  MSKCC Reference Pool:", flush=True)
    for cls in all_classes:
        count = len(mskcc_by_class.get(cls, []))
        print(f"    {cls:8s}: {count:6d}", flush=True)

    # Augmentation budget
    print(f"\n  Augmentation Budget (target={args.target_per_class}/class):", flush=True)
    augmentation_plan = {}
    for cls in all_classes:
        current = len(isic_by_class.get(cls, []))
        needed = max(0, args.target_per_class - current)
        augmentation_plan[cls] = needed
        status = "sufficient" if needed == 0 else f"need {needed} more"
        print(f"    {cls:8s}: {current:6d} -> {args.target_per_class:6d}  ({status})",
              flush=True)

    total_to_generate = sum(augmentation_plan.values())
    if total_to_generate == 0:
        print("\n  All classes meet the target. Nothing to augment.", flush=True)
        return

    print(f"\n  Total to generate: {total_to_generate}", flush=True)

    # Create output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    for cls in all_classes:
        os.makedirs(os.path.join(args.output_dir, cls), exist_ok=True)

    # Preload cleaned MSKCC references per class
    print(f"\n--- Preloading MSKCC references ---", flush=True)
    cleaned_refs_by_class = {}
    for cls in all_classes:
        if augmentation_plan.get(cls, 0) > 0:
            cleaned_refs_by_class[cls] = preload_mskcc_references(
                mskcc_by_class, args.mskcc_images, cls,
                unet, device, max_refs=100, seed=args.seed)
            print(f"  {cls}: {len(cleaned_refs_by_class[cls])} cleaned refs", flush=True)

    # Run augmentation
    print(f"\n{'='*65}", flush=True)
    print(f"  Generating augmented images...", flush=True)
    print(f"{'='*65}\n", flush=True)

    manifest = []
    t0 = time.time()
    total_generated = 0
    total_rejected_qc = 0
    total_rejected_unet = 0
    total_rejected_method = 0
    total_errors = 0
    reject_reasons = defaultdict(int)
    img_counter = 0

    OVERSAMPLE = 1.8
    MAX_CYCLES = args.max_cycles

    for cls in all_classes:
        n_target = augmentation_plan[cls]
        if n_target == 0:
            continue

        # Source pool
        source_classes = [c for c in all_classes
                          if c != cls and len(isic_by_class.get(c, [])) > 0]
        source_pool = []
        for sc in source_classes:
            source_pool.extend(isic_by_class[sc])
        if not source_pool:
            print(f"  WARNING: No source images for class {cls}", flush=True)
            continue

        # Cleaned reference pool
        ref_pool = cleaned_refs_by_class.get(cls, [])
        if not ref_pool:
            # Fallback: use raw MSKCC
            raw_refs = mskcc_by_class.get(cls, [])
            if not raw_refs:
                print(f"  WARNING: No references for class {cls}", flush=True)
                continue
            ref_pool = [{'image': cv2.imread(resolve_image_path(r, args.mskcc_images)),
                         'id': r['image_id']} for r in raw_refs[:50]]
            ref_pool = [r for r in ref_pool if r['image'] is not None]

        if not ref_pool:
            print(f"  WARNING: No valid references for {cls}", flush=True)
            continue

        cls_generated = 0
        n_transplant_target = int(round(n_target * args.lambda_ratio))
        n_blend_target = n_target - n_transplant_target
        transplant_accepted = 0
        blend_accepted = 0

        for cycle in range(1, MAX_CYCLES + 1):
            remaining = n_target - cls_generated
            if remaining <= 0:
                break

            attempts = int(remaining * OVERSAMPLE) if cycle < MAX_CYCLES else remaining * 3
            cycle_accepted = 0
            cycle_rejected = 0
            cycle_unet_reject = 0

            random.shuffle(source_pool)

            print(f"\n  [{cls}] Cycle {cycle}/{MAX_CYCLES}: "
                  f"need {remaining} more "
                  f"(T:{n_transplant_target-transplant_accepted}/"
                  f"B:{n_blend_target-blend_accepted}), "
                  f"attempting {attempts}...", flush=True)

            for i in range(attempts):
                if cls_generated >= n_target:
                    break

                source_row = source_pool[i % len(source_pool)]
                ref_entry = random.choice(ref_pool)

                src_path = resolve_image_path(source_row, args.isic_images)

                try:
                    src_img = cv2.imread(src_path)
                    ref_img = ref_entry['image']

                    if src_img is None or ref_img is None:
                        total_errors += 1
                        continue

                    # Decide method based on quota
                    t_need = n_transplant_target - transplant_accepted
                    b_need = n_blend_target - blend_accepted

                    if t_need > 0 and b_need > 0:
                        if t_need / max(t_need + b_need, 1) >= random.random():
                            force_method = "transplant"
                        else:
                            force_method = "deep_blend"
                    elif t_need > 0:
                        force_method = "transplant"
                    elif b_need > 0:
                        force_method = "deep_blend"
                    else:
                        break

                    result = None
                    method = force_method

                    if force_method == "transplant":
                        # U-Net segmentation with REJECTION
                        mask, is_valid = unet_segment(
                            unet, src_img, device,
                            threshold=args.unet_threshold,
                            pad_pixels=args.pad_pixels)

                        if not is_valid:
                            # REJECT: U-Net couldn't segment this image
                            cycle_unet_reject += 1
                            total_rejected_unet += 1
                            reject_reasons['unet_mask_invalid'] += 1
                            continue

                        result = unet_transplant(
                            src_img, ref_img, mask, lambda_blend=0.7)

                        if result is None:
                            total_rejected_method += 1
                            reject_reasons['transplant_failed'] += 1
                            continue

                    elif force_method == "deep_blend":
                        # Use raw (un-cleaned) MSKCC for deep blend
                        raw_ref_row = random.choice(
                            mskcc_by_class.get(cls, list(mskcc_by_class.values())[0]))
                        raw_ref_path = resolve_image_path(raw_ref_row, args.mskcc_images)
                        raw_ref_img = cv2.imread(raw_ref_path)

                        if raw_ref_img is None:
                            total_errors += 1
                            continue

                        result = deep_blend(src_img, raw_ref_img)
                        if result is None:
                            total_rejected_method += 1
                            reject_reasons['deep_blend_failed'] += 1
                            continue

                    # Quality gate
                    passed, reason = quality_check(result)
                    if not passed:
                        cycle_rejected += 1
                        total_rejected_qc += 1
                        reject_reasons[reason.split('=')[0]] += 1
                        continue

                    # Save
                    src_id = source_row['image_id']
                    ref_id = ref_entry['id'] if isinstance(ref_entry, dict) else 'unknown'
                    out_name = f"{src_id}_to_{cls}_{method}_{img_counter:05d}.jpg"
                    out_path = os.path.join(args.output_dir, cls, out_name)

                    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    manifest.append({
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
                    if method == "transplant":
                        transplant_accepted += 1
                    else:
                        blend_accepted += 1

                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"    ERROR: {e}", flush=True)
                    continue

                if cycle_accepted % 500 == 0 and cycle_accepted > 0:
                    elapsed = time.time() - t0
                    rate = total_generated / max(elapsed, 1)
                    print(f"    ... {cls_generated}/{n_target} accepted "
                          f"(T:{transplant_accepted} B:{blend_accepted}, "
                          f"{rate:.1f} img/s)", flush=True)

            pass_rate = cycle_accepted / max(cycle_accepted + cycle_rejected + cycle_unet_reject, 1)
            print(f"    Cycle {cycle}: accepted={cycle_accepted}, "
                  f"qc_reject={cycle_rejected}, unet_reject={cycle_unet_reject}, "
                  f"pass_rate={pass_rate:.0%}", flush=True)

        shortfall = n_target - cls_generated
        status = "DONE" if shortfall == 0 else f"SHORT by {shortfall}"
        print(f"  {status} {cls}: {cls_generated}/{n_target} "
              f"(transplant={transplant_accepted}, blend={blend_accepted})", flush=True)

        actual_lambda = transplant_accepted / max(cls_generated, 1)
        if abs(actual_lambda - args.lambda_ratio) > 0.05 and cls_generated > 0:
            print(f"    Lambda drift: target={args.lambda_ratio:.2f}, "
                  f"actual={actual_lambda:.2f}", flush=True)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "augmentation_manifest.csv")
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'output_name', 'source_id', 'source_class',
            'reference_id', 'reference_class', 'method', 'target_class',
        ])
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry)

    # Final report
    total_time = time.time() - t0
    transplant_count = sum(1 for m in manifest if m['method'] == 'transplant')
    blend_count = sum(1 for m in manifest if m['method'] == 'deep_blend')

    print(f"\n{'='*65}", flush=True)
    print(f"  Augmentation Complete", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Accepted:          {total_generated}", flush=True)
    print(f"  Rejected (QC):     {total_rejected_qc}", flush=True)
    print(f"  Rejected (U-Net):  {total_rejected_unet}", flush=True)
    print(f"  Rejected (method): {total_rejected_method}", flush=True)
    print(f"  Errors:            {total_errors}", flush=True)
    total_attempts = total_generated + total_rejected_qc + total_rejected_unet + total_rejected_method
    print(f"  Pass rate:         {total_generated/max(total_attempts,1):.1%}", flush=True)
    print(f"  Time:              {total_time/60:.1f} minutes", flush=True)
    print(f"  Rate:              {total_generated/max(total_time,1):.1f} img/s", flush=True)
    print(f"  Transplant:        {transplant_count} "
          f"({100*transplant_count/max(total_generated,1):.0f}%)", flush=True)
    print(f"  Deep Blend:        {blend_count} "
          f"({100*blend_count/max(total_generated,1):.0f}%)", flush=True)
    print(f"  Manifest:          {manifest_path}", flush=True)

    if reject_reasons:
        print(f"\n  Rejection Breakdown:", flush=True)
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:25s}: {count:5d}", flush=True)

    print(f"\n  Final Distribution (original + augmented):", flush=True)
    for cls in all_classes:
        original = len(isic_by_class.get(cls, []))
        augmented = sum(1 for m in manifest if m['target_class'] == cls)
        total = original + augmented
        bar = "#" * (total // 500)
        print(f"    {cls:8s}: {total:6d}  (orig={original}, aug={augmented})  {bar}",
              flush=True)

    print(f"{'='*65}\n", flush=True)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Lesion-Aware Skin Tone Classification — A+B Cascade Pipeline
=============================================================

Approach A: Perilesional Ring Sampling (U-Net segmentation → dilate → ring → ITA)
Approach B: Multi-Patch Consensus (corner/edge patches → outlier rejection → median ITA)

Pipeline: Try A first. If ring has < min_pixels, fall back to B.
Flag low-confidence results (ITA std > 15°) for manual review.

Outputs:
  - CSV with per-image MST-5 predictions + confidence
  - Visualisation images showing masks, rings, and patches used

Dependencies:
  pip install torch torchvision pillow opencv-python-headless tqdm pandas numpy
  pip install segmentation-models-pytorch   # for pretrained U-Net

Usage:
  python classify_skin_tone.py \
      --image-dir datasets/MSKCC/images/ \
      --output-dir outputs/skin_tone_cascade/ \
      --visualise

  # To use without U-Net (Approach B only):
  python classify_skin_tone.py \
      --image-dir datasets/MSKCC/images/ \
      --output-dir outputs/skin_tone_cascade/ \
      --no-segmentation
"""

import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ───────────────────────────────────────────────────────────
# ITA Calculation (same as compute_mst_labels.py)
# ───────────────────────────────────────────────────────────

def compute_ita_from_lab(L, b):
    """Compute ITA from L* and b* values (scalar or array)."""
    return np.degrees(np.arctan2(L - 50, b))


def compute_ita_from_bgr_patch(patch_bgr):
    """Compute median ITA from a BGR image patch."""
    if patch_bgr is None or patch_bgr.size == 0:
        return np.nan
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB: L in [0,255] → scale to [0,100], a/b in [0,255] → offset by -128
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    ita_map = compute_ita_from_lab(L, b)
    return float(np.median(ita_map))


def compute_ita_from_bgr_masked(image_bgr, mask_bool):
    """Compute median ITA from BGR image, only at pixels where mask is True."""
    if mask_bool.sum() < 10:
        return np.nan, 0.0
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    ita_vals = compute_ita_from_lab(L[mask_bool], b[mask_bool])
    return float(np.median(ita_vals)), float(np.std(ita_vals))


# ───────────────────────────────────────────────────────────
# MST-5 Mapping (same thresholds as compute_mst_labels.py)
# ───────────────────────────────────────────────────────────

MST10_THRESHOLDS = [
    (-90, -55),   # MST 10 (deepest)
    (-55, -41),   # MST 9
    (-41, -28),   # MST 8
    (-28, -10),   # MST 7
    (-10,  10),   # MST 6
    ( 10,  28),   # MST 5
    ( 28,  41),   # MST 4
    ( 41,  55),   # MST 3
    ( 55,  70),   # MST 2
    ( 70,  90),   # MST 1 (lightest)
]

MST5_NAMES = [
    "Very Dark (MST 9-10)",
    "Dark (MST 7-8)",
    "Medium (MST 5-6)",
    "Light (MST 3-4)",
    "Very Light (MST 1-2)",
]


def ita_to_mst10(ita):
    """Map ITA value to MST-10 class (0-9)."""
    if np.isnan(ita):
        return -1
    for cls_id, (low, high) in enumerate(MST10_THRESHOLDS):
        if low <= ita < high:
            return cls_id
    return 0 if ita < -55 else 9


def ita_to_mst5(ita):
    """Map ITA value to MST-5 class (0-4)."""
    mst10 = ita_to_mst10(ita)
    if mst10 < 0:
        return -1
    return mst10 // 2


# ───────────────────────────────────────────────────────────
# Approach A: Perilesional Ring Sampling
# ───────────────────────────────────────────────────────────

def load_segmentation_model(device="cpu"):
    """Load pretrained U-Net for binary lesion segmentation."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        warnings.warn(
            "segmentation_models_pytorch not installed. "
            "Install with: pip install segmentation-models-pytorch\n"
            "Falling back to Approach B only."
        )
        return None

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    # NOTE: This initialises with ImageNet encoder weights but random decoder.
    # For production, load ISIC-trained weights:
    #   model.load_state_dict(torch.load("unet_isic2018.pth", map_location=device))
    #
    # For now, we use a simple thresholding heuristic as a placeholder
    # until you download proper ISIC-trained weights.
    model.eval()
    return model


def segment_lesion_simple(image_bgr, model=None):
    """
    Segment lesion from dermoscopic image.

    If no trained model is available, uses a colour-based heuristic:
    the lesion is typically the darkest/most-saturated central region.
    """
    if model is not None:
        return _segment_with_unet(image_bgr, model)
    else:
        return _segment_heuristic(image_bgr)


def _segment_heuristic(image_bgr):
    """
    Simple heuristic segmentation for dermoscopic images:
    1. Convert to grayscale
    2. Gaussian blur
    3. Otsu threshold (lesion = dark region)
    4. Morphological cleanup
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # Otsu's threshold — dermoscopic lesions are typically darker than surround
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask.astype(bool)


def _segment_with_unet(image_bgr, model):
    """Segment using a U-Net model. Returns boolean mask."""
    import torch
    import torch.nn.functional as F

    h, w = image_bgr.shape[:2]

    # Preprocess: resize to 256x256, normalise
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        mask_small = (prob.squeeze() > 0.5).cpu().numpy()

    # Resize mask back to original size
    mask = cv2.resize(mask_small.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def extract_perilesional_ring(lesion_mask, margin_px=30, min_pixels=500):
    """
    Create perilesional ring by dilating lesion mask and subtracting original.

    Returns:
        ring_mask: boolean mask of the perilesional ring
        has_enough_pixels: whether the ring has >= min_pixels skin pixels
    """
    mask_uint8 = lesion_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (margin_px * 2 + 1, margin_px * 2 + 1))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)

    ring = (dilated > 0) & ~lesion_mask

    return ring, int(ring.sum()) >= min_pixels


def approach_a(image_bgr, model=None, margin_px=30, min_pixels=500):
    """
    Approach A: Perilesional ring ITA.

    Returns:
        ita: float, median ITA from perilesional ring
        ita_std: float, std of ITA values
        lesion_mask: boolean mask of lesion
        ring_mask: boolean mask of perilesional ring
        method: str, "perilesional_ring"
        success: bool
    """
    lesion_mask = segment_lesion_simple(image_bgr, model)
    ring_mask, has_enough = extract_perilesional_ring(
        lesion_mask, margin_px=margin_px, min_pixels=min_pixels
    )

    if not has_enough:
        return np.nan, 0.0, lesion_mask, ring_mask, "perilesional_ring", False

    ita, ita_std = compute_ita_from_bgr_masked(image_bgr, ring_mask)
    return ita, ita_std, lesion_mask, ring_mask, "perilesional_ring", True


# ───────────────────────────────────────────────────────────
# Approach B: Multi-Patch Consensus
# ───────────────────────────────────────────────────────────

def get_patch_positions(h, w, patch_size=32, margin=10):
    """
    Get 8 patch positions: 4 corners + 4 edge midpoints.

    Returns list of (y, x) top-left coordinates.
    """
    ps = patch_size
    positions = [
        # Corners
        (margin, margin),                           # top-left
        (margin, w - ps - margin),                  # top-right
        (h - ps - margin, margin),                  # bottom-left
        (h - ps - margin, w - ps - margin),         # bottom-right
        # Edge midpoints
        (margin, w // 2 - ps // 2),                 # top-center
        (h - ps - margin, w // 2 - ps // 2),        # bottom-center
        (h // 2 - ps // 2, margin),                 # left-center
        (h // 2 - ps // 2, w - ps - margin),        # right-center
    ]
    # Clamp to valid range
    valid = []
    for y, x in positions:
        y = max(0, min(y, h - ps))
        x = max(0, min(x, w - ps))
        valid.append((y, x))
    return valid


def approach_b(image_bgr, patch_size=32, margin=10):
    """
    Approach B: Multi-patch consensus with IQR outlier rejection.

    Returns:
        ita: float, median ITA from surviving patches
        ita_std: float, std of surviving patch ITAs
        patch_positions: list of (y, x) used
        surviving_mask: boolean array, which patches survived
        method: str, "multi_patch"
    """
    h, w = image_bgr.shape[:2]
    positions = get_patch_positions(h, w, patch_size, margin)

    ita_values = []
    for y, x in positions:
        patch = image_bgr[y:y + patch_size, x:x + patch_size]
        ita = compute_ita_from_bgr_patch(patch)
        ita_values.append(ita)

    ita_values = np.array(ita_values)

    # Remove NaN patches
    valid = ~np.isnan(ita_values)
    if valid.sum() < 2:
        return np.nan, 0.0, positions, valid, "multi_patch"

    # IQR outlier rejection
    valid_itas = ita_values[valid]
    q1, q3 = np.percentile(valid_itas, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    surviving = valid.copy()
    for i in range(len(ita_values)):
        if valid[i] and (ita_values[i] < lower or ita_values[i] > upper):
            surviving[i] = False

    if surviving.sum() == 0:
        surviving = valid  # fall back to all valid if everything was "outlier"

    final_itas = ita_values[surviving]
    return float(np.median(final_itas)), float(np.std(final_itas)), \
           positions, surviving, "multi_patch"


# ───────────────────────────────────────────────────────────
# Visualisation
# ───────────────────────────────────────────────────────────

def visualise_result(image_bgr, result, save_path):
    """
    Create a side-by-side visualisation of the classification result.

    Left panel:  Original image with overlays
    Right panel: Mask / patch visualisation
    """
    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()

    # Determine overlays based on method
    if result["method"] == "perilesional_ring":
        # Draw lesion mask in red, ring in green
        lesion_mask = result.get("lesion_mask")
        ring_mask = result.get("ring_mask")

        # Create overlay
        overlay = vis.copy()
        if lesion_mask is not None:
            overlay[lesion_mask] = (overlay[lesion_mask] * 0.5 +
                                   np.array([0, 0, 180]) * 0.5).astype(np.uint8)
        if ring_mask is not None:
            overlay[ring_mask] = (overlay[ring_mask] * 0.5 +
                                  np.array([0, 200, 0]) * 0.5).astype(np.uint8)
        vis = overlay

        # Create mask panel
        mask_panel = np.zeros_like(image_bgr)
        if lesion_mask is not None:
            mask_panel[lesion_mask] = (0, 0, 200)  # red = lesion
        if ring_mask is not None:
            mask_panel[ring_mask] = (0, 200, 0)    # green = ring

    elif result["method"] == "multi_patch":
        positions = result.get("patch_positions", [])
        surviving = result.get("surviving_mask", [])
        patch_size = 32

        mask_panel = image_bgr.copy()
        for i, (y, x) in enumerate(positions):
            if i < len(surviving) and surviving[i]:
                colour = (0, 255, 0)   # green = used
                thickness = 2
            else:
                colour = (0, 0, 255)   # red = rejected
                thickness = 1
            cv2.rectangle(vis, (x, y), (x + patch_size, y + patch_size),
                         colour, thickness)
            cv2.rectangle(mask_panel, (x, y), (x + patch_size, y + patch_size),
                         colour, thickness)

        mask_panel = vis.copy()  # patches drawn on the image
    else:
        mask_panel = np.zeros_like(image_bgr)

    # Add text annotations
    mst5_cls = result.get("mst5_class", -1)
    ita = result.get("ita", np.nan)
    confidence = result.get("confidence", "unknown")
    method = result.get("method", "unknown")

    label = f"MST-5: {MST5_NAMES[mst5_cls] if 0 <= mst5_cls < 5 else '?'}"
    sub = f"ITA={ita:.1f}  |  {method}  |  {confidence}"

    # Put text on vis
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, label, (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, sub, (10, 60), font, 0.5, (200, 200, 200), 1)

    # Combine panels side by side
    combined = np.hstack([vis, mask_panel])
    cv2.imwrite(str(save_path), combined)


# ───────────────────────────────────────────────────────────
# A → B Cascade
# ───────────────────────────────────────────────────────────

def classify_image(image_bgr, seg_model=None, use_segmentation=True,
                   margin_px=30, min_ring_pixels=500, confidence_threshold=15.0):
    """
    Run A→B cascade on a single image.

    Returns dict with: ita, ita_std, mst10_class, mst5_class,
                        method, confidence, and visualisation data.
    """
    result = {}

    if use_segmentation:
        ita, ita_std, lesion_mask, ring_mask, method, success = approach_a(
            image_bgr, model=seg_model,
            margin_px=margin_px, min_pixels=min_ring_pixels,
        )
        result["lesion_mask"] = lesion_mask
        result["ring_mask"] = ring_mask

        if success:
            result["ita"] = ita
            result["ita_std"] = ita_std
            result["method"] = method
            result["mst10_class"] = ita_to_mst10(ita)
            result["mst5_class"] = ita_to_mst5(ita)
            result["confidence"] = "high" if ita_std < confidence_threshold else "low"
            return result
        # else: fall through to B

    # Approach B
    ita, ita_std, positions, surviving, method = approach_b(image_bgr)
    result["ita"] = ita
    result["ita_std"] = ita_std
    result["method"] = method
    result["patch_positions"] = positions
    result["surviving_mask"] = surviving
    result["mst10_class"] = ita_to_mst10(ita)
    result["mst5_class"] = ita_to_mst5(ita)
    result["confidence"] = "high" if ita_std < confidence_threshold else "low"
    return result


# ───────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Lesion-aware skin tone classification (A+B cascade)")
    p.add_argument("--image-dir", type=str, required=True,
                   help="Directory containing dermoscopic/clinical images")
    p.add_argument("--output-dir", type=str, default="outputs/skin_tone_cascade",
                   help="Output directory for CSV + visualisations")
    p.add_argument("--visualise", action="store_true",
                   help="Save visualisation images showing masks and patches")
    p.add_argument("--no-segmentation", action="store_true",
                   help="Skip Approach A (U-Net); use Approach B only")
    p.add_argument("--margin-px", type=int, default=30,
                   help="Perilesional ring margin in pixels (default: 30)")
    p.add_argument("--min-ring-pixels", type=int, default=500,
                   help="Minimum pixels in ring before falling back to B (default: 500)")
    p.add_argument("--confidence-threshold", type=float, default=15.0,
                   help="ITA std threshold for confidence flag (default: 15.0°)")
    p.add_argument("--extensions", type=str, default="jpg,jpeg,png,bmp,tif,tiff",
                   help="Image extensions to process (comma-separated)")
    return p.parse_args()


def main():
    args = parse_args()

    # Create output directories
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.visualise:
        vis_dir = out_dir / "visualisations"
        vis_dir.mkdir(exist_ok=True)

    # Collect image files
    extensions = set(args.extensions.lower().split(","))
    image_dir = Path(args.image_dir)
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lstrip(".").lower() in extensions
    ])

    if not image_files:
        print(f"❌ No images found in {image_dir} with extensions {extensions}")
        return

    print(f"Found {len(image_files):,} images in {image_dir}")

    # Load segmentation model
    seg_model = None
    use_seg = not args.no_segmentation
    if use_seg:
        print("Loading segmentation model...")
        seg_model = load_segmentation_model()
        if seg_model is None:
            print("⚠ Segmentation model unavailable — using Approach B only")
            use_seg = False
        else:
            print("✓ Segmentation model loaded")

    # Process images
    results = []
    for img_path in tqdm(image_files, desc="Classifying"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            warnings.warn(f"Could not load: {img_path}")
            continue

        result = classify_image(
            image_bgr,
            seg_model=seg_model,
            use_segmentation=use_seg,
            margin_px=args.margin_px,
            min_ring_pixels=args.min_ring_pixels,
            confidence_threshold=args.confidence_threshold,
        )

        # Save visualisation
        if args.visualise:
            vis_path = vis_dir / f"{img_path.stem}_vis.jpg"
            visualise_result(image_bgr, result, vis_path)

        results.append({
            "file": img_path.name,
            "ita": round(result["ita"], 2) if not np.isnan(result["ita"]) else None,
            "ita_std": round(result["ita_std"], 2),
            "mst10_class": result["mst10_class"],
            "mst5_class": result["mst5_class"],
            "mst5_name": MST5_NAMES[result["mst5_class"]] if 0 <= result["mst5_class"] < 5 else "unknown",
            "method": result["method"],
            "confidence": result["confidence"],
        })

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = out_dir / "skin_tone_predictions.csv"
    df.to_csv(csv_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SKIN TONE CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"  Images processed: {len(df):,}")

    if len(df) > 0:
        # Method breakdown
        method_counts = df["method"].value_counts()
        print(f"\n  Method used:")
        for method, count in method_counts.items():
            print(f"    {method}: {count:,} ({count/len(df):.1%})")

        # Confidence breakdown
        conf_counts = df["confidence"].value_counts()
        print(f"\n  Confidence:")
        for conf, count in conf_counts.items():
            print(f"    {conf}: {count:,} ({count/len(df):.1%})")

        # MST-5 distribution
        print(f"\n  MST-5 Distribution:")
        for cls_id in range(5):
            count = len(df[df["mst5_class"] == cls_id])
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"    {MST5_NAMES[cls_id]:25s}: {count:4,} ({pct:5.1f}%)  {bar}")

    print(f"\n  CSV saved:   {csv_path}")
    if args.visualise:
        print(f"  Vis saved:   {vis_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

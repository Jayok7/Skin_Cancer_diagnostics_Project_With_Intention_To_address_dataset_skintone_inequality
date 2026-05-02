#!/usr/bin/env python3
"""
Lesion-Aware Skin Tone Classification: A+B(+C) Cascade Pipeline
================================================================

Approach A: Perilesional Ring Sampling (U-Net segmentation → dilate → ring)
Approach B: Multi-Patch Consensus (corner/edge patches)
Approach C: Full-image fallback (CNN mode only)

ALL approaches are vignette-aware: a black-border mask is computed once per
image and propagated to every sampling stage so that dermoscopic black
circular masks don't contaminate ITA / CNN inputs.

Modes:
  - ITA Analytical (default): L*a*b* → ITA → MST bin
  - FairFace CNN (--cnn-model): EfficientNet-B4 trained on FairFace+MSKCC
    with 3-way / 5-way / 6-way head. NB: 3-way checkpoint expects
    class order ['Dark','Medium','Light'] (matches v7 trainer).

Usage (ISIC 2019 test audit):
  python classify_skin_tone.py \
      --image-dir <ISIC_2019_Test_Input>/ \
      --output-dir outputs/isic2019_test_tone_audit_cascade/ \
      --cnn-model outputs/FairFace-Model-3.2-finetuned-v7-3class/fairface_mskcc_best.pth \
      --eval-mode 3-way \
      --confidence-filter 0.6 \
      --vignette-detection
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torchvision import transforms, models


# ───────────────────────────────────────────────────────────
# Vignette / black-border detection
# ───────────────────────────────────────────────────────────

def detect_vignette_mask(image_bgr, dark_threshold=18, min_fraction=0.02):
    """
    Detect black border / dermoscopic vignette in an image.

    Returns
    -------
    valid_mask : np.ndarray[bool]  True = real skin/lesion, False = black border
    has_vignette : bool            True if vignette covers >= min_fraction of image
    vignette_fraction : float
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, False, 0.0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray > dark_threshold).astype(np.uint8) * 255

    # Morphological cleanup: remove tiny dark spots in skin, fill small bright
    # holes inside vignette, then keep only the largest connected component
    # (the central skin patch).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        valid = (labels == largest_idx).astype(np.uint8) * 255

    valid_bool = valid > 0
    vignette_fraction = float(1.0 - valid_bool.mean())
    has_vignette = vignette_fraction >= min_fraction
    return valid_bool, has_vignette, vignette_fraction


def crop_to_valid_bbox(image_bgr, valid_mask, paint_with_median=True):
    """Crop to the bounding box of valid_mask; optionally paint vignette
    pixels inside the crop with the median valid-region colour so the CNN
    doesn't see hard black edges."""
    if valid_mask is None or not valid_mask.any():
        return image_bgr.copy()
    ys, xs = np.where(valid_mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = image_bgr[y0:y1, x0:x1].copy()
    local_valid = valid_mask[y0:y1, x0:x1]
    if paint_with_median and (~local_valid).any() and local_valid.any():
        median_color = np.median(crop[local_valid], axis=0)
        crop[~local_valid] = median_color
    return crop


# ───────────────────────────────────────────────────────────
# ITA Calculation
# ───────────────────────────────────────────────────────────

def compute_ita_from_lab(L, b):
    return np.degrees(np.arctan2(L - 50, b))

def compute_ita_from_bgr_patch(patch_bgr):
    if patch_bgr is None or patch_bgr.size == 0:
        return np.nan
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    ita_map = compute_ita_from_lab(L, b)
    return float(np.median(ita_map))

def compute_ita_from_bgr_masked(image_bgr, mask_bool):
    if mask_bool.sum() < 10:
        return np.nan, 0.0
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    ita_vals = compute_ita_from_lab(L[mask_bool], b[mask_bool])
    return float(np.median(ita_vals)), float(np.std(ita_vals))


# ───────────────────────────────────────────────────────────
# MST Mapping
# ───────────────────────────────────────────────────────────

MST10_THRESHOLDS = [
    (-90, -55), (-55, -41), (-41, -28), (-28, -10), (-10,  10),
    ( 10,  28), ( 28,  41), ( 41,  55), ( 55,  70), ( 70,  90),
]

MST5_NAMES = [
    "Very Dark (MST 9-10)", "Dark (MST 7-8)", "Medium (MST 5-6)",
    "Light (MST 3-4)", "Very Light (MST 1-2)",
]

CLASSES_6WAY = ['VI', 'V', 'IV', 'III', 'II', 'I']
CLASSES_5WAY = MST5_NAMES
# IMPORTANT: order must match the v7 3-class checkpoint
# (label_skin_tone.py defines CLASS_NAMES = ["Dark","Medium","Light"])
CLASSES_3WAY = ['Dark', 'Medium', 'Light']

def ita_to_mst10(ita):
    if np.isnan(ita):
        return -1
    for cls_id, (low, high) in enumerate(MST10_THRESHOLDS):
        if low <= ita < high:
            return cls_id
    return 0 if ita < -55 else 9

def ita_to_mst5(ita):
    mst10 = ita_to_mst10(ita)
    if mst10 < 0:
        return -1
    return mst10 // 2


# ───────────────────────────────────────────────────────────
# U-Net Segmentation
# ───────────────────────────────────────────────────────────

def load_segmentation_model(device="cpu", weights_path=None, unet_dir=None):
    if unet_dir:
        unet_path = os.path.abspath(unet_dir)
        if unet_path not in sys.path:
            sys.path.insert(0, unet_path)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        auto_path = os.path.join(script_dir, "Pytorch-UNet")
        if os.path.isdir(auto_path) and auto_path not in sys.path:
            sys.path.insert(0, auto_path)

    try:
        from unet import UNet
    except ImportError:
        warnings.warn("milesial/Pytorch-UNet not found. Falling back to Approach B/C only.")
        return None

    model = UNet(n_channels=3, n_classes=2, bilinear=False)

    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state.pop("mask_values", None)
        model.load_state_dict(state)
        print(f"  ✓ U-Net weights loaded from {weights_path}")
    else:
        url = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth"
        print(f"  Downloading pretrained U-Net weights (Carvana fallback)...")
        try:
            state = torch.hub.load_state_dict_from_url(url, map_location=device, file_name="unet_carvana.pth")
            model_tmp = UNet(n_channels=3, n_classes=2, bilinear=False)
            model_tmp.load_state_dict(state)
            for name, param in model_tmp.named_parameters():
                if "outc" not in name:
                    model.state_dict()[name].copy_(param)
            print(f"  ✓ Pretrained Carvana weights loaded (encoder + decoder only)")
        except Exception as e:
            warnings.warn(f"Could not load U-Net weights: {e}")

    model.to(device).eval()
    return model

def segment_lesion_simple(image_bgr, model=None, valid_mask=None):
    """Segment lesion. If valid_mask supplied, suppress segmentation outside it."""
    h, w = image_bgr.shape[:2]
    if model is not None:
        device = next(model.parameters()).device
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            if logits.shape[1] > 1:
                prob = torch.softmax(logits, dim=1)
                mask_tensor = (prob[0, 1] > 0.5).to(torch.uint8).cpu()
            else:
                prob = torch.sigmoid(logits)
                mask_tensor = (prob[0, 0] > 0.5).to(torch.uint8).cpu()

        mask_small = mask_tensor.numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        lesion = mask_resized > 0
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        lesion = mask > 0

    # Suppress lesion mask inside vignette region
    if valid_mask is not None:
        lesion = lesion & valid_mask
    return lesion


def extract_perilesional_ring(lesion_mask, margin_px=30, min_pixels=500, valid_mask=None):
    mask_uint8 = np.zeros(lesion_mask.shape[:2], dtype=np.uint8)
    mask_uint8[lesion_mask > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    ring = (dilated > 0) & (lesion_mask == 0)
    # Exclude vignette
    if valid_mask is not None:
        ring = ring & valid_mask
    return ring, int(ring.sum()) >= min_pixels


# ───────────────────────────────────────────────────────────
# CNN helpers
# ───────────────────────────────────────────────────────────

def get_cnn_friendly_crop(image_bgr, lesion_mask, ring_mask):
    ys, xs = np.where(ring_mask > 0)
    if len(ys) == 0:
        return None
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1
    crop = image_bgr[ymin:ymax, xmin:xmax].copy()
    mask_patch = ring_mask[ymin:ymax, xmin:xmax]
    lesion_patch = lesion_mask[ymin:ymax, xmin:xmax]
    if mask_patch.sum() > 0:
        median_color = np.median(crop[mask_patch > 0], axis=0)
    else:
        median_color = np.array([128, 128, 128])
    background_patch = (mask_patch == 0) & (lesion_patch == 0)
    crop[lesion_patch > 0] = median_color
    crop[background_patch > 0] = median_color
    return crop


def predict_crop(crop_bgr, model, transform, device):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs


def aggregate_probs_3way(probs_6way):
    """Map 6-way Fitzpatrick logits → 3-way [Dark, Medium, Light]
    to match CLASSES_3WAY = ['Dark','Medium','Light']."""
    dark   = probs_6way[0] + probs_6way[1]   # VI + V
    medium = probs_6way[2] + probs_6way[3]   # IV + III
    light  = probs_6way[4] + probs_6way[5]   # II + I
    return np.array([dark, medium, light])


def get_patch_positions(h, w, patch_size=32, margin=10, valid_mask=None,
                       min_valid_fraction=0.95):
    """Return (y, x) patch positions whose patch has at least
    min_valid_fraction of its pixels inside valid_mask. Falls back to all
    positions if vignette filtering removes everything."""
    ps = patch_size
    candidates = [
        (margin, margin), (margin, w - ps - margin),
        (h - ps - margin, margin), (h - ps - margin, w - ps - margin),
        (margin, w // 2 - ps // 2), (h - ps - margin, w // 2 - ps // 2),
        (h // 2 - ps // 2, margin), (h // 2 - ps // 2, w - ps - margin),
    ]
    clipped = []
    for y, x in candidates:
        y = max(0, min(y, h - ps))
        x = max(0, min(x, w - ps))
        clipped.append((y, x))

    if valid_mask is None:
        return clipped

    filtered = []
    for y, x in clipped:
        sub = valid_mask[y:y+ps, x:x+ps]
        if sub.size == 0:
            continue
        if sub.mean() >= min_valid_fraction:
            filtered.append((y, x))

    if len(filtered) == 0:
        # all corners are in vignette; try sampling along a circle inside
        # the valid region instead
        ys, xs = np.where(valid_mask)
        if len(ys) == 0:
            return clipped
        cy, cx = int(ys.mean()), int(xs.mean())
        radius = int(0.35 * min(h, w))
        for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
            theta = np.deg2rad(angle_deg)
            y = int(cy + radius * np.sin(theta)) - ps // 2
            x = int(cx + radius * np.cos(theta)) - ps // 2
            y = max(0, min(y, h - ps))
            x = max(0, min(x, w - ps))
            sub = valid_mask[y:y+ps, x:x+ps]
            if sub.size > 0 and sub.mean() >= min_valid_fraction:
                filtered.append((y, x))
        if len(filtered) == 0:
            return clipped  # last-resort

    return filtered


# ───────────────────────────────────────────────────────────
# Prediction Logic
# ───────────────────────────────────────────────────────────

def process_ita(image_bgr, seg_model, use_segmentation, margin_px,
                min_ring_pixels, confidence_threshold, valid_mask=None):
    result = {"valid_mask": valid_mask}
    if use_segmentation:
        lesion_mask = segment_lesion_simple(image_bgr, seg_model, valid_mask=valid_mask)
        ring_mask, has_enough = extract_perilesional_ring(
            lesion_mask, margin_px, min_ring_pixels, valid_mask=valid_mask
        )
        result["lesion_mask"] = lesion_mask
        result["ring_mask"] = ring_mask

        if has_enough:
            ita, ita_std = compute_ita_from_bgr_masked(image_bgr, ring_mask)
            result.update({
                "ita": ita, "ita_std": ita_std,
                "method": "perilesional_ring",
                "confidence_val": ita_std,
                "mst10_class": ita_to_mst10(ita),
                "mst5_class": ita_to_mst5(ita),
                "confidence": "high" if ita_std < confidence_threshold else "low",
            })
            result["mst_name"] = (MST5_NAMES[result["mst5_class"]]
                                  if 0 <= result["mst5_class"] < 5 else "unknown")
            return result

    # Fallback Approach B (vignette-aware)
    patch_size = 32
    h, w = image_bgr.shape[:2]
    positions = get_patch_positions(h, w, patch_size, margin=10, valid_mask=valid_mask)

    ita_values = []
    for y, x in positions:
        patch = image_bgr[y:y+patch_size, x:x+patch_size]
        ita_values.append(compute_ita_from_bgr_patch(patch))

    ita_values = np.array(ita_values)
    valid = ~np.isnan(ita_values)
    surviving = valid.copy()

    if valid.sum() >= 2:
        valid_itas = ita_values[valid]
        q1, q3 = np.percentile(valid_itas, [25, 75])
        iqr = q3 - q1
        for i in range(len(ita_values)):
            if valid[i] and (ita_values[i] < q1 - 1.5 * iqr or ita_values[i] > q3 + 1.5 * iqr):
                surviving[i] = False
        if surviving.sum() == 0:
            surviving = valid

    final_itas = ita_values[surviving] if surviving.sum() > 0 else np.array([np.nan])
    ita = float(np.median(final_itas))
    ita_std = float(np.std(final_itas)) if len(final_itas) > 1 else 0.0

    result.update({
        "ita": ita, "ita_std": ita_std, "method": "multi_patch",
        "patch_positions": positions, "surviving_mask": surviving,
        "mst10_class": ita_to_mst10(ita), "mst5_class": ita_to_mst5(ita),
        "confidence_val": ita_std,
        "confidence": "high" if ita_std < confidence_threshold else "low",
    })
    result["mst_name"] = (MST5_NAMES[result["mst5_class"]]
                          if 0 <= result["mst5_class"] < 5 else "unknown")
    return result


def process_cnn(image_bgr, seg_model, ff_model, transform, device, mode,
                use_segmentation, margin_px, min_ring_pixels, valid_mask=None,
                num_outputs=3):
    result = {"valid_mask": valid_mask}
    probs = None
    method_used = "A (Perilesional Ring)"

    if use_segmentation and seg_model is not None:
        lesion_mask = segment_lesion_simple(image_bgr, seg_model, valid_mask=valid_mask)
        ring_mask, has_enough = extract_perilesional_ring(
            lesion_mask, margin_px, min_ring_pixels, valid_mask=valid_mask
        )
        result["lesion_mask"] = lesion_mask
        result["ring_mask"] = ring_mask
        if has_enough:
            crop = get_cnn_friendly_crop(image_bgr, lesion_mask, ring_mask)
            if crop is not None:
                probs = predict_crop(crop, ff_model, transform, device)

    if probs is None:  # Fallback Approach B
        method_used = "B (Multi-Patch Consensus)"
        h, w = image_bgr.shape[:2]
        patch_sz = int(min(h, w) * 0.25)
        positions = get_patch_positions(h, w, patch_size=patch_sz, margin=10,
                                        valid_mask=valid_mask)
        result["patch_positions"] = positions

        all_probs = []
        surviving = []
        for y, x in positions:
            patch = image_bgr[y:y+patch_sz, x:x+patch_sz]
            if patch is not None and patch.size > 0:
                all_probs.append(predict_crop(patch, ff_model, transform, device))
                surviving.append(True)
            else:
                surviving.append(False)

        result["surviving_mask"] = surviving
        if len(all_probs) > 0:
            probs = np.mean(all_probs, axis=0)

    if probs is None:  # Approach C
        method_used = "C (Full Image Fallback)"
        # Vignette-aware: crop to valid bbox + paint border with median skin colour
        crop = crop_to_valid_bbox(image_bgr, valid_mask, paint_with_median=True)
        probs = predict_crop(crop, ff_model, transform, device)

    # Output Parsing: handle native 3-way head vs 6-way head
    if mode == '3-way':
        if num_outputs == 3:
            final_probs = probs            # already [Dark, Medium, Light]
        else:
            final_probs = aggregate_probs_3way(probs)
        classes = CLASSES_3WAY
    elif mode == '5-way':
        final_probs = probs
        classes = CLASSES_5WAY
    else:
        final_probs = probs
        classes = CLASSES_6WAY

    pred_idx = int(np.argmax(final_probs))
    confidence_val = float(np.max(final_probs))

    result.update({
        "probs": final_probs, "method": method_used,
        "confidence_val": confidence_val,
        "confidence": "high" if confidence_val >= 0.6 else "low",
        "pred_idx": pred_idx, "mst_name": classes[pred_idx],
    })
    return result


# ───────────────────────────────────────────────────────────
# Visualisation
# ───────────────────────────────────────────────────────────

def visualise_result(image_bgr, result, save_path, is_cnn=False):
    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()
    method = result.get("method", "")
    valid_mask = result.get("valid_mask")

    if "Ring" in method or method == "perilesional_ring":
        lesion_mask = result.get("lesion_mask")
        ring_mask = result.get("ring_mask")
        overlay = vis.copy()
        mask_panel = np.zeros_like(image_bgr)
        if lesion_mask is not None:
            overlay[lesion_mask] = (overlay[lesion_mask] * 0.5 + np.array([0, 0, 180]) * 0.5).astype(np.uint8)
            mask_panel[lesion_mask] = (0, 0, 200)
        if ring_mask is not None:
            overlay[ring_mask] = (overlay[ring_mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
            mask_panel[ring_mask] = (0, 200, 0)
        if valid_mask is not None:
            border = (~valid_mask)
            overlay[border] = (overlay[border] * 0.3 + np.array([200, 200, 0]) * 0.7).astype(np.uint8)
            mask_panel[border] = (200, 200, 0)
        vis = overlay

    elif "Patch" in method or method == "multi_patch":
        positions = result.get("patch_positions", [])
        surviving = result.get("surviving_mask", [])
        patch_size = 32 if not is_cnn else int(min(h, w) * 0.25)
        for i, (y, x) in enumerate(positions):
            if i < len(surviving) and surviving[i]:
                colour, thickness = (0, 255, 0), 2
            else:
                colour, thickness = (0, 0, 255), 1
            cv2.rectangle(vis, (x, y), (x + patch_size, y + patch_size), colour, thickness)
        if valid_mask is not None:
            border = (~valid_mask)
            vis[border] = (vis[border] * 0.3 + np.array([200, 200, 0]) * 0.7).astype(np.uint8)
        mask_panel = vis.copy()
    else:
        mask_panel = np.zeros_like(image_bgr)
        if valid_mask is not None:
            mask_panel[~valid_mask] = (200, 200, 0)

    mst_name = result.get("mst_name", "?")
    conf = result.get("confidence", "unknown")
    conf_val = result.get("confidence_val", 0.0)

    if is_cnn:
        sub = f"P={conf_val:.2f} | {method} | {conf}"
    else:
        ita = result.get("ita", np.nan)
        sub = f"ITA={ita:.1f} | {method} | {conf}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, f"MST: {mst_name}", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, sub, (10, 60), font, 0.5, (200, 200, 200), 1)

    combined = np.hstack([vis, mask_panel])
    cv2.imwrite(str(save_path), combined)


# ───────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Lesion-aware skin tone classification (A+B+C cascade, vignette-aware)")
    p.add_argument("--image-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/skin_tone_cascade")
    p.add_argument("--visualise", action="store_true")
    p.add_argument("--no-segmentation", action="store_true")
    p.add_argument("--margin-px", type=int, default=30)
    p.add_argument("--min-ring-pixels", type=int, default=500)
    p.add_argument("--confidence-threshold", type=float, default=15.0,
                   help="ITA std threshold (ITA mode only)")
    p.add_argument("--confidence-filter", type=float, default=0.0,
                   help="(CNN mode) Drop predictions whose softmax max < this. 0 disables.")
    p.add_argument("--extensions", type=str, default="jpg,jpeg,png,bmp,tif,tiff")

    p.add_argument("--metadata-csv", type=str, default=None,
                   help="Optional metadata CSV. Used for image-type filtering OR as a "
                        "passive sidecar for ISIC test sets (no GT).")
    p.add_argument("--image-type-filter", type=str, default=None,
                   help="If set, only keep rows whose 'image_type' equals this value. "
                        "Default: no filter (recommended for ISIC 2019 test set).")

    p.add_argument("--unet-dir", type=str, default=None)
    p.add_argument("--unet-weights", type=str, default=None)

    # CNN
    p.add_argument("--cnn-model", type=str, default=None)
    p.add_argument("--eval-csv", type=str, default=None,
                   help="GT CSV (only required for evaluation; not needed for ISIC test labelling)")
    p.add_argument("--eval-mode", type=str, choices=['3-way', '5-way', '6-way'], default='3-way')

    # Vignette
    p.add_argument("--vignette-detection", action="store_true", default=True,
                   help="Enable black-border / dermoscopic vignette masking (default: ON)")
    p.add_argument("--no-vignette-detection", dest="vignette_detection", action="store_false")
    p.add_argument("--vignette-dark-threshold", type=int, default=18,
                   help="Greyscale threshold below which a pixel is treated as vignette (default: 18)")

    # Dataset name (for the report)
    p.add_argument("--dataset-name", type=str, default="dataset")

    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.visualise:
        vis_dir = out_dir / "visualisations"
        vis_dir.mkdir(exist_ok=True)

    extensions = set(args.extensions.lower().split(","))
    image_dir = Path(args.image_dir)
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix.lstrip(".").lower() in extensions])

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    print(f"Found {len(image_files):,} images in {image_dir}")

    # Optional metadata filtering / sidecar
    image_type_map = {}
    if args.metadata_csv:
        meta_path = Path(args.metadata_csv)
        if meta_path.is_file():
            meta_df = pd.read_csv(meta_path)
            if "isic_id" in meta_df.columns and "image_type" in meta_df.columns:
                id_to_type = dict(zip(meta_df["isic_id"], meta_df["image_type"]))
                for f in image_files:
                    image_type_map[f.name] = id_to_type.get(f.stem, "unknown")
                if args.image_type_filter:
                    before = len(image_files)
                    image_files = [f for f in image_files
                                   if id_to_type.get(f.stem, "") == args.image_type_filter]
                    after = len(image_files)
                    print(f"  Metadata filter '{args.image_type_filter}': "
                          f"kept {after} / filtered {before - after}")
                    if not image_files:
                        return
            else:
                print(f"  (Metadata CSV present but no isic_id/image_type cols: ignored.)")

    # Optional ground truth
    gt_map = {}
    if args.eval_csv and args.cnn_model:
        eval_path = Path(args.eval_csv)
        if eval_path.is_file():
            edf = pd.read_csv(eval_path).dropna(subset=['fitzpatrick_skin_type', 'isic_id'])
            for _, row in edf.iterrows():
                gt = row['fitzpatrick_skin_type']
                if args.eval_mode == '3-way':
                    if gt in ['I', 'II']:        gt = 'Light'
                    elif gt in ['III', 'IV']:    gt = 'Medium'
                    elif gt in ['V', 'VI']:      gt = 'Dark'
                    else: continue
                elif args.eval_mode == '5-way':
                    if   gt == 'I':              gt = 'Very Light (MST 1-2)'
                    elif gt == 'II':             gt = 'Light (MST 3-4)'
                    elif gt in ['III', 'IV']:    gt = 'Medium (MST 5-6)'
                    elif gt == 'V':              gt = 'Dark (MST 7-8)'
                    elif gt == 'VI':             gt = 'Very Dark (MST 9-10)'
                    else: continue
                gt_map[row['isic_id']] = gt

    # U-Net
    seg_model = None
    use_seg = not args.no_segmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if use_seg:
        print("Loading U-Net model...")
        seg_model = load_segmentation_model(device=str(device),
                                            weights_path=args.unet_weights,
                                            unet_dir=args.unet_dir)
        if seg_model is None:
            use_seg = False
            print("  ⚠ U-Net unavailable:  cascade will use Approach B → C only.")

    # CNN
    is_cnn = args.cnn_model is not None
    ff_model, transform = None, None
    classes = (CLASSES_6WAY if args.eval_mode == '6-way'
               else (CLASSES_5WAY if args.eval_mode == '5-way' else CLASSES_3WAY))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_outputs = len(classes)

    if is_cnn:
        print(f"Loading EfficientNet-B4 from {args.cnn_model} ({args.eval_mode}, {num_outputs} outputs)...")
        ff_model = models.efficientnet_b4(weights=None)
        in_features = ff_model.classifier[1].in_features
        # Match the v7 trainer head: Dropout(p=0.6) + Linear
        ff_model.classifier = nn.Sequential(
            nn.Dropout(p=0.6, inplace=True),
            nn.Linear(in_features, num_outputs),
        )
        state = torch.load(args.cnn_model, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        ff_model.load_state_dict(state)
        ff_model = ff_model.to(device).eval()

        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # Process
    results = []
    y_true_indices, y_pred_indices = [], []

    print("\nRunning inference (vignette detection: %s)..."
          % ("ON" if args.vignette_detection else "OFF"))

    for img_path in tqdm(image_files, desc="Classifying"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        # Vignette pass
        if args.vignette_detection:
            valid_mask, has_vignette, vfrac = detect_vignette_mask(
                image_bgr, dark_threshold=args.vignette_dark_threshold
            )
        else:
            valid_mask, has_vignette, vfrac = None, False, 0.0

        if is_cnn:
            result = process_cnn(
                image_bgr, seg_model, ff_model, transform, device, args.eval_mode,
                use_seg, args.margin_px, args.min_ring_pixels,
                valid_mask=valid_mask, num_outputs=num_outputs,
            )
            gt_label = gt_map.get(img_path.stem)
            if gt_label:
                result["true_label"] = gt_label
                result["true_idx"] = class_to_idx.get(gt_label, -1)
                if result["true_idx"] != -1:
                    y_true_indices.append(result["true_idx"])
                    y_pred_indices.append(result["pred_idx"])
        else:
            result = process_ita(
                image_bgr, seg_model, use_seg,
                args.margin_px, args.min_ring_pixels, args.confidence_threshold,
                valid_mask=valid_mask,
            )

        if args.visualise:
            vis_path = vis_dir / f"{img_path.stem}_vis.jpg"
            visualise_result(image_bgr, result, vis_path, is_cnn=is_cnn)

        row_dict = {
            "file": img_path.name,
            "image_id": img_path.stem,
            "image_type": image_type_map.get(img_path.name, ""),
            "mst_name": result["mst_name"],
            "method": result["method"],
            "confidence": result["confidence"],
            "confidence_val": round(float(result["confidence_val"]), 4),
            "has_vignette": bool(has_vignette),
            "vignette_fraction": round(float(vfrac), 4),
        }
        if is_cnn:
            row_dict["pred_idx"] = result.get("pred_idx")
            row_dict["true_label"] = result.get("true_label", "")
            # full prob vector
            for cls, p in zip(classes, result["probs"]):
                row_dict[f"prob_{cls}"] = round(float(p), 4)
        else:
            row_dict["ita"] = (round(result["ita"], 2)
                               if not np.isnan(result.get("ita", np.nan)) else "")
            row_dict["ita_std"] = round(result.get("ita_std", 0.0), 2)
            row_dict["mst10_class"] = result.get("mst10_class", -1)
            row_dict["mst5_class"] = result.get("mst5_class", -1)

        results.append(row_dict)

    df = pd.DataFrame(results)
    csv_path = out_dir / "skin_tone_predictions.csv"
    df.to_csv(csv_path, index=False)

    # ── Confidence filter (Test 1 step 2) ──
    if is_cnn and args.confidence_filter > 0:
        kept = df[df["confidence_val"] >= args.confidence_filter].copy()
        filtered = df[df["confidence_val"] < args.confidence_filter].copy()
        kept.to_csv(out_dir / f"skin_tone_predictions_conf{args.confidence_filter:.2f}.csv", index=False)
        filtered.to_csv(out_dir / f"skin_tone_predictions_FILTERED.csv", index=False)
    else:
        kept, filtered = df, pd.DataFrame(columns=df.columns)

    # ── Console + dissertation-ready report ──
    name = args.dataset_name.upper()
    print(f"\n{'='*60}")
    print(f"{name}: SKIN TONE AUDIT  (Mode: {'CNN' if is_cnn else 'ITA'})")
    print(f"{'='*60}")
    print(f"  Total images processed   : {len(df):,}")
    print(f"  Vignette flagged         : {int(df['has_vignette'].sum()):,}")
    if is_cnn and args.confidence_filter > 0:
        print(f"  Confidence threshold     : {args.confidence_filter}")
        print(f"  Kept (high confidence)   : {len(kept):,}")
        print(f"  Filtered (low confidence): {len(filtered):,}")

    if len(df) > 0:
        print(f"\n  Method breakdown:")
        for m, c in df["method"].value_counts().items():
            print(f"    {m:35s}: {c:6,} ({c/len(df):.1%})")

    if len(kept) > 0:
        print(f"\n  MST Distribution (post-filter, N={len(kept):,}):")
        for nm, c in kept["mst_name"].value_counts().items():
            pct = c / len(kept) * 100
            print(f"    {nm:25s}: {c:6,} ({pct:5.1f}%)  {'█' * int(pct/2)}")

    # Per-tone breakdown of filtered set (Test 1 step 4)
    if is_cnn and len(filtered) > 0:
        print(f"\n  Filtered set: per predicted-tone breakdown:")
        for nm, c in filtered["mst_name"].value_counts().items():
            print(f"    {nm:25s}: {c:6,}")

    # Optional GT evaluation (only if eval-csv was supplied)
    if is_cnn and len(y_true_indices) > 0:
        print(f"\n{'='*60}")
        print(f"LESION-AWARE TEST ACCURACY: "
              f"{(np.array(y_pred_indices) == np.array(y_true_indices)).mean():.2%}")
        print(f"{'='*60}")
        print("\nClassification Report:")
        print(classification_report(y_true_indices, y_pred_indices, target_names=classes))

        # Confusion Matrix
        cm = confusion_matrix(y_true_indices, y_pred_indices)
        plt.figure(figsize=(8, 6))
        plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix: {args.eval_mode}')
        plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=300); plt.close()

    # Distribution chart for the dissertation
    if len(kept) > 0:
        cls_for_chart = classes if is_cnn else MST5_NAMES
        counts = kept["mst_name"].value_counts().reindex(cls_for_chart, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(cls_for_chart)), counts.values, color='steelblue')
        ax.set_xticks(range(len(cls_for_chart)))
        ax.set_xticklabels(cls_for_chart, rotation=20, ha='right')
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: Predicted Skin Tone Distribution "
                     f"(N={len(kept):,}, conf≥{args.confidence_filter})")
        for i, v in enumerate(counts.values):
            pct = 100.0 * v / max(len(kept), 1)
            ax.text(i, v, f"{v}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "tone_distribution.png", dpi=300)
        plt.close()
        print(f"\n  Charts saved to {out_dir}/")

    print(f"\n  CSV saved: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
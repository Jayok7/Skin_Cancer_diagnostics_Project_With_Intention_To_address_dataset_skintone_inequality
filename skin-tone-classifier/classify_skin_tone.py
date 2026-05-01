#!/usr/bin/env python3
"""
Lesion-Aware Skin Tone Classification — A+B Cascade Pipeline
=============================================================

Approach A: Perilesional Ring Sampling (U-Net segmentation → dilate → ring)
Approach B: Multi-Patch Consensus (corner/edge patches)

Pipeline Modes:
  1. ITA Analytical (Default): Computes ITA from the L*a*b* color space.
  2. FairFace CNN: Inpaints the lesion and passes the healthy skin to
     a trained EfficientNet-B4 model to predict the MST directly.
     
Ground Truth Evaluation:
  Provide --eval-csv to compute accuracy, classification report, confusion matrix,
  and MST category count charts.

Usage (CNN Mode with Evaluation):
  python classify_skin_tone.py \
      --image-dir datasets/MSKCC-images/ \
      --metadata-csv datasets/MSKCC-images/metadata.csv \
      --output-dir outputs/skin_tone_cascade/ \
      --cnn-model outputs/FairFace-Model-3.2-mst5/best_finetuned_model.pth \
      --eval-csv datasets/mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv \
      --visualise
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
# ITA Calculation (same as compute_mst_labels.py)
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
# MST-5 Mapping
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
CLASSES_3WAY = ['Light', 'Medium', 'Dark']

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
        warnings.warn("milesial/Pytorch-UNet not found. Falling back to Approach B only.")
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
        print(f"  Downloading pretrained U-Net weights from milesial release...")
        try:
            state = torch.hub.load_state_dict_from_url(url, map_location=device, file_name="unet_carvana.pth")
            model_tmp = UNet(n_channels=3, n_classes=2, bilinear=False)
            model_tmp.load_state_dict(state)
            for name, param in model_tmp.named_parameters():
                if "outc" not in name:
                    model.state_dict()[name].copy_(param)
            print(f"  ✓ Pretrained Carvana weights loaded (encoder + decoder)")
        except Exception as e:
            warnings.warn(f"Could not load U-Net weights: {e}")

    model.to(device).eval()
    return model

def segment_lesion_simple(image_bgr, model=None):
    if model is not None:
        import torch
        h, w = image_bgr.shape[:2]
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

        mask_list = mask_tensor.flatten().tolist()
        mask_bytes = bytes(mask_list)
        mask_flat = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
        mask_small = mask_flat.reshape(mask_tensor.shape[0], mask_tensor.shape[1])
        mask_resized = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_resized > 0
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask > 0

def extract_perilesional_ring(lesion_mask, margin_px=30, min_pixels=500):
    mask_uint8 = np.zeros(lesion_mask.shape[:2], dtype=np.uint8)
    mask_uint8[lesion_mask > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    ring = (dilated > 0) & (lesion_mask == 0)
    return ring, int(ring.sum()) >= min_pixels


# ───────────────────────────────────────────────────────────
# CNN Helper Functions
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
        median_color = np.median(crop[mask_patch > 0], axis=0) # [B, G, R]
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
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0] # Shape (6,)
    return probs

def aggregate_probs_3way(probs_6way):
    light = probs_6way[4] + probs_6way[5]
    medium = probs_6way[2] + probs_6way[3]
    dark = probs_6way[0] + probs_6way[1]
    return np.array([light, medium, dark])

def get_patch_positions(h, w, patch_size=32, margin=10):
    ps = patch_size
    positions = [
        (margin, margin), (margin, w - ps - margin), 
        (h - ps - margin, margin), (h - ps - margin, w - ps - margin),
        (margin, w // 2 - ps // 2), (h - ps - margin, w // 2 - ps // 2),
        (h // 2 - ps // 2, margin), (h // 2 - ps // 2, w - ps - margin),
    ]
    valid = []
    for y, x in positions:
        y = max(0, min(y, h - ps))
        x = max(0, min(x, w - ps))
        valid.append((y, x))
    return valid


# ───────────────────────────────────────────────────────────
# Prediction Logic
# ───────────────────────────────────────────────────────────

def process_ita(image_bgr, seg_model, use_segmentation, margin_px, min_ring_pixels, confidence_threshold):
    result = {}
    if use_segmentation:
        lesion_mask = segment_lesion_simple(image_bgr, seg_model)
        ring_mask, has_enough = extract_perilesional_ring(lesion_mask, margin_px, min_ring_pixels)
        result["lesion_mask"] = lesion_mask
        result["ring_mask"] = ring_mask
        
        if has_enough:
            ita, ita_std = compute_ita_from_bgr_masked(image_bgr, ring_mask)
            result.update({"ita": ita, "ita_std": ita_std, "method": "perilesional_ring", "confidence_val": ita_std})
            result["mst10_class"] = ita_to_mst10(ita)
            result["mst5_class"] = ita_to_mst5(ita)
            result["mst_name"] = MST5_NAMES[result["mst5_class"]] if 0 <= result["mst5_class"] < 5 else "unknown"
            result["confidence"] = "high" if ita_std < confidence_threshold else "low"
            return result

    # Fallback to B
    patch_size = 32
    h, w = image_bgr.shape[:2]
    positions = get_patch_positions(h, w, patch_size, margin=10)
    
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
    ita, ita_std = float(np.median(final_itas)), float(np.std(final_itas)) if len(final_itas) > 1 else 0.0
    
    result.update({
        "ita": ita, "ita_std": ita_std, "method": "multi_patch", 
        "patch_positions": positions, "surviving_mask": surviving,
        "mst10_class": ita_to_mst10(ita), "mst5_class": ita_to_mst5(ita),
        "confidence_val": ita_std, "confidence": "high" if ita_std < confidence_threshold else "low"
    })
    result["mst_name"] = MST5_NAMES[result["mst5_class"]] if 0 <= result["mst5_class"] < 5 else "unknown"
    return result


def process_cnn(image_bgr, seg_model, ff_model, transform, device, mode, use_segmentation, margin_px, min_ring_pixels):
    result = {}
    probs = None
    method_used = "A (Perilesional Ring)"
    
    if use_segmentation and seg_model is not None:
        lesion_mask = segment_lesion_simple(image_bgr, seg_model)
        ring_mask, has_enough = extract_perilesional_ring(lesion_mask, margin_px, min_ring_pixels)
        result["lesion_mask"] = lesion_mask
        result["ring_mask"] = ring_mask
        if has_enough:
            crop = get_cnn_friendly_crop(image_bgr, lesion_mask, ring_mask)
            if crop is not None:
                probs = predict_crop(crop, ff_model, transform, device)

    if probs is None: # Fallback Approach B
        method_used = "B (Multi-Patch Consensus)"
        h, w = image_bgr.shape[:2]
        patch_sz = int(min(h, w) * 0.25)
        # Using larger patches for CNN than ITA
        positions = get_patch_positions(h, w, patch_size=patch_sz, margin=10)
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
            
    if probs is None: # Approach C
        method_used = "C (Full Image Fallback)"
        probs = predict_crop(image_bgr, ff_model, transform, device)

    # Output Parsing
    if mode == '3-way':
        final_probs = aggregate_probs_3way(probs)
        classes = CLASSES_3WAY
    elif mode == '5-way':
        final_probs = probs
        classes = CLASSES_5WAY
    else:
        final_probs = probs
        classes = CLASSES_6WAY
        
    pred_idx = int(np.argmax(final_probs))
    confidence_val = np.max(final_probs)
    
    result.update({
        "probs": final_probs, "method": method_used, "confidence_val": confidence_val,
        "confidence": "high" if confidence_val >= 0.6 else "low",
        "pred_idx": pred_idx, "mst_name": classes[pred_idx]
    })
    return result


# ───────────────────────────────────────────────────────────
# Visualisation
# ───────────────────────────────────────────────────────────

def visualise_result(image_bgr, result, save_path, is_cnn=False):
    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()

    method = result.get("method", "")
    
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
        vis = overlay

    elif "Patch" in method or method == "multi_patch":
        positions = result.get("patch_positions", [])
        surviving = result.get("surviving_mask", [])
        patch_size = 32 if not is_cnn else int(min(h, w) * 0.25)
        mask_panel = image_bgr.copy()
        
        for i, (y, x) in enumerate(positions):
            if i < len(surviving) and surviving[i]:
                colour, thickness = (0, 255, 0), 2
            else:
                colour, thickness = (0, 0, 255), 1
            cv2.rectangle(vis, (x, y), (x + patch_size, y + patch_size), colour, thickness)
            cv2.rectangle(mask_panel, (x, y), (x + patch_size, y + patch_size), colour, thickness)
        mask_panel = vis.copy()
    else:
        mask_panel = np.zeros_like(image_bgr)

    # Text
    mst_name = result.get("mst_name", "?")
    conf = result.get("confidence", "unknown")
    conf_val = result.get("confidence_val", 0.0)
    
    if is_cnn:
        sub = f"Prob={conf_val:.2f} | {method} | {conf}"
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
    p = argparse.ArgumentParser(description="Lesion-aware skin tone classification (A+B cascade)")
    p.add_argument("--image-dir", type=str, required=True, help="Images directory")
    p.add_argument("--output-dir", type=str, default="outputs/skin_tone_cascade", help="Output directory")
    p.add_argument("--visualise", action="store_true", help="Save visualization images")
    p.add_argument("--no-segmentation", action="store_true", help="Skip Approach A (U-Net); use Approach B only")
    p.add_argument("--margin-px", type=int, default=30, help="Perilesional ring margin in pixels (default: 30)")
    p.add_argument("--min-ring-pixels", type=int, default=500, help="Minimum pixels in ring (default: 500)")
    p.add_argument("--confidence-threshold", type=float, default=15.0, help="ITA std threshold for confidence flag (default: 15.0)")
    p.add_argument("--extensions", type=str, default="jpg,jpeg,png,bmp,tif,tiff", help="Image extensions (comma-separated)")
    
    p.add_argument("--metadata-csv", type=str, default=None, help="Metadata CSV to filter images via --image-type-filter")
    p.add_argument("--image-type-filter", type=str, default="clinical: close-up", help="Filter for image types.")
    
    p.add_argument("--unet-dir", type=str, default=None, help="Path to milesial/Pytorch-UNet repo")
    p.add_argument("--unet-weights", type=str, default=None, help="Path to milesial/Pytorch-UNet weights .pth")
    
    # CNN Extension Arguments
    p.add_argument("--cnn-model", type=str, default=None, help="Path to trained FairFace model to use CNN instead of analytical ITA.")
    p.add_argument("--eval-csv", type=str, default=None, help="Metadata CSV with Ground Truth for evaluation (accuracy, bar charts).")
    p.add_argument("--eval-mode", type=str, choices=['3-way', '5-way', '6-way'], default='5-way', help="Evaluation mode for CNN: 3-way, 5-way, or 6-way")
    
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
    image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lstrip(".").lower() in extensions])

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    print(f"Found {len(image_files):,} images in {image_dir}")

    # Metadata filtering
    image_type_map = {}
    if args.metadata_csv:
        meta_path = Path(args.metadata_csv)
        if meta_path.is_file():
            meta_df = pd.read_csv(meta_path)
            if "isic_id" in meta_df.columns and "image_type" in meta_df.columns:
                id_to_type = dict(zip(meta_df["isic_id"], meta_df["image_type"]))
                for f in image_files:
                    image_type_map[f.name] = id_to_type.get(f.stem, "unknown")

                before = len(image_files)
                image_files = [f for f in image_files if id_to_type.get(f.stem, "") == args.image_type_filter]
                after = len(image_files)
                print(f"  Metadata filter: '{args.image_type_filter}' | Kept {after} (Filtered {before - after})")
                if not image_files: return

    # Ground Truth parsing for CNN evaluation
    gt_map = {}
    if args.eval_csv and args.cnn_model:
        eval_path = Path(args.eval_csv)
        if eval_path.is_file():
            edf = pd.read_csv(eval_path).dropna(subset=['fitzpatrick_skin_type', 'isic_id'])
            for _, row in edf.iterrows():
                gt = row['fitzpatrick_skin_type']
                if args.eval_mode == '3-way':
                    if gt in ['I', 'II']: gt = 'Light'
                    elif gt in ['III', 'IV']: gt = 'Medium'
                    elif gt in ['V', 'VI']: gt = 'Dark'
                    else: continue
                elif args.eval_mode == '5-way':
                    if gt == 'I': gt = 'Very Light (MST 1-2)'
                    elif gt == 'II': gt = 'Light (MST 3-4)'
                    elif gt in ['III', 'IV']: gt = 'Medium (MST 5-6)'
                    elif gt == 'V': gt = 'Dark (MST 7-8)'
                    elif gt == 'VI': gt = 'Very Dark (MST 9-10)'
                    else: continue
                gt_map[row['isic_id']] = gt

    # Load UNet
    seg_model = None
    use_seg = not args.no_segmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if use_seg:
        print("Loading U-Net model...")
        seg_model = load_segmentation_model(device=str(device), weights_path=args.unet_weights, unet_dir=args.unet_dir)
        if seg_model is None:
            use_seg = False

    # Load CNN
    is_cnn = args.cnn_model is not None
    ff_model, transform = None, None
    classes = CLASSES_6WAY if args.eval_mode == '6-way' else (CLASSES_5WAY if args.eval_mode == '5-way' else CLASSES_3WAY)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    if is_cnn:
        print(f"Loading FairFace EfficientNet-B4 from {args.cnn_model}...")
        ff_model = models.efficientnet_b4(weights=None)
        in_features = ff_model.classifier[1].in_features
        num_outs = 5 if args.eval_mode == '5-way' else 6
        ff_model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_outs))
        ff_model.load_state_dict(torch.load(args.cnn_model, map_location=device, weights_only=True))
        ff_model = ff_model.to(device).eval()
        
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Process Setup
    results = []
    y_true_indices, y_pred_indices = [], []
    
    print("\nRunning Inference...")
    for img_path in tqdm(image_files, desc="Classifying"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None: continue

        if is_cnn:
            result = process_cnn(
                image_bgr, seg_model, ff_model, transform, device, args.eval_mode,
                use_seg, args.margin_px, args.min_ring_pixels
            )
            # Add evaluation data
            gt_label = gt_map.get(img_path.stem)
            if gt_label:
                result["true_label"] = gt_label
                result["true_idx"] = class_to_idx.get(gt_label, -1)
                if result["true_idx"] != -1:
                    y_true_indices.append(result["true_idx"])
                    y_pred_indices.append(result["pred_idx"])
        else:
            result = process_ita(
                image_bgr, seg_model, use_seg, args.margin_px, args.min_ring_pixels, args.confidence_threshold
            )

        if args.visualise:
            vis_path = vis_dir / f"{img_path.stem}_vis.jpg"
            visualise_result(image_bgr, result, vis_path, is_cnn=is_cnn)

        row_dict = {
            "file": img_path.name,
            "image_type": image_type_map.get(img_path.name, ""),
            "mst_name": result["mst_name"],
            "method": result["method"],
            "confidence": result["confidence"],
            "confidence_val": round(result["confidence_val"], 3),
        }
        
        if is_cnn:
            row_dict["pred_idx"] = result.get("pred_idx")
            row_dict["true_label"] = result.get("true_label", "")
        else:
            row_dict["ita"] = round(result["ita"], 2) if not np.isnan(result.get("ita", np.nan)) else ""
            row_dict["ita_std"] = round(result.get("ita_std", 0.0), 2)
            row_dict["mst10_class"] = result.get("mst10_class", -1)
            row_dict["mst5_class"] = result.get("mst5_class", -1)
            
        results.append(row_dict)

    df = pd.DataFrame(results)
    csv_path = out_dir / "skin_tone_predictions.csv"
    df.to_csv(csv_path, index=False)

    # Console Summary
    print(f"\n{'='*60}")
    print(f"SKIN TONE CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Mode: {'CNN' if is_cnn else 'ITA Analytical'}")
    print(f"  Images processed: {len(df):,}")

    if len(df) > 0:
        print(f"\n  Method Used:")
        for m, c in df["method"].value_counts().items(): print(f"    {m}: {c:,} ({c/len(df):.1%})")

        print(f"\n  Confidence:")
        for c_val, c in df["confidence"].value_counts().items(): print(f"    {c_val}: {c:,} ({c/len(df):.1%})")

        print(f"\n  MST Distribution:")
        for mst_nm, c in df["mst_name"].value_counts().items():
            pct = c / len(df) * 100
            print(f"    {mst_nm:25s}: {c:4,} ({pct:5.1f}%)  {'█' * int(pct/2)}")

    # Ground Truth Evaluation Mode (only if CNN + GT provided)
    if is_cnn and len(y_true_indices) > 0:
        print(f"\n{'='*60}")
        print(f"LESION-AWARE TEST SET ACCURACY: {(np.array(y_pred_indices) == np.array(y_true_indices)).mean():.2%}")
        print(f"{'='*60}")
        print("\nClassification Report:")
        print(classification_report(y_true_indices, y_pred_indices, target_names=classes))

        # Confusion Matrix
        cm = confusion_matrix(y_true_indices, y_pred_indices)
        plt.figure(figsize=(8, 6))
        plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Lesion-Aware Confusion Matrix - {args.eval_mode}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        plt.savefig(str(cm_path), dpi=300)
        plt.close()

        # Bar chart
        pred_counts = df['mst_name'].value_counts().reindex(classes, fill_value=0)
        true_counts = df['true_label'].value_counts().reindex(classes, fill_value=0)
        
        x = np.arange(len(classes))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, true_counts.values, width, label='True Count', color='midnightblue')
        ax.bar(x + width/2, pred_counts.values, width, label='Predicted Count', color='cornflowerblue')
        
        ax.set_ylabel('Count')
        ax.set_title('MST Category Counts (True vs Predicted) - Lesion Aware')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        plt.tight_layout()
        bar_path = out_dir / "mst_category_counts.png"
        plt.savefig(str(bar_path), dpi=300)
        plt.close()
        
        print(f"\n  Charts saved to {out_dir}/")

    print(f"\n  CSV saved: {csv_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

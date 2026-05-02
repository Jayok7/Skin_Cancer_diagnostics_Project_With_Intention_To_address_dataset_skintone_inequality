#!/usr/bin/env python3
"""
ISIC 2020 Skin Tone Classification
====================================
Uses the fine-tuned FairFace v3.2 (3-class) model to assign skin tone
predictions to the entire ISIC 2020 training dataset (~33k images).

Outputs a CSV with:
  - isic_id, patient_id (from metadata)
  - predicted_class (0=Dark, 1=Medium, 2=Light)
  - predicted_label (human-readable)
  - confidence (max softmax probability)
  - prob_dark, prob_medium, prob_light (per-class softmax)

Usage:
    # On CSF (GPU):
    python classify_isic2020.py \
        --checkpoint outputs/FairFace-Model-3.2-finetuned-v5-3class/fairface_mskcc_best.pth \
        --metadata datasets/challenge-2020-training_metadata_2026-04-04.csv \
        --image-root datasets/ISIC-2020-images/ \
        --output datasets/isic2020_skin_tone_predictions.csv

    # Locally (CPU, slower):
    python classify_isic2020.py \
        --checkpoint outputs/.../fairface_mskcc_best.pth \
        --metadata datasets/challenge-2020-training_metadata_2026-04-04.csv \
        --image-root datasets/ISIC-2020-images/ \
        --device cpu
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ═══════════════════════════════════════════════════════════════
# CLASS NAMES (must match training)
# ═══════════════════════════════════════════════════════════════

MST3_NAMES = [
    "Dark (MST 7-10)",     # class 0
    "Medium (MST 3-6)",    # class 1
    "Light (MST 1-2)",     # class 2
]


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════

def build_model(num_classes: int = 3):
    """Build EfficientNet-B4 with custom head (must match training architecture)."""
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features  # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load fine-tuned model from checkpoint."""
    model = build_model(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════

class ISIC2020InferenceDataset(Dataset):
    """
    Simple dataset for batch inference on ISIC 2020 images.
    Returns (image_tensor, isic_id) pairs.
    Skips missing files gracefully.
    """

    def __init__(self, metadata_csv: str, image_root: str,
                 image_size: int = 380, image_ext: str = ".jpg"):
        self.image_root = image_root
        self.image_ext = image_ext

        # Load metadata
        df = pd.read_csv(metadata_csv)
        print(f"  Metadata loaded: {len(df):,} rows")

        # Build file paths and filter to existing images
        self.records = []
        missing = 0
        for _, row in df.iterrows():
            isic_id = row["isic_id"]
            patient_id = row.get("patient_id", "")
            img_path = os.path.join(image_root, f"{isic_id}{image_ext}")
            if os.path.isfile(img_path):
                self.records.append({
                    "isic_id": isic_id,
                    "patient_id": patient_id,
                    "img_path": img_path,
                })
            else:
                missing += 1

        print(f"  Found: {len(self.records):,} images")
        if missing > 0:
            print(f"  Missing: {missing:,} images (will be skipped)")

        # Inference transform (same as training validation)
        self.transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["img_path"]).convert("RGB")
            tensor = self.transform(img)
        except Exception as e:
            print(f"  ⚠ Failed to load {rec['isic_id']}: {e}")
            # Return a black image — will be flagged by low confidence
            tensor = torch.zeros(3, 380, 380)
        return tensor, rec["isic_id"], rec["patient_id"]


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_batch_inference(model, dataloader, device, class_names):
    """
    Run inference on all images in the dataloader.
    Returns a list of result dicts.
    """
    results = []
    total = len(dataloader.dataset)
    processed = 0
    t0 = time.time()

    for batch_idx, (images, isic_ids, patient_ids) in enumerate(dataloader):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        for i in range(len(isic_ids)):
            results.append({
                "isic_id": isic_ids[i],
                "patient_id": patient_ids[i],
                "predicted_class": int(preds[i]),
                "predicted_label": class_names[preds[i]],
                "confidence": float(confs[i]),
                "prob_dark": float(probs[i, 0]),
                "prob_medium": float(probs[i, 1]),
                "prob_light": float(probs[i, 2]),
            })

        processed += len(isic_ids)
        if (batch_idx + 1) % 50 == 0 or processed == total:
            elapsed = time.time() - t0
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else 0
            print(f"  [{processed:>6,}/{total:,}] "
                  f"{rate:.0f} img/s  ETA: {eta:.0f}s")

    return results


# ═══════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════

def print_summary(results_df, class_names):
    """Print distribution and confidence statistics."""
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images classified: {len(results_df):,}")

    print(f"\n  {'Label':<25s}  {'Count':>6s}  {'%':>6s}  {'Mean Conf':>9s}  {'Median':>7s}")
    print(f"  {'-'*60}")
    for cls in range(len(class_names)):
        subset = results_df[results_df["predicted_class"] == cls]
        n = len(subset)
        pct = n / len(results_df) * 100
        mean_conf = subset["confidence"].mean() if n > 0 else 0
        med_conf = subset["confidence"].median() if n > 0 else 0
        print(f"  {class_names[cls]:<25s}  {n:6,}  {pct:5.1f}%  {mean_conf:8.1%}  {med_conf:6.1%}")

    # Confidence distribution
    print(f"\n  Confidence Quartiles:")
    for q in [0.25, 0.5, 0.75, 0.9]:
        val = results_df["confidence"].quantile(q)
        print(f"    {q:.0%}: {val:.3f}")

    # Low-confidence flagging
    low_conf = results_df[results_df["confidence"] < 0.5]
    print(f"\n  Low-confidence predictions (<50%): {len(low_conf):,} "
          f"({len(low_conf)/len(results_df)*100:.1f}%)")

    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Classify ISIC 2020 images using fine-tuned FairFace 3-class model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model .pth (3-class)")
    parser.add_argument("--metadata", type=str,
                        default="datasets/challenge-2020-training_metadata_2026-04-04.csv",
                        help="Path to ISIC 2020 metadata CSV")
    parser.add_argument("--image-root", type=str,
                        default="datasets/ISIC-2020-images/",
                        help="Root directory containing ISIC 2020 .jpg images")
    parser.add_argument("--output", type=str,
                        default="datasets/isic2020_skin_tone_predictions.csv",
                        help="Output CSV path for predictions")
    parser.add_argument("--image-ext", type=str, default=".jpg",
                        help="Image file extension (default: .jpg)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of output classes (default: 3)")
    parser.add_argument("--image-size", type=int, default=380,
                        help="Image size (must match training, default: 380)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', or 'cpu'")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    class_names = MST3_NAMES[:args.num_classes]

    print(f"\n{'='*60}")
    print(f"ISIC 2020 Skin Tone Classification")
    print(f"{'='*60}")
    print(f"  Model:      {args.checkpoint}")
    print(f"  Metadata:   {args.metadata}")
    print(f"  Image root: {args.image_root}")
    print(f"  Output:     {args.output}")
    print(f"  Classes:    {args.num_classes} ({', '.join(class_names)})")
    print(f"  Device:     {device}")
    print(f"  Batch size: {args.batch_size}")

    # ── Load model ──
    print(f"\n--- Loading model ---")
    model = load_model(args.checkpoint, args.num_classes, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model loaded ({n_params:,} params)")

    # ── Load dataset ──
    print(f"\n--- Loading dataset ---")
    dataset = ISIC2020InferenceDataset(
        args.metadata, args.image_root,
        image_size=args.image_size, image_ext=args.image_ext,
    )

    if len(dataset) == 0:
        print("  ✗ No images found! Check --image-root and --image-ext.")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Run inference ──
    print(f"\n--- Running inference ({len(dataset):,} images) ---")
    t_start = time.time()
    results = run_batch_inference(model, dataloader, device, class_names)
    t_elapsed = time.time() - t_start
    print(f"\n  ✓ Inference complete: {t_elapsed:.1f}s "
          f"({len(results)/t_elapsed:.0f} img/s)")

    # ── Save results ──
    results_df = pd.DataFrame(results)

    # Merge with original metadata for extra context
    meta_df = pd.read_csv(args.metadata)
    # Keep useful columns from metadata
    meta_cols = ["isic_id"]
    for col in ["fitzpatrick_skin_type", "diagnosis_1", "anatom_site_general",
                 "age_approx", "sex", "image_type"]:
        if col in meta_df.columns:
            meta_cols.append(col)
    meta_subset = meta_df[meta_cols].drop_duplicates(subset=["isic_id"])

    results_df = results_df.merge(meta_subset, on="isic_id", how="left")

    # Sort by isic_id for consistency
    results_df = results_df.sort_values("isic_id").reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n  ✓ Results saved to: {args.output}")

    # ── Summary ──
    print_summary(results_df, class_names)

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()

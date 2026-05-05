#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_milk10k_predictions.py
================================
Stage 1 (GPU): Run each EfficientNet-B3 model on MILK10K and dump per-model
prediction CSVs plus a single manifest CSV.

NOTE: MILK10K skin_tone_class is INVERTED relative to Fitzpatrick17k:
   0 = darkest, 5 = lightest
The manifest's `tone_tertile` column normalises this so downstream evaluation
code works identically to the Fitzpatrick pipeline.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"))
from milk10k_to_isic_mapping import (
    DIAGNOSIS3_MAP, SIMPLIFIED_MAP, MILK_TONE_TERTILES, apply_mapping,
)

# ---------------------------------------------------------------------------
# Config (must match training)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 300
BATCH_SIZE = 32
TRAIN_CLASSES = sorted(['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC', 'UNK'])
NUM_CLASSES = len(TRAIN_CLASSES)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_id_column(df):
    """Return the most likely image identifier column."""
    for c in ["isic_id", "image_id", "image", "lesion_id", "id"]:
        if c in df.columns:
            return c
    sys.exit(f"Cannot find image-id column in: {list(df.columns)}")


def find_image(image_dir, image_id):
    candidates = [
        os.path.join(image_dir, f"{image_id}.jpg"),
        os.path.join(image_dir, f"{image_id}.JPG"),
        os.path.join(image_dir, f"{image_id}.png"),
        os.path.join(image_dir, f"{image_id}.jpeg"),
        os.path.join(image_dir, str(image_id)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def detect_source_column(df):
    """MILK10K may carry attribution/dataset metadata. Return col name or None."""
    for c in ["attribution", "dataset", "source", "study"]:
        if c in df.columns:
            return c
    return None


def build_manifest(csv_path, image_dir, mapping_name):
    print(f"\n--- Building MILK10K manifest ---")
    df = pd.read_csv(csv_path)
    print(f"  Rows in CSV: {len(df)}")

    id_col = detect_id_column(df)
    print(f"  Using id column: '{id_col}'")

    df = apply_mapping(df, mapping_name)

    # Skin tone -> tertile
    if "skin_tone_class" not in df.columns:
        sys.exit("ERROR: no 'skin_tone_class' column. Cannot stratify.")
    df["skin_tone_class"] = pd.to_numeric(df["skin_tone_class"], errors="coerce")
    df = df[df["skin_tone_class"].between(0, 5)].copy()
    df["skin_tone_class"] = df["skin_tone_class"].astype(int)
    df["tone_tertile"] = df["skin_tone_class"].map(MILK_TONE_TERTILES)
    print(f"  After tone filter (0-5): {len(df)}")

    # Resolve image paths
    df["image_id"] = df[id_col].astype(str)
    df["path"] = df["image_id"].map(lambda x: find_image(image_dir, x))
    n_missing = df["path"].isna().sum()
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    print(f"  Images on disk: {len(df)}  (missing: {n_missing})")
    if len(df) == 0:
        sys.exit("ERROR: No images found. Check --image-dir.")

    df["true_idx"] = df["isic_label"].map(lambda c: TRAIN_CLASSES.index(c))
    df["true_class"] = df["isic_label"]

    # Source/site (optional)
    src_col = detect_source_column(df)
    if src_col:
        df["source"] = df[src_col].fillna("unknown").astype(str)
        print(f"  Using source column: '{src_col}'")
    else:
        df["source"] = "milk10k"

    df["mapping"] = mapping_name.upper()

    keep = ["image_id", "path", "true_idx", "true_class",
            "skin_tone_class", "tone_tertile", "source", "mapping"]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(path, device):
    m = models.efficientnet_b3(weights=None)
    inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(inf), nn.Linear(inf, 256),
        nn.ReLU(True), nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
    )
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.to(device).eval()
    return m


class MilkDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(Image.open(row["path"]).convert("RGB"))
        return img, idx


@torch.no_grad()
def run_inference(model, loader, device):
    all_probs, all_idx = [], []
    for imgs, idxs in loader:
        logits = model(imgs.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_idx.extend(idxs.numpy().tolist())
    P = np.concatenate(all_probs, axis=0)
    return P[np.argsort(all_idx)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mapping", default="DIAGNOSIS3",
                   choices=["DIAGNOSIS3", "SIMPLIFIED"])
    p.add_argument("--model", action="append", default=[],
                   help='Repeatable. "Tag:path/to/weights.pth"')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    specs = []
    for s in args.model:
        if ":" not in s:
            print(f"  WARN skipping malformed --model '{s}'")
            continue
        tag, path = s.split(":", 1)
        tag, path = tag.strip(), path.strip()
        if not os.path.isfile(path):
            print(f"  WARN missing checkpoint: {tag} -> {path}")
            continue
        specs.append((tag, path))
    if not specs:
        sys.exit("ERROR: No valid --model specs")

    manifest = build_manifest(args.csv, args.image_dir, args.mapping)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"  Manifest -> {manifest_path}  (n={len(manifest)})")

    print("\n  Class x tertile:")
    pivot = manifest.pivot_table(
        index="true_class", columns="tone_tertile",
        values="image_id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in ["Light", "Medium", "Dark"]
                                   if c in pivot.columns], fill_value=0)
    print(pivot.to_string())
    print(f"\n  Tertile totals: {manifest['tone_tertile'].value_counts().to_dict()}")
    print(f"  Source distribution: {manifest['source'].value_counts().head(10).to_dict()}")

    ds = MilkDataset(manifest, test_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    prob_cols = [f"prob_{c}" for c in TRAIN_CLASSES]
    for tag, path in specs:
        print(f"\n--- Inference: {tag} ---")
        print(f"  Weights: {path}")
        model = load_model(path, device)
        P = run_inference(model, loader, device)

        out = pd.DataFrame({
            "image_id": manifest["image_id"].values,
            "pred_idx": P.argmax(axis=1),
        })
        for i, c in enumerate(prob_cols):
            out[c] = P[:, i]
        out_path = os.path.join(args.output_dir, f"{tag}_predictions.csv")
        out.to_csv(out_path, index=False)
        print(f"  Saved {len(out)} predictions -> {out_path}")

        merged = manifest.merge(out, on="image_id")
        acc = float((merged["pred_idx"] == merged["true_idx"]).mean())
        print(f"  Quick overall accuracy: {acc:.4f}")
        for t in ["Light", "Medium", "Dark"]:
            sub = merged[merged["tone_tertile"] == t]
            if len(sub):
                a = float((sub["pred_idx"] == sub["true_idx"]).mean())
                print(f"    {t:6s} (n={len(sub):4d}): acc={a:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[OK] All predictions written to {args.output_dir}/")


if __name__ == "__main__":
    main()
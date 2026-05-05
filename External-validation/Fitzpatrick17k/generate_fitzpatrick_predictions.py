#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_fitzpatrick_predictions.py
====================================
Stage 1 (GPU): Run each of N EfficientNet-B3 models on Fitzpatrick17k and
dump per-model prediction CSVs plus a single manifest CSV.

Outputs in --output-dir:
    manifest.csv               # image_id, path, true_idx, true_class,
                               # fitzpatrick_scale, fitzpatrick_centaur,
                               # fst_tertile, source, qc_flag, mapping
    <tag>_predictions.csv      # image_id, pred_idx, prob_<class>...

Stage 2 (CPU) is evaluate_fitzpatrick_stratified.py, which consumes these.
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
from fitzpatrick_to_isic_mapping import (
    STRICT_MAP, HIERARCHY_MAP, apply_mapping,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
IMAGE_SIZE = 300
BATCH_SIZE = 32

# Match training: 9-class output (alphabetical)
TRAIN_CLASSES = sorted(['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC', 'UNK'])
NUM_CLASSES = len(TRAIN_CLASSES)

FST_TERTILES = {1: "Light", 2: "Light",
                3: "Medium", 4: "Medium",
                5: "Dark", 6: "Dark"}

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_image(image_dir, md5hash, url_alphanum=None):
    candidates = [
        os.path.join(image_dir, f"{md5hash}.jpg"),
        os.path.join(image_dir, f"{md5hash}.png"),
        os.path.join(image_dir, f"{md5hash}.jpeg"),
        os.path.join(image_dir, str(md5hash)),
    ]
    if isinstance(url_alphanum, str):
        candidates += [
            os.path.join(image_dir, url_alphanum),
            os.path.join(image_dir, f"{url_alphanum}.jpg"),
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def infer_source(url):
    """Extract image source host (dermaamin / atlas-dermatologico / other)."""
    if not isinstance(url, str):
        return "unknown"
    u = url.lower()
    if "dermaamin" in u:
        return "dermaamin"
    if "atlasdermatologico" in u or "atlas-dermatologico" in u:
        return "atlas_dermatologico"
    return "other"


def build_manifest(csv_path, image_dir, mapping_name):
    print(f"\n--- Building manifest ---")
    df = pd.read_csv(csv_path)
    print(f"  Rows in CSV: {len(df)}")

    mapping = HIERARCHY_MAP if mapping_name.upper() == "HIERARCHY" else STRICT_MAP
    df = apply_mapping(df, mapping)

    df["fitzpatrick_scale"] = pd.to_numeric(df["fitzpatrick_scale"], errors="coerce")
    df = df[df["fitzpatrick_scale"].between(1, 6)].copy()
    df["fitzpatrick_scale"] = df["fitzpatrick_scale"].astype(int)

    if "fitzpatrick_centaur" in df.columns:
        df["fitzpatrick_centaur"] = pd.to_numeric(
            df["fitzpatrick_centaur"], errors="coerce")
    else:
        df["fitzpatrick_centaur"] = np.nan

    has_alphanum = "url_alphanum" in df.columns
    df["path"] = df.apply(
        lambda r: find_image(
            image_dir, r["md5hash"],
            r["url_alphanum"] if has_alphanum else None,
        ),
        axis=1,
    )
    n_missing = df["path"].isna().sum()
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    print(f"  Images on disk: {len(df)}  (missing: {n_missing})")

    if len(df) == 0:
        sys.exit("ERROR: No images found. Check --image-dir.")

    df["image_id"] = df["md5hash"]
    df["true_idx"] = df["isic_label"].map(lambda c: TRAIN_CLASSES.index(c))
    df["true_class"] = df["isic_label"]
    df["fst_tertile"] = df["fitzpatrick_scale"].map(FST_TERTILES)
    df["source"] = df["url"].map(infer_source) if "url" in df.columns else "unknown"
    df["qc_flag"] = df["qc"] if "qc" in df.columns else np.nan
    df["mapping"] = mapping_name.upper()

    keep_cols = ["image_id", "path", "true_idx", "true_class",
                 "fitzpatrick_scale", "fitzpatrick_centaur",
                 "fst_tertile", "source", "qc_flag", "mapping"]
    return df[keep_cols].reset_index(drop=True)


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


class FitzInferenceDataset(Dataset):
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
    all_probs = []
    all_idx = []
    for imgs, idxs in loader:
        logits = model(imgs.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_idx.extend(idxs.numpy().tolist())
    P = np.concatenate(all_probs, axis=0)
    # Restore original order
    order = np.argsort(all_idx)
    return P[order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Run all models on Fitzpatrick17k and dump prediction CSVs")
    p.add_argument("--csv", default="datasets/fitzpatrick17k.csv")
    p.add_argument("--image-dir", default="datasets/fitzpatrick17k/images")
    p.add_argument("--mapping", default="HIERARCHY",
                   choices=["STRICT", "HIERARCHY"])
    p.add_argument("--model", action="append", default=[],
                   help='Repeatable. Format: "Tag:path/to/weights.pth"')
    p.add_argument("--output-dir", default="outputs/fitzpatrick_predictions")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Parse models
    specs = []
    for s in args.model:
        if ":" not in s:
            print(f"  WARN: skipping malformed --model '{s}'")
            continue
        tag, path = s.split(":", 1)
        tag, path = tag.strip(), path.strip()
        if not os.path.isfile(path):
            print(f"  WARN: model file missing: {tag} -> {path}")
            continue
        specs.append((tag, path))
    if not specs:
        sys.exit("ERROR: No valid --model specs.")

    # Manifest (also informs which images we run inference on)
    manifest = build_manifest(args.csv, args.image_dir, args.mapping)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"  Manifest -> {manifest_path}  (n={len(manifest)})")

    print("\n  Class x FST tertile in manifest:")
    pivot = manifest.pivot_table(
        index="true_class", columns="fst_tertile",
        values="image_id", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in ["Light", "Medium", "Dark"]
                                   if c in pivot.columns], fill_value=0)
    print(pivot.to_string())
    print(f"\n  Tertile totals: {manifest['fst_tertile'].value_counts().to_dict()}")

    # Build loader once
    ds = FitzInferenceDataset(manifest, test_transform)
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

        # Quick console accuracy
        merged = manifest.merge(out, on="image_id")
        acc_overall = float((merged["pred_idx"] == merged["true_idx"]).mean())
        print(f"  Quick overall accuracy: {acc_overall:.4f}")
        for t in ["Light", "Medium", "Dark"]:
            sub = merged[merged["fst_tertile"] == t]
            if len(sub):
                a = float((sub["pred_idx"] == sub["true_idx"]).mean())
                print(f"    {t:6s} (n={len(sub):4d}): acc={a:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[OK] All predictions written to {args.output_dir}/")


if __name__ == "__main__":
    main()
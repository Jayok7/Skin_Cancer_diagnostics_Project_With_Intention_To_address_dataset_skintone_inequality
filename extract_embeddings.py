#!/usr/bin/env python3
"""
Extract Embeddings — CSF Script (torch only, no matplotlib)
===========================================================
Extracts penultimate-layer embeddings from a trained model
and saves them as a .npz file. Plot locally with visualise_latent_space.py.

Usage:
    python extract_embeddings.py \
        --model outputs/FairFace-Model-2.4/best_finetuned_model.pth \
        --data-csv datasets/fairface_lstar_labels.csv \
        --image-root datasets/fairface-img-margin025-trainval/ \
        --output outputs/FairFace-Model-2.4/embeddings.npz
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torchvision import models

from fairface_dataset import FairFaceDataset


def build_model(num_classes=6):
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    features = {}
    def hook_fn(module, input, output):
        features["embedding"] = output.squeeze()

    handle = model.avgpool.register_forward_hook(hook_fn)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        with autocast(enabled=(device.type == "cuda")):
            _ = model(images)

        emb = features["embedding"].cpu().numpy()
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        all_embeddings.append(emb)
        all_labels.extend(labels.numpy())

        if (i + 1) % 50 == 0:
            print(f"  Processed {(i+1) * loader.batch_size} images...")

    handle.remove()
    return np.vstack(all_embeddings), np.array(all_labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-csv", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", default="embeddings.npz")
    p.add_argument("--num-classes", type=int, default=6, choices=[3, 6])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=380)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(num_classes=args.num_classes)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model = model.to(device)
    print(f"Loaded model from {args.model}")

    val_ds = FairFaceDataset(
        args.data_csv, args.image_root,
        split="val", image_size=args.image_size,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    print(f"Extracting embeddings from {len(val_ds)} images...")
    embeddings, labels = extract_embeddings(model, val_loader, device)

    # Save class names alongside embeddings
    class_names = val_ds.class_names

    np.savez(args.output,
             embeddings=embeddings,
             labels=labels,
             class_names=class_names)
    print(f"Saved embeddings {embeddings.shape} to {args.output}")


if __name__ == "__main__":
    main()

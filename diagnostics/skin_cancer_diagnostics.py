#!/usr/bin/env python3
"""
Combined Skin Cancer Diagnostics — EfficientNet Fairness + Grad-CAM
====================================================================

All output (plots, metrics, Grad-CAM images) is saved to the output
directory specified by --output-dir.

Designed to be run as a Slurm job on CSF.

Usage:
    python skin_cancer_diagnostics.py \
        --base-path datasets/Ham10000 \
        --csv-path  datasets/Ham10000/HAM10000_metadata.csv \
        --output-dir outputs/skin_cancer_diagnostics \
        --device cuda
"""

import os
import sys
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# =====================================================================
# SECTION 0: CONFIGURATION
# =====================================================================
IMAGE_SIZE      = 300       # EfficientNet-B3 native resolution
BATCH_SIZE      = 32
HEAD_EPOCHS     = 20
FINETUNE_EPOCHS = 20        # Total: 40 epochs (matching notebook)
LR_HEAD         = 1e-3
LR_FINETUNE     = 1e-5
LABEL_SMOOTHING = 0.1
NUM_GCAM        = 5         # number of Grad-CAM samples per class


# =====================================================================
# SECTION 1: DATA LOADING
# =====================================================================
def load_data(base_path: str, csv_path: str = None):
    if csv_path is None:
        csv_path = os.path.join(base_path, "HAM10000_metadata.csv")
    dir1 = os.path.join(base_path, "HAM10000_images_part_1")
    dir2 = os.path.join(base_path, "HAM10000_images_part_2")

    print(f"  CSV:    {csv_path}")
    print(f"  Dir 1:  {dir1}")
    print(f"  Dir 2:  {dir2}")

    df = pd.read_csv(csv_path)
    def resolve_path(img_id):
        p1 = os.path.join(dir1, img_id + ".jpg")
        p2 = os.path.join(dir2, img_id + ".jpg")
        if os.path.exists(p1):
            return p1
        elif os.path.exists(p2):
            return p2
        return None

    df["path"] = df["image_id"].apply(resolve_path)
    before = len(df)
    df = df.dropna(subset=["path"])
    after = len(df)
    if before != after:
        print(f"  WARNING: {before - after} images not found on disk, skipped")
    label_columns = sorted(df["dx"].unique())
    num_classes = len(label_columns)

    # One-hot encode
    df_onehot = pd.get_dummies(df["dx"], dtype="float32")
    df = pd.concat([df, df_onehot], axis=1)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["dx"]
    )
    print(f"  Train samples: {len(train_df)},  Validation samples: {len(val_df)}")
    return train_df, val_df, label_columns, num_classes


# =====================================================================
# SECTION 2: PYTORCH DATASET
# =====================================================================
class SkinDataset(Dataset):
    def __init__(self, df, label_columns, transform, augment=False):
        self.df = df.reset_index(drop=True)
        self.label_columns = label_columns
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(
            [row[c] for c in self.label_columns], dtype=torch.float32
        )
        return img, label


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# =====================================================================
# SECTION 3: MODEL
# =====================================================================
def build_efficientnet(num_classes, dropout=0.4, freeze_backbone=True):
    """Build EfficientNet-B3 with custom classification head.
    Matches the notebook architecture: B3 backbone + BN + Dense(256) + Dropout + Softmax.
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features  # 1536 for B3
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model


# =====================================================================
# SECTION 4: TRAINING LOOP
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=str(device)):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        targets = labels.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets, all_confs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            targets = labels.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_targets, all_confs


def train_model(model, train_loader, val_loader, device, output_dir,
                head_epochs=HEAD_EPOCHS, ft_epochs=FINETUNE_EPOCHS):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    scaler = torch.amp.GradScaler(device=str(device))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Best model tracking (validation accuracy) ──
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    save_path = os.path.join(output_dir, "best_efficientnet_b3_ham10000.pth")

    def _log_epoch(epoch_num, total, tloss, tacc, vloss, vacc):
        """Log epoch and track best model."""
        nonlocal best_val_acc, best_epoch, best_state
        history["train_loss"].append(tloss)
        history["train_acc"].append(tacc)
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)

        marker = ""
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_epoch = epoch_num
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            marker = " ★ best"

        print(f"  Epoch {epoch_num}/{total} — "
              f"train_loss={tloss:.4f}, train_acc={tacc:.4f}, "
              f"val_loss={vloss:.4f}, val_acc={vacc:.4f}{marker}")

    total_epochs = head_epochs + ft_epochs

    # Stage 1: Head only
    print(f"\n--- Stage 1: Training head (backbone frozen) [{head_epochs} epochs] ---")
    for epoch in range(head_epochs):
        tloss, tacc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vloss, vacc, _, _, _ = evaluate(model, val_loader, criterion, device)
        _log_epoch(epoch + 1, total_epochs, tloss, tacc, vloss, vacc)

    # Stage 2: Full fine-tune
    print(f"\n--- Stage 2: Fine-tuning entire network [{ft_epochs} epochs] ---")
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)

    for epoch in range(ft_epochs):
        tloss, tacc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vloss, vacc, _, _, _ = evaluate(model, val_loader, criterion, device)
        _log_epoch(head_epochs + epoch + 1, total_epochs, tloss, tacc, vloss, vacc)

    # ── Restore best weights ──
    print(f"\n  ✓ Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"  ✓ Restoring best weights from epoch {best_epoch}")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"  ✓ Best model saved → {save_path}")

    return history


# =====================================================================
# SECTION 5: EVALUATION PLOTS
# =====================================================================
def plot_training_curves(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_acc"], label="Train")
    ax1.plot(history["val_acc"], label="Validation")
    ax1.set_title("EfficientNet-B3 Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_loss"], label="Train")
    ax2.plot(history["val_loss"], label="Validation")
    ax2.set_title("EfficientNet-B3 Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  ✓ Training curves → {output_dir}/training_curves.png")


def plot_confusion_matrix(y_true, y_pred, label_columns, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_columns)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix → {output_dir}/confusion_matrix.png")


def print_classification_report(y_true, y_pred, label_columns, output_dir):
    report = classification_report(y_true, y_pred, target_names=label_columns)
    print("\nClassification Report (EfficientNet-B3):")
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print(f"  ✓ Report → {output_dir}/classification_report.txt")


def print_confidence_stats(y_true, y_pred, confs, label_columns, output_dir):
    """Print and save per-class confidence statistics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confs  = np.array(confs)

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("Confidence Statistics by Class")
    lines.append("=" * 60)
    lines.append(f"{'Class':>12s}  {'N':>5s}  {'Mean':>6s}  {'Std':>6s}  {'Min':>6s}  {'Max':>6s}")
    lines.append("-" * 60)

    for i, cls in enumerate(label_columns):
        mask = y_pred == i
        if mask.sum() > 0:
            c = confs[mask]
            lines.append(
                f"{cls:>12s}  {mask.sum():5d}  {c.mean():.4f}  "
                f"{c.std():.4f}  {c.min():.4f}  {c.max():.4f}"
            )

    # Correct vs incorrect
    correct_mask = y_true == y_pred
    if correct_mask.sum() > 0:
        lines.append("")
        cc = confs[correct_mask]
        ic = confs[~correct_mask]
        lines.append(f"  Correct predictions   (n={correct_mask.sum():5d}):  "
                     f"mean conf = {cc.mean():.4f}")
        if (~correct_mask).sum() > 0:
            lines.append(f"  Incorrect predictions (n={(~correct_mask).sum():5d}):  "
                         f"mean conf = {ic.mean():.4f}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(output_dir, "confidence_stats.txt"), "w") as f:
        f.write(report)
    print(f"  ✓ Confidence stats → {output_dir}/confidence_stats.txt")


def plot_confidence_distribution(y_true, y_pred, confs, label_columns, output_dir):
    """Plot confidence distribution for correct vs incorrect predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confs  = np.array(confs)
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall: correct vs incorrect
    axes[0].hist(confs[correct], bins=30, alpha=0.7, label="Correct", color="green")
    if (~correct).sum() > 0:
        axes[0].hist(confs[~correct], bins=30, alpha=0.7, label="Incorrect", color="red")
    axes[0].set_title("Confidence: Correct vs Incorrect")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-class
    for i, cls in enumerate(label_columns):
        mask = y_pred == i
        if mask.sum() > 0:
            axes[1].hist(confs[mask], bins=20, alpha=0.5, label=cls)
    axes[1].set_title("Confidence by Predicted Class")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
    plt.close()
    print(f"  ✓ Confidence plot → {output_dir}/confidence_distribution.png")


# =====================================================================
# SECTION 6: GRAD-CAM
# =====================================================================
class GradCAM:
    """Simple Grad-CAM implementation for EfficientNet."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def generate_gradcam_images(model, val_df, label_columns, device, output_dir):
    """Generate Grad-CAM overlays for a sample of validation images."""
    gcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gcam_dir, exist_ok=True)

    # Target layer: last conv block of EfficientNet features
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    transform = get_transforms(augment=False)
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )

    print("\n--- Generating Grad-CAM visualisations ---")

    for cls in label_columns:
        cls_df = val_df[val_df["dx"] == cls].sample(
            n=min(NUM_GCAM, len(val_df[val_df["dx"] == cls])),
            random_state=42,
        )
        for idx, (_, row) in enumerate(cls_df.iterrows()):
            img_pil = Image.open(row["path"]).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            cam, pred_idx = grad_cam(img_tensor)

            # Denormalize for display
            img_display = inv_normalize(img_tensor.squeeze().cpu())
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original
            axes[0].imshow(img_display)
            axes[0].set_title(f"Original (GT: {cls})")
            axes[0].axis("off")

            # Grad-CAM heatmap
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"Grad-CAM (Pred: {label_columns[pred_idx]})")
            axes[1].axis("off")

            # Overlay
            axes[2].imshow(img_display)
            axes[2].imshow(cam, cmap="jet", alpha=0.4)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            fname = f"gradcam_{cls}_{idx}.png"
            plt.savefig(os.path.join(gcam_dir, fname), dpi=150)
            plt.close()

        print(f"  ✓ {cls}: {len(cls_df)} Grad-CAM images saved")

    print(f"\nAll Grad-CAM images saved to {gcam_dir}/")


# =====================================================================
# SECTION 7: MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Skin cancer diagnostics — EfficientNet + Grad-CAM"
    )
    parser.add_argument("--base-path", default="datasets/Ham10000",
                        help="Path to HAM10000 image root (containing image dirs)")
    parser.add_argument("--csv-path", default=None,
                        help="Path to HAM10000_metadata.csv (if separate from base-path)")
    parser.add_argument("--output-dir", default="outputs/skin_cancer_diagnostics",
                        help="Output directory for all results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing model")
    parser.add_argument("--model-path", default=None,
                        help="Path to existing model (used with --skip-train)")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    # ── Data ──
    print("\n--- Loading data ---")
    train_df, val_df, label_columns, num_classes = load_data(args.base_path, args.csv_path)

    # ── Class distribution ──
    print("\nClass distribution:")
    print(train_df["dx"].value_counts())

    # ── Datasets ──
    train_dataset = SkinDataset(train_df, label_columns, get_transforms(augment=True))
    val_dataset   = SkinDataset(val_df, label_columns, get_transforms(augment=False))
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True)

    # ── Model ──
    model = build_efficientnet(num_classes).to(device)

    if args.skip_train and args.model_path:
        print(f"\nLoading existing model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        # ── Train ──
        history = train_model(model, train_loader, val_loader, device, args.output_dir)
        plot_training_curves(history, args.output_dir)

    # ── Evaluate ──
    print("\n--- Evaluating model ---")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    _, _, y_pred, y_true, confs = evaluate(model, val_loader, criterion, device)
    print_classification_report(y_true, y_pred, label_columns, args.output_dir)
    plot_confusion_matrix(y_true, y_pred, label_columns, args.output_dir)
    print_confidence_stats(y_true, y_pred, confs, label_columns, args.output_dir)
    plot_confidence_distribution(y_true, y_pred, confs, label_columns, args.output_dir)

    # ── Grad-CAM ──
    generate_gradcam_images(model, val_df, label_columns, device, args.output_dir)

    print("\n✅ All diagnostics complete!")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

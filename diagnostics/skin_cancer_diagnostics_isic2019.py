#!/usr/bin/env python3
"""
ISIC 2019 Diagnostics — EfficientNet-B3 (with augmented data support)
=====================================================================
Adapted from skin_cancer_diagnostics.py for the ISIC 2019 dataset.
Same architecture, same 40-epoch schedule, same checkpointing.

ISIC 2019 has 8 classes: AK, BCC, BKL, DF, MEL, NV, SCC, VASC
Ground truth CSV is one-hot encoded.
"""
import os, sys, gc, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support
)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

IMAGE_SIZE = 300; BATCH_SIZE = 32
HEAD_EPOCHS = 20; FINETUNE_EPOCHS = 80
LR_HEAD = 1e-3; LR_FINETUNE = 1e-5; LABEL_SMOOTHING = 0.1; NUM_GCAM = 5

def load_isic2019_data(image_dir, gt_csv, aug_dir=None, aug_manifest=None):
    gt_df = pd.read_csv(gt_csv)
    class_cols = sorted([c for c in gt_df.columns if c != 'image'])
    num_classes = len(class_cols)
    print(f"  Classes ({num_classes}): {class_cols}")
    records = []
    for _, row in gt_df.iterrows():
        img_name = row['image']
        img_path = None
        for ext in ['.jpg', '.png', '']:
            c = os.path.join(image_dir, f"{img_name}{ext}")
            if os.path.exists(c): img_path = c; break
        if not img_path: continue
        label_vec = [float(row[c]) for c in class_cols]
        dx = class_cols[np.argmax(label_vec)]
        records.append({'image_id': img_name, 'path': img_path, 'dx': dx,
                        **{c: float(row[c]) for c in class_cols}})
    df = pd.DataFrame(records)
    print(f"  Original images: {len(df)}")
    if aug_dir and aug_manifest and os.path.exists(aug_manifest):
        aug_mf = pd.read_csv(aug_manifest)
        src_labels = {}
        for _, r in gt_df.iterrows():
            cols = sorted([c for c in gt_df.columns if c != 'image'])
            src_labels[r['image']] = cols[np.argmax([r[c] for c in cols])]
        aug_recs = []
        for _, ar in aug_mf.iterrows():
            if ar['source_id'] not in src_labels: continue
            dx = src_labels[ar['source_id']]
            p = os.path.join(aug_dir, ar['target_class'], ar['output_name'])
            if not os.path.exists(p): continue
            lv = {c: 0.0 for c in class_cols}; lv[dx] = 1.0
            aug_recs.append({'image_id': f"aug_{ar['output_name']}", 'path': p, 'dx': dx, **lv})
        if aug_recs:
            df = pd.concat([df, pd.DataFrame(aug_recs)], ignore_index=True)
            print(f"  + {len(aug_recs)} augmented images")
    print(f"  Total: {len(df)}")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["dx"])
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df, class_cols, num_classes

class SkinDataset(Dataset):
    def __init__(self, df, label_columns, transform):
        self.df = df.reset_index(drop=True); self.label_columns = label_columns; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(Image.open(row["path"]).convert("RGB"))
        label = torch.tensor([row[c] for c in self.label_columns], dtype=torch.float32)
        return img, label

def get_transforms(augment=False):
    if augment:
        return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20), transforms.ColorJitter(0.1, 0.1),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def build_efficientnet(num_classes, dropout=0.4, freeze_backbone=True):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    if freeze_backbone:
        for p in model.features.parameters(): p.requires_grad = False
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.BatchNorm1d(inf), nn.Linear(inf, 256),
        nn.ReLU(True), nn.Dropout(dropout), nn.Linear(256, num_classes))
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train(); tl, cor, tot = 0, 0, 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device); optimizer.zero_grad()
        with torch.amp.autocast(device_type=str(device)):
            out = model(imgs); loss = criterion(out, labs)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        tl += loss.item()*imgs.size(0); cor += (out.argmax(1)==labs.argmax(1)).sum().item(); tot += imgs.size(0)
    return tl/tot, cor/tot

def evaluate(model, loader, criterion, device):
    model.eval(); tl, cor, tot = 0, 0, 0; ap, at, ac = [], [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs); loss = criterion(out, labs)
            tl += loss.item()*imgs.size(0); probs = torch.softmax(out, 1)
            confs, preds = probs.max(1); targets = labs.argmax(1)
            cor += (preds==targets).sum().item(); tot += imgs.size(0)
            ap.extend(preds.cpu().numpy()); at.extend(targets.cpu().numpy()); ac.extend(confs.cpu().numpy())
    return tl/tot, cor/tot, ap, at, ac

def train_model(model, tl, vl, device, odir):
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    scaler = torch.amp.GradScaler(device=str(device))
    hist = {"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    best_va, best_ep, best_st = 0.0, 0, None
    sp = os.path.join(odir, "best_efficientnet_b3_isic2019.pth")
    total = HEAD_EPOCHS + FINETUNE_EPOCHS
    print(f"\n--- Stage 1: Head only [{HEAD_EPOCHS} epochs] ---")
    for e in range(HEAD_EPOCHS):
        tlo, ta = train_one_epoch(model, tl, criterion, opt, device, scaler)
        vlo, va, _, _, _ = evaluate(model, vl, criterion, device)
        hist["train_loss"].append(tlo); hist["train_acc"].append(ta)
        hist["val_loss"].append(vlo); hist["val_acc"].append(va)
        m = ""
        if va > best_va:
            best_va, best_ep = va, e+1
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_st, sp); m = " ★"
        print(f"  Epoch {e+1}/{total} — tl={tlo:.4f} ta={ta:.4f} vl={vlo:.4f} va={va:.4f}{m}")
    print(f"\n--- Stage 2: Fine-tune [{FINETUNE_EPOCHS} epochs] ---")
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    
    # Early Stopping Variables
    patience = 15
    stagnant_epochs = 0
    best_ta_ft = 0.0 # Track best train accuracy during fine-tuning
    
    for e in range(FINETUNE_EPOCHS):
        tlo, ta = train_one_epoch(model, tl, criterion, opt, device, scaler)
        vlo, va, _, _, _ = evaluate(model, vl, criterion, device)
        
        hist["train_loss"].append(tlo); hist["train_acc"].append(ta)
        hist["val_loss"].append(vlo); hist["val_acc"].append(va)
        m = ""
        
        # 1. Track Overall Best Validation Accuracy (for saving the model)
        if va > best_va:
            best_va, best_ep = va, HEAD_EPOCHS+e+1
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_st, sp)
            m = " ★"
            
        print(f"  Epoch {HEAD_EPOCHS+e+1}/{total} — tl={tlo:.4f} ta={ta:.4f} vl={vlo:.4f} va={va:.4f}{m}")

        # 2. Early Stopping Logic (Trigger only if BOTH train and val stop improving)
        improved = False
        if va >= max(hist["val_acc"][-patience:]): # Did val acc hit a new recent high?
            improved = True
        if ta > best_ta_ft:
            best_ta_ft = ta
            improved = True
            
        if improved:
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            print(f"    -> No improvement in train/val acc. Patience: {stagnant_epochs}/{patience}")
            
        if stagnant_epochs >= patience:
            print(f"\n  [Early Stopping] Triggered at epoch {HEAD_EPOCHS+e+1}. Training halted.")
            break

    print(f"\n  ✓ Best val_acc={best_va:.4f} @ epoch {best_ep}")
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    return hist

def plot_curves(hist, odir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(hist["train_acc"], label="Train"); a1.plot(hist["val_acc"], label="Val")
    a1.set_title("Accuracy (ISIC 2019)"); a1.legend(); a1.grid(True)
    a2.plot(hist["train_loss"], label="Train"); a2.plot(hist["val_loss"], label="Val")
    a2.set_title("Loss (ISIC 2019)"); a2.legend(); a2.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(odir, "training_curves.png"), dpi=150); plt.close()

class GradCAM:
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
        if class_idx is None: class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

def generate_gradcam_images(model, val_df, label_columns, device, output_dir):
    gcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gcam_dir, exist_ok=True)
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    transform = get_transforms(augment=False)
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

    print("\n--- Generating Grad-CAM visualisations ---")
    for cls in label_columns:
        cls_df = val_df[val_df["dx"] == cls]
        if len(cls_df) == 0: continue
        cls_df = cls_df.sample(n=min(NUM_GCAM, len(cls_df)), random_state=42)
        
        for idx, (_, row) in enumerate(cls_df.iterrows()):
            img_pil = Image.open(row["path"]).convert("RGB")
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            cam, pred_idx = grad_cam(img_tensor)

            img_display = inv_normalize(img_tensor.squeeze().cpu()).permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_display); axes[0].set_title(f"Original (GT: {cls})"); axes[0].axis("off")
            axes[1].imshow(cam, cmap="jet"); axes[1].set_title(f"Grad-CAM (Pred: {label_columns[pred_idx]})"); axes[1].axis("off")
            axes[2].imshow(img_display); axes[2].imshow(cam, cmap="jet", alpha=0.4); axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(gcam_dir, f"gradcam_{cls}_{idx}.png"), dpi=150)
            plt.close()
    print(f"  ✓ All Grad-CAM images saved to {gcam_dir}/")

def print_confidence_stats(y_true, y_pred, confs, label_columns, output_dir):
    y_true = np.array(y_true); y_pred = np.array(y_pred); confs = np.array(confs)
    lines = ["", "=" * 60, "Confidence Statistics by Class", "=" * 60, f"{'Class':>12s}  {'N':>5s}  {'Mean':>6s}  {'Std':>6s}  {'Min':>6s}  {'Max':>6s}", "-" * 60]
    for i, cls in enumerate(label_columns):
        mask = y_pred == i
        if mask.sum() > 0:
            c = confs[mask]
            lines.append(f"{cls:>12s}  {mask.sum():5d}  {c.mean():.4f}  {c.std():.4f}  {c.min():.4f}  {c.max():.4f}")
    correct_mask = y_true == y_pred
    if correct_mask.sum() > 0:
        lines.append("")
        cc = confs[correct_mask]
        lines.append(f"  Correct predictions   (n={correct_mask.sum():5d}):  mean conf = {cc.mean():.4f}")
        if (~correct_mask).sum() > 0:
            ic = confs[~correct_mask]
            lines.append(f"  Incorrect predictions (n={(~correct_mask).sum():5d}):  mean conf = {ic.mean():.4f}")
    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(output_dir, "confidence_stats.txt"), "w") as f: f.write(report)

def plot_confidence_distribution(y_true, y_pred, confs, label_columns, output_dir):
    y_true = np.array(y_true); y_pred = np.array(y_pred); confs = np.array(confs)
    correct = y_true == y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(confs[correct], bins=30, alpha=0.7, label="Correct", color="green")
    if (~correct).sum() > 0: axes[0].hist(confs[~correct], bins=30, alpha=0.7, label="Incorrect", color="red")
    axes[0].set_title("Confidence: Correct vs Incorrect"); axes[0].set_xlabel("Confidence"); axes[0].set_ylabel("Count"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for i, cls in enumerate(label_columns):
        mask = y_pred == i
        if mask.sum() > 0: axes[1].hist(confs[mask], bins=20, alpha=0.5, label=cls)
    axes[1].set_title("Confidence by Predicted Class"); axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Count"); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150); plt.close()

def main():
    p = argparse.ArgumentParser(description="ISIC 2019 — EfficientNet-B3")
    p.add_argument("--image-dir", default="datasets/ISIC_2019_Training_Input")
    p.add_argument("--gt-csv", default="datasets/ISIC_2019_Training_GroundTruth.csv")
    p.add_argument("--aug-dir", default=None)
    p.add_argument("--aug-manifest", default=None)
    p.add_argument("--output-dir", default="outputs/isic2019_diagnostics")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--model-path", default=None)
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}")
    print("\n--- Loading ISIC 2019 ---")
    train_df, val_df, lc, nc = load_isic2019_data(args.image_dir, args.gt_csv, args.aug_dir, args.aug_manifest)
    print("\nTrain distribution:"); print(train_df["dx"].value_counts())
    tds = SkinDataset(train_df, lc, get_transforms(True))
    vds = SkinDataset(val_df, lc, get_transforms(False))
    tldr = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vldr = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = build_efficientnet(nc).to(device)
    if args.skip_train and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        hist = train_model(model, tldr, vldr, device, args.output_dir)
        plot_curves(hist, args.output_dir)
    print("\n--- Evaluating ---")
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    _, _, yp, yt, confs = evaluate(model, vldr, crit, device)
    
    # ── CALCULATE EXPLICIT METRICS ──
    acc = accuracy_score(yt, yp)
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(yt, yp, average='macro')
    _, _, wt_f1, _ = precision_recall_fscore_support(yt, yp, average='weighted')
    
    print("\n" + "="*60)
    print("  FINAL EVALUATION METRICS")
    print("="*60)
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  Macro Precision:  {mac_p:.4f}")
    print(f"  Macro Recall:     {mac_r:.4f}")
    print(f"  Macro F1:         {mac_f1:.4f}")
    print(f"  Weighted F1:      {wt_f1:.4f}")
    print("="*60 + "\n")

    report = classification_report(yt, yp, target_names=lc)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f: 
        f.write(report)
        
    cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=lc).plot(cmap="Blues", xticks_rotation=45, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    # Generate Grad-CAM for validation set
    generate_gradcam_images(model, val_df, lc, device, args.output_dir)
    print_confidence_stats(yt, yp, confs, lc, args.output_dir)
    plot_confidence_distribution(yt, yp, confs, lc, args.output_dir)
    print("\n✅ ISIC 2019 diagnostics complete!")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__": main()

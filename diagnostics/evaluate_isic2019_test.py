#!/usr/bin/env python3
"""
evaluate_isic2019_test.py
=========================
Flexible evaluation of N EfficientNet-B3 models on ISIC 2019 test set.

Pass any number of models via repeated --model flags:
    --model "Original:outputs/isic2019_orig/best_efficientnet_b3_isic2019.pth"
    --model "Aug_L00:outputs/isic2019_aug00/best_efficientnet_b3_isic2019.pth"
    --model "Aug_L03:outputs/isic2019_aug03/best_efficientnet_b3_isic2019.pth"

Generates per model:
  - Classification report (.txt)
  - Confusion matrix (.png)
  - ROC curves per class + macro (.png)
  - Confidence distribution (.png)
  - Grad-CAM samples (.png)

Generates cross-model:
  - Comparison bar chart (accuracy, balanced acc, macro F1, weighted F1)
  - Per-class F1 heatmap
  - Combined summary report (.txt)
"""

import os, sys, argparse, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score,
)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

# =====================================================================
# CONFIG
# =====================================================================
IMAGE_SIZE = 300
BATCH_SIZE = 32
NUM_GCAM = 5

TRAIN_CLASSES = sorted(['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC', 'UNK'])
NUM_CLASSES = len(TRAIN_CLASSES)

# Exclude UNK from evaluation metrics (index 8 in sorted list)
EVAL_CLASSES = [c for c in TRAIN_CLASSES if c != 'UNK']

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225],
)


# =====================================================================
# DATA
# =====================================================================
def load_test_data(test_images_dir, test_gt_csv):
    gt = pd.read_csv(test_gt_csv)
    disease_cols = [c for c in gt.columns
                    if c not in ['image', 'score_weight', 'validation_weight']]
    print(f"  Ground truth rows: {len(gt)}")
    print(f"  GT columns: {disease_cols}")

    records = []
    unk_count = 0

    for _, row in gt.iterrows():
        img_name = row['image']
        label_vec = {c: float(row[c]) for c in disease_cols}
        dx = max(label_vec, key=label_vec.get)

        if dx == 'UNK':
            unk_count += 1
            continue
        if dx not in TRAIN_CLASSES:
            continue

        img_path = None
        for subdir in ['', 'ISIC_2019_Test_Input']:
            for ext in ['.jpg', '.png', '']:
                candidate = os.path.join(test_images_dir, subdir, f"{img_name}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path:
                break
        if not img_path:
            continue

        records.append({
            'image_id': img_name, 'path': img_path,
            'dx': dx, 'label_idx': TRAIN_CLASSES.index(dx),
        })

    df = pd.DataFrame(records)
    print(f"  Loaded: {len(df)} test images  (UNK excluded: {unk_count})")
    for cls in EVAL_CLASSES:
        print(f"    {cls:6s}: {len(df[df['dx'] == cls]):5d}")
    return df


class TestDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(Image.open(row['path']).convert('RGB'))
        return img, row['label_idx']


# =====================================================================
# MODEL
# =====================================================================
def load_model(path, device):
    model = models.efficientnet_b3(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(inf), nn.Linear(inf, 256),
        nn.ReLU(True), nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


# =====================================================================
# INFERENCE
# =====================================================================
@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_targets, all_probs = [], [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_targets.extend(labels.numpy())
    return (np.array(all_preds), np.array(all_targets),
            np.concatenate(all_probs, axis=0))


# =====================================================================
# GRAD-CAM
# =====================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1.0
        out.backward(gradient=one_hot)
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def generate_gradcam(model, test_df, device, out_dir, model_name):
    safe = model_name.replace(' ', '_').lower()
    gcam_dir = os.path.join(out_dir, f"gradcam_{safe}")
    os.makedirs(gcam_dir, exist_ok=True)
    gcam = GradCAM(model, model.features[-1])

    for cls in EVAL_CLASSES:
        cls_df = test_df[test_df['dx'] == cls]
        if len(cls_df) == 0:
            continue
        sample = cls_df.sample(n=min(NUM_GCAM, len(cls_df)), random_state=42)
        for idx, (_, row) in enumerate(sample.iterrows()):
            img_t = test_transform(Image.open(row['path']).convert('RGB'))
            img_t = img_t.unsqueeze(0).to(device)
            cam, pred_idx = gcam(img_t)

            img_np = inv_normalize(img_t.squeeze().cpu()).permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original (GT: {cls})")
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"Grad-CAM (Pred: {TRAIN_CLASSES[pred_idx]})")
            axes[2].imshow(img_np)
            axes[2].imshow(cam, cmap="jet", alpha=0.4)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(gcam_dir, f"{cls}_{idx}.png"), dpi=150)
            plt.close()
    print(f"    Grad-CAM -> {gcam_dir}/")


# =====================================================================
# PER-MODEL REPORTS
# =====================================================================
def make_safe_name(name):
    return name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').lower()


def generate_confusion_matrix(name, y_true, y_pred, out_dir):
    safe = make_safe_name(name)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=TRAIN_CLASSES).plot(
        cmap='Blues', xticks_rotation=45, ax=ax)
    ax.set_title(f"{name} -- Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe}_cm.png"), dpi=150)
    plt.close()


def generate_roc_curves(name, y_true, y_probs, out_dir):
    """One-vs-rest ROC per class + macro average."""
    safe = make_safe_name(name)
    n_samples = len(y_true)

    # Binarise ground truth
    y_bin = np.zeros((n_samples, NUM_CLASSES))
    for i, t in enumerate(y_true):
        y_bin[i, t] = 1

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    all_fpr, all_tpr, all_auc = {}, {}, {}

    for i, cls in enumerate(TRAIN_CLASSES):
        if cls == 'UNK':
            continue
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        all_fpr[cls] = fpr
        all_tpr[cls] = tpr
        all_auc[cls] = roc_auc_val
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"{cls} (AUC={roc_auc_val:.3f})")

    # Macro average ROC
    eval_indices = [i for i, c in enumerate(TRAIN_CLASSES) if c != 'UNK' and c in all_auc]
    if len(eval_indices) > 1:
        try:
            macro_auc = roc_auc_score(
                y_bin[:, eval_indices], y_probs[:, eval_indices],
                average='macro', multi_class='ovr')
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
            ax.set_title(f"{name} -- ROC Curves  (Macro AUC={macro_auc:.3f})")
        except Exception:
            ax.set_title(f"{name} -- ROC Curves")
    else:
        ax.set_title(f"{name} -- ROC Curves")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe}_roc.png"), dpi=150)
    plt.close()

    return all_auc


def generate_confidence_plot(name, y_true, y_pred, y_probs, out_dir):
    """Confidence distribution for correct vs incorrect predictions."""
    safe = make_safe_name(name)
    confs = y_probs.max(axis=1)
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(confs[correct], bins=50, alpha=0.6, label='Correct', color='#55A868')
    axes[0].hist(confs[~correct], bins=50, alpha=0.6, label='Incorrect', color='#C44E52')
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{name} -- Confidence Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-class mean confidence
    cls_conf_correct, cls_conf_incorrect = [], []
    for i, cls in enumerate(TRAIN_CLASSES):
        if cls == 'UNK':
            continue
        mask_cls = y_true == i
        if mask_cls.sum() == 0:
            continue
        c_mask = mask_cls & correct
        w_mask = mask_cls & ~correct
        cls_conf_correct.append(confs[c_mask].mean() if c_mask.sum() > 0 else 0)
        cls_conf_incorrect.append(confs[w_mask].mean() if w_mask.sum() > 0 else 0)

    x = np.arange(len(EVAL_CLASSES))
    w = 0.35
    axes[1].bar(x - w/2, cls_conf_correct, w, label='Correct', color='#55A868')
    axes[1].bar(x + w/2, cls_conf_incorrect, w, label='Incorrect', color='#C44E52')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(EVAL_CLASSES, rotation=45)
    axes[1].set_ylabel("Mean Confidence")
    axes[1].set_title("Per-Class Confidence")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe}_confidence.png"), dpi=150)
    plt.close()

    return {
        'mean_conf': float(confs.mean()),
        'correct_conf': float(confs[correct].mean()) if correct.sum() > 0 else 0,
        'incorrect_conf': float(confs[~correct].mean()) if (~correct).sum() > 0 else 0,
    }


def evaluate_single_model(name, path, test_df, test_loader, device, out_dir):
    """Full evaluation pipeline for one model. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Weights:    {path}")
    print(f"{'='*60}")

    model = load_model(path, device)
    y_pred, y_true, y_probs = run_inference(model, test_loader, device)
    y_conf = y_probs.max(axis=1)

    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    _, _, wt_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)

    # Per-class F1
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro Precision:   {mac_p:.4f}")
    print(f"  Macro Recall:      {mac_r:.4f}")
    print(f"  Macro F1:          {mac_f1:.4f}")
    print(f"  Weighted F1:       {wt_f1:.4f}")

    report_str = classification_report(
        y_true, y_pred, target_names=TRAIN_CLASSES, zero_division=0)
    print(f"\n{report_str}")

    # Artifacts
    generate_confusion_matrix(name, y_true, y_pred, out_dir)
    class_aucs = generate_roc_curves(name, y_true, y_probs, out_dir)
    conf_stats = generate_confidence_plot(name, y_true, y_pred, y_probs, out_dir)
    generate_gradcam(model, test_df, device, out_dir, name)

    # Macro AUC
    try:
        eval_idx = [i for i, c in enumerate(TRAIN_CLASSES) if c != 'UNK']
        y_bin = np.zeros((len(y_true), NUM_CLASSES))
        for i, t in enumerate(y_true):
            y_bin[i, t] = 1
        macro_auc = float(roc_auc_score(
            y_bin[:, eval_idx], y_probs[:, eval_idx],
            average='macro', multi_class='ovr'))
    except Exception:
        macro_auc = 0.0

    # Write individual text report
    safe = make_safe_name(name)
    report_path = os.path.join(out_dir, f"{safe}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{'='*65}\n")
        f.write(f"  {name}  --  ISIC 2019 Test Set Evaluation\n")
        f.write(f"{'='*65}\n\n")
        f.write(f"  Accuracy:            {acc:.4f}\n")
        f.write(f"  Balanced Accuracy:   {bal_acc:.4f}\n")
        f.write(f"  Macro Precision:     {mac_p:.4f}\n")
        f.write(f"  Macro Recall:        {mac_r:.4f}\n")
        f.write(f"  Macro F1:            {mac_f1:.4f}\n")
        f.write(f"  Weighted F1:         {wt_f1:.4f}\n")
        f.write(f"  Macro AUC (OvR):     {macro_auc:.4f}\n")
        f.write(f"  Mean Confidence:     {conf_stats['mean_conf']:.4f}\n")
        f.write(f"  Correct Confidence:  {conf_stats['correct_conf']:.4f}\n")
        f.write(f"  Incorrect Confidence:{conf_stats['incorrect_conf']:.4f}\n")
        f.write(f"\n{'='*65}\n")
        f.write(f"  Per-Class Breakdown\n")
        f.write(f"{'='*65}\n\n")
        f.write(f"  {'Class':8s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}  {'AUC':>8s}  {'Support':>8s}\n")
        f.write(f"  {'-'*52}\n")
        for i, cls in enumerate(TRAIN_CLASSES):
            if cls == 'UNK':
                continue
            cls_auc = class_aucs.get(cls, 0)
            f.write(f"  {cls:8s}  {per_p[i]:8.4f}  {per_r[i]:8.4f}  {per_f1[i]:8.4f}  {cls_auc:8.3f}  {per_sup[i]:8d}\n")
        f.write(f"\n{'='*65}\n")
        f.write(f"  Full Classification Report\n")
        f.write(f"{'='*65}\n\n")
        f.write(report_str)
        f.write(f"\n\nArtifacts:\n")
        f.write(f"  Confusion Matrix:   {safe}_cm.png\n")
        f.write(f"  ROC Curves:         {safe}_roc.png\n")
        f.write(f"  Confidence Plot:    {safe}_confidence.png\n")
        f.write(f"  Grad-CAM:           gradcam_{safe}/\n")
    print(f"    Report -> {report_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'name': name, 'accuracy': acc, 'balanced_accuracy': bal_acc,
        'mac_p': float(mac_p), 'mac_r': float(mac_r),
        'mac_f1': float(mac_f1), 'wt_f1': float(wt_f1),
        'macro_auc': macro_auc,
        'conf': conf_stats,
        'per_class_f1': {TRAIN_CLASSES[i]: float(per_f1[i])
                         for i in range(NUM_CLASSES) if TRAIN_CLASSES[i] != 'UNK'},
        'per_class_auc': {k: float(v) for k, v in class_aucs.items()},
    }


# =====================================================================
# CROSS-MODEL COMPARISONS
# =====================================================================
def plot_comparison_bars(results, out_dir):
    names = [r['name'] for r in results]
    metrics = ['accuracy', 'balanced_accuracy', 'mac_f1', 'wt_f1', 'macro_auc']
    labels = ['Accuracy', 'Balanced Acc', 'Macro F1', 'Weighted F1', 'Macro AUC']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    x = np.arange(len(names))
    n_metrics = len(metrics)
    w = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 3), 7))
    for j, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [r[metric] for r in results]
        offset = (j - n_metrics / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('ISIC 2019 Test Set -- Model Comparison')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=150)
    plt.close()


def plot_per_class_f1_heatmap(results, out_dir):
    rows = []
    for r in results:
        row = {'Model': r['name']}
        row.update(r['per_class_f1'])
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')

    fig, ax = plt.subplots(figsize=(max(10, len(EVAL_CLASSES)), max(4, len(results) * 1.2)))
    sns.heatmap(df[EVAL_CLASSES], annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_f1_heatmap.png'), dpi=150)
    plt.close()


def write_combined_report(results, out_dir):
    path = os.path.join(out_dir, 'combined_summary.txt')
    with open(path, 'w') as f:
        f.write(f"{'='*100}\n")
        f.write(f"  ISIC 2019 Test Set -- Combined Evaluation Summary\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"  Models evaluated: {len(results)}\n\n")

        # Summary table
        header = f"  {'Model':25s}  {'Acc':>7s}  {'BalAcc':>7s}  {'MacP':>7s}  {'MacR':>7s}  {'MacF1':>7s}  {'WtF1':>7s}  {'AUC':>7s}  {'Conf':>7s}\n"
        f.write(header)
        f.write(f"  {'-'*90}\n")
        for r in results:
            f.write(f"  {r['name']:25s}  {r['accuracy']:7.4f}  {r['balanced_accuracy']:7.4f}  "
                    f"{r['mac_p']:7.4f}  {r['mac_r']:7.4f}  {r['mac_f1']:7.4f}  "
                    f"{r['wt_f1']:7.4f}  {r['macro_auc']:7.4f}  {r['conf']['mean_conf']:7.4f}\n")

        # Best model per metric
        f.write(f"\n{'='*100}\n")
        f.write(f"  Best Model per Metric\n")
        f.write(f"{'='*100}\n\n")
        for metric, label in [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced Accuracy'),
                               ('mac_f1', 'Macro F1'), ('wt_f1', 'Weighted F1'),
                               ('macro_auc', 'Macro AUC')]:
            best = max(results, key=lambda r: r[metric])
            f.write(f"  {label:25s}: {best['name']}  ({best[metric]:.4f})\n")

        # Per-class F1 table
        f.write(f"\n{'='*100}\n")
        f.write(f"  Per-Class F1 Scores\n")
        f.write(f"{'='*100}\n\n")
        header = f"  {'Model':25s}  " + "  ".join(f"{c:>7s}" for c in EVAL_CLASSES) + "\n"
        f.write(header)
        f.write(f"  {'-'*(25 + 9*len(EVAL_CLASSES))}\n")
        for r in results:
            vals = "  ".join(f"{r['per_class_f1'].get(c, 0):7.4f}" for c in EVAL_CLASSES)
            f.write(f"  {r['name']:25s}  {vals}\n")

        # Per-class AUC table
        f.write(f"\n{'='*100}\n")
        f.write(f"  Per-Class AUC (One-vs-Rest)\n")
        f.write(f"{'='*100}\n\n")
        header = f"  {'Model':25s}  " + "  ".join(f"{c:>7s}" for c in EVAL_CLASSES) + "\n"
        f.write(header)
        f.write(f"  {'-'*(25 + 9*len(EVAL_CLASSES))}\n")
        for r in results:
            vals = "  ".join(f"{r['per_class_auc'].get(c, 0):7.3f}" for c in EVAL_CLASSES)
            f.write(f"  {r['name']:25s}  {vals}\n")

        # Confidence
        f.write(f"\n{'='*100}\n")
        f.write(f"  Confidence Statistics\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"  {'Model':25s}  {'Mean':>8s}  {'Correct':>8s}  {'Incorrect':>8s}\n")
        f.write(f"  {'-'*55}\n")
        for r in results:
            c = r['conf']
            f.write(f"  {r['name']:25s}  {c['mean_conf']:8.4f}  {c['correct_conf']:8.4f}  {c['incorrect_conf']:8.4f}\n")

        f.write(f"\n{'='*100}\n")
        f.write(f"  Artifacts\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"  model_comparison.png        -- Side-by-side metric bars\n")
        f.write(f"  per_class_f1_heatmap.png    -- F1 heatmap across models\n")
        for r in results:
            safe = make_safe_name(r['name'])
            f.write(f"  {safe}_report.txt\n")
            f.write(f"  {safe}_cm.png\n")
            f.write(f"  {safe}_roc.png\n")
            f.write(f"  {safe}_confidence.png\n")
            f.write(f"  gradcam_{safe}/\n")

        f.write(f"\n{'='*100}\n")

    print(f"\n  Combined report -> {path}")

    # Also save as JSON for programmatic use
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON results  -> {json_path}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    p = argparse.ArgumentParser(
        description='Evaluate EfficientNet-B3 models on ISIC 2019 test set')
    p.add_argument('--test-images', default='datasets/ISIC_2019_Test_Input')
    p.add_argument('--test-gt', default='datasets/ISIC_2019_Test_GroundTruth.csv')
    p.add_argument('--model', action='append', default=[],
                   help='Repeatable. Format: "DisplayName:path/to/weights.pth"')
    p.add_argument('--output-dir', default='outputs/isic2019_test_eval')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = p.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse model specs
    model_specs = []
    for spec in args.model:
        if ':' not in spec:
            print(f"  WARNING: Skipping malformed --model '{spec}' (expected Name:path)")
            continue
        name, path = spec.split(':', 1)
        if not os.path.isfile(path):
            print(f"  WARNING: Model file not found, skipping: {name} -> {path}")
            continue
        model_specs.append((name.strip(), path.strip()))

    if not model_specs:
        print("ERROR: No valid models provided. Use --model 'Name:path' (repeatable)")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  ISIC 2019 Test Set Evaluation")
    print(f"{'='*65}")
    print(f"  Device:  {device}")
    print(f"  Models:  {len(model_specs)}")
    for name, path in model_specs:
        print(f"    {name:25s} -> {path}")

    # Load test data (once)
    print(f"\n--- Loading test data ---")
    test_df = load_test_data(args.test_images, args.test_gt)
    test_ds = TestDataset(test_df, test_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Evaluate each model
    results = []
    for name, path in model_specs:
        r = evaluate_single_model(name, path, test_df, test_loader,
                                  device, args.output_dir)
        results.append(r)

    # Cross-model comparisons
    if len(results) > 1:
        print(f"\n--- Generating cross-model comparisons ---")
        plot_comparison_bars(results, args.output_dir)
        plot_per_class_f1_heatmap(results, args.output_dir)

    write_combined_report(results, args.output_dir)

    # Console summary
    print(f"\n{'='*100}")
    print(f"  Summary")
    print(f"{'='*100}")
    for r in results:
        print(f"  {r['name']:25s}  acc={r['accuracy']:.4f}  bal={r['balanced_accuracy']:.4f}  "
              f"macF1={r['mac_f1']:.4f}  wtF1={r['wt_f1']:.4f}  AUC={r['macro_auc']:.4f}")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
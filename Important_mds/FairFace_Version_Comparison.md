# FairFace Model Version Comparison

## Overview

Skin tone classifier using EfficientNet-B4, trained on FairFace dataset.
- **v1.0–v2.6:** 6-way Fitzpatrick (Type I–VI) with L*-derived labels
- **v3.0:** 5-way MST (ITA-derived labels) — paradigm shift

---

## Version Summary

| Version | Key Changes | Macro Acc | Collapse? | Notes |
|---------|-------------|:---------:|:---------:|-------|
| **1.0** | Baseline EfficientNet-B4, equal L* centroids | ~79% (wtd) | ❌ III/II | High overfitting |
| **2.0** | Widened L* centroids, rebalancing | ~55% | ❌ III/II→0% | Severe overfitting |
| **2.1** | Rollback overcorrections, scalar focal alpha | ~67% (wtd) | ❌ III/II ~1% | Val loss stabilised |
| **2.2** | OrdinalCE loss, 3-way mode | ~63% | ✅ III 9%, II 19% | First middle-class fix |
| **2.3** | ↑patience, ↑LR, val_acc tracking | ~69% (wtd) | ❌ III/II ~0.5% | Best Fitz accuracy, overfit |
| **2.4** | Partial freeze, MixUp, ↑WD | ~63% | ⚠️ III 4%, II 9% | Underfitting |
| **2.5** | ↑backbone, wider sigma, ↓MixUp | ~67% | ❌ III/II 1% | Sigma too wide |
| **2.6** | Class weights, tight sigma, no MixUp | ~67% | ❌ III/II collapsed | Labels are the problem |
| **3.0** 🏆 | **MST-5 (ITA labels), paradigm shift** | **77.0%** | **✅ All ≥65%** | **No collapse!** |

---

## v3.0 Results — MST-5 (Current Best)

### Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|--------:|
| Very Dark (MST 9-10) | 0.61 | **0.90** | 0.73 | 1,263 |
| Dark (MST 7-8) | 0.77 | **0.84** | 0.80 | 4,793 |
| Medium (MST 5-6) | **0.96** | 0.65 | 0.77 | 4,094 |
| Light (MST 3-4) | 0.54 | **0.81** | 0.65 | 463 |
| Very Light (MST 1-2) | 0.59 | **0.74** | 0.66 | 341 |

| Metric | Value |
|--------|------:|
| Macro Accuracy | **0.7704** |
| Macro Precision | 0.6951 |
| Macro Recall | 0.7857 |
| Macro F1 | **0.7223** |

### Confusion Matrix

![v3.0 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-3.0-mst5/fairface_confusion_matrix.png)

### Latent Space (UMAP)

![v3.0 UMAP](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-3.0-mst5/fairface_latent_space.png)

**Observations:**
- Clean ordinal gradient — skin tones transition smoothly through embedding space
- Distinct clusters for Very Dark and Very Light at extremes
- Medium tones show good separation (unlike Fitzpatrick III/IV overlap)
- No single class dominates or absorbs neighbours

### Training Curves

![v3.0 Training Curves](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-3.0-mst5/fairface_training_curves.png)

**Observations:**
- Steady convergence after fine-tune start (~epoch 10)
- Val loss tracks train loss without divergence — no overfitting
- Val accuracy reaches ~75% and holds stable

---

## Per-Class Recall History (Fitzpatrick v1.0–v2.4)

| Class | v1.0 | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 |
|-------|:----:|:----:|:----:|:----:|:----:|:----:|
| **Type VI** | 78.9% | 51.7% | 90.5% | 84.7% | 89.6% | 84.8% |
| **Type V** | 77.5% | 78.0% | 75.4% | 72.0% | 76.1% | 62.1% |
| **Type IV** | 56.4% | 77.1% | 42.8% | 45.9% | 65.9% | 52.2% |
| **Type III** | 53.1% | 0.1% | 0.7% | 8.8% | 0.4% | 3.8% |
| **Type II** | 34.7% | 0.0% | 0.9% | 19.3% | 0.5% | 8.7% |
| **Type I** | 86.1% | 91.5% | 89.6% | 66.3% | 89.6% | 82.6% |

> [!IMPORTANT]
> v3.0 is not directly comparable to v1.0–v2.6 because the class system changed from 6-way Fitzpatrick to 5-way MST. The Fitzpatrick Type III/II collapse was caused by overlapping L* centroids, which ITA-based MST labels completely avoid.

---

## Confusion Matrix Gallery (Fitzpatrick Era)

### v1.0 — Baseline

![v1.0 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-1.0/fairface_confusion_matrix.png)

### v2.0 — Widened Centroids

![v2.0 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.0/fairface_confusion_matrix.png)

### v2.1 — Ordinal Label Smoothing

![v2.1 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.1/fairface_confusion_matrix.png)

### v2.2 — OrdinalCE Breakthrough

![v2.2 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.2/fairface_confusion_matrix.png)

### v2.3 — Best Fitzpatrick Accuracy

![v2.3 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.3/fairface_confusion_matrix.png)

### v2.4 — Partial Freeze + MixUp

![v2.4 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.4/fairface_confusion_matrix.png)

---

## v2.4 UMAP Latent Space

![v2.4 Latent Space](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.4/fairface_latent_space.png)

---

## Version Changelog (Fitzpatrick Era)

### v1.0
- EfficientNet-B4 backbone, ImageNet pretrained
- Equal-width L* centroids for 6-way classification
- Standard cross-entropy loss; full backbone unfreezing

### v2.0
- Widened L* centroids to separate Type IV
- **Regression:** Type III/II collapsed to 0% recall

### v2.1
- Removed per-class focal alpha; scalar 0.25
- Reverted fine-tune LR 1e-5 → 5e-6; patience 15 → 10
- Toned down minority augmentations

### v2.2
- OrdinalCrossEntropy replaced FocalLoss (sigma=1.0, gamma=2.0)
- ReduceLROnPlateau replaced CosineAnnealingWarmRestarts
- **Breakthrough:** First non-zero Type III/II recall (9%/19%)

### v2.3
- Best model tracked by val_acc (not val_loss)
- ↑patience (20), ↑LR (8e-6)
- Highest Fitzpatrick weighted acc (~69%) but severe overfitting

### v2.4
- Partial backbone freeze (last 2 blocks, ~5M params)
- MixUp (alpha=0.4), weight decay 1e-5 → 5e-5
- UMAP visualisation integrated

### v2.5–v2.6
- Various sigma/gamma/weight experiments — all failed to fix middle-class collapse
- **Conclusion:** The labels (L*-only Fitzpatrick) were the root cause, not the model

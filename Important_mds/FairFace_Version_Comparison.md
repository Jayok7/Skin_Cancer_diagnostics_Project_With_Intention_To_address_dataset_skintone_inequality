# FairFace Model Version Comparison

## Overview

6-way Fitzpatrick skin tone classifier (Type I–VI) using EfficientNet-B4, trained on FairFace dataset with L*-derived labels.

---

## Version Summary

| Version       | Key Changes                                    | Weighted Acc |      Middle-Class Fix      | Overfitting |
| ------------- | ---------------------------------------------- | :----------: | :-------------------------: | :---------: |
| **1.0** | Baseline EfficientNet-B4, equal L* centroids   |     ~79%     |  ❌ Type III/II collapsed  |    High    |
| **2.0** | Widened L* centroids, better Type IV boundary  |     ~73%     |   ⚠️ Type III still ~0%   |   Medium   |
| **2.1** | Ordinal label smoothing (v1)                   |     ~76%     | ⚠️ Type III/II still ~1% |   Medium   |
| **2.2** | 3-way mode support, tweaked centroids          |     ~66%     | ✅ Type III 9%, Type II 19% |   Medium   |
| **2.3** | Optimised centroids, ordinal smoothing refined |     ~69%     | ✅ Type III 4%, Type II 30% |    High    |
| **2.4** | Partial backbone freeze, MixUp, ↑weight decay |     ~63%     | ⚠️ Type III 4%, Type II 9% | Underfitting |
| **2.5** | ↑backbone capacity, wider sigma, ↓MixUp, ↓WD |   Pending   |           Pending           | Expected fix |

---

## Per-Class Recall (from Confusion Matrices)

*Recall = correct predictions / total samples in that class.*

| Class              | v1.0 | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 | v2.5 |
| ------------------ | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Type VI**  | 78.9% | 51.7% | 90.5% | 84.7% | 89.6% | 84.8% | — |
| **Type V**   | 77.5% | 78.0% | 75.4% | 72.0% | 76.1% | 62.1% | — |
| **Type IV**  | 56.4% | 77.1% | 42.8% | 45.9% | 65.9% | 52.2% | — |
| **Type III** | 53.1% | 0.1% | 0.7% | 8.8% | 0.4% | 3.8% | — |
| **Type II**  | 34.7% | 0.0% | 0.9% | 19.3% | 0.5% | 8.7% | — |
| **Type I**   | 86.1% | 91.5% | 89.6% | 66.3% | 89.6% | 82.6% | — |

### Key Observations

1. **Type VI and Type I** consistently perform well — they occupy the extremes of the L* spectrum with clear boundaries
2. **Type III and Type II** are the hardest — they sit in the middle of the ordinal space and blend with adjacent types
3. **v2.2** was the first version to meaningfully predict middle classes, despite lower overall accuracy
4. **v2.0** caused Type III/II to collapse to 0% — aggressive centroids pushed all predictions to neighbouring classes

---

## Confusion Matrix Gallery

### v1.0 — Baseline

![v1.0 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-1.0/fairface_confusion_matrix.png)

### v2.0 — Widened Centroids

![v2.0 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.0/fairface_confusion_matrix.png)

### v2.1 — Ordinal Label Smoothing

![v2.1 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.1/fairface_confusion_matrix.png)

### v2.2 — Tweaked Centroids

![v2.2 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.2/fairface_confusion_matrix.png)

### v2.3 — Optimised Centroids + Refined Smoothing

![v2.3 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.3/fairface_confusion_matrix.png)

### v2.4 — Partial Freeze + MixUp + Weight Decay

![v2.4 Confusion Matrix](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.4/fairface_confusion_matrix.png)

---

## v2.4 UMAP Latent Space

![v2.4 Latent Space Visualisation](file:///D:/skin%20cancer%20project/outputs/FairFace-Model-2.4/fairface_latent_space.png)

**Observations:**

- Darkest tones (Type VI) form a distinct cluster (top-right)
- Lightest tones (Type I) cluster at bottom-left
- Middle types (III–IV) heavily overlap, confirming why recall is lowest for these classes
- The ordinal gradient is visible — skin tones transition smoothly through the embedding space

---

## Version Changelog

### v1.0

- EfficientNet-B4 backbone, ImageNet pretrained
- Equal-width L* centroids for 6-way classification
- Standard cross-entropy loss
- Full backbone unfreezing after head training

### v2.0

- Widened L* centroids to better separate Type IV from neighbours
- Larger gap between dark/light types
- **Regression:** Type III and Type II collapsed to 0% recall

| File                       | Change                                                              | Why                                                                                         |
| -------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| compute_fairface_labels.py | Centroids: 68/66/64/58/47/43 →<br />**69/64/59/53/46/38**    | Minimum gap increased from 2→**5 L*** , derived from MILK10k+MSKCC weighted averages |
| fairface_dataset.py        | Doubled ColorJitter, added GaussianBlur + RandomErasing             | Improve domain robustness for clinical images                                               |
| train_fairface.py          | label_smoothing=0.1, fine-tune LR 5e-6 → 1e-5                     | Soften noisy pseudo-labels, faster adaptation                                               |
| submit_csf.sh              | 2-step pipeline: re-label → train, outputs to FairFace-Model-2.0/ | One-click retrain, original model preserved                                                 |



- Introduced ordinal label smoothing (softens labels by assigning probability to adjacent classes)
- Minor improvements to Type IV recall
- Middle classes still collapsed

| Setting                 | v2.0 (overfitted) | v2.1 (fixed) |
| ----------------------- | ----------------- | ------------ |
| Focal alpha             | Per-class tensor  | scalar 0.25  |
| Fine-tune LR            | 1e-5              | 5e-6         |
| Early stopping patience | 15 epochs         | 10 epochs    |

### v2.2

- Added 3-way mode (Light/Medium/Dark) for coarser classification
- Implementing Ordinal-Aware Loss
- Replaced FocalLoss with OrdinalCrossEntropy (soft Gaussian labels, sigma=1.0, gamma=2.0)
- **Breakthrough:** First version with non-zero Type III/II recall (9%/19%)

| Setting               | v2.1                        | v2.2                                      |
| --------------------- | --------------------------- | ----------------------------------------- |
| Loss                  | FocalLoss(alpha=0.25)       | OrdinalCrossEntropy(sigma=1.0, gamma=2.0) |
| Scheduler             | CosineAnnealingWarmRestarts | ReduceLROnPlateau(patience=5, factor=0.5) |
| Fine-tune epochs      | 40                          | 60                                        |
| Best model tracked by | val_acc                     | val_loss (more reliable with soft labels) |
| LR reduction          | Oscillating cosine          | Only when val_loss plateaus for 5 epochs  |

### v2.3

- Fully optimised L* centroids based on data distribution analysis
- Refined ordinal label smoothing parameters
- Best overall weighted accuracy (~69%)
- **Problem:** Significant overfitting (train loss 0.35 vs val loss 0.70)

| Setting                    | v2.2 (stopped early) | v2.3 (proposed)                                   |
| -------------------------- | -------------------- | ------------------------------------------------- |
| Best model tracked by      | val_loss             | **val_acc** (what we actually care about)   |
| Early stopping patience    | 10                   | **20** (give ordinal learning more time)    |
| ReduceLROnPlateau patience | 5                    | **8** (don't cut LR too aggressively)       |
| Fine-tune starting LR      | 5e-6                 | **8e-6** (slightly faster initial learning) |

### v2.4

- Partial backbone freeze (only last 2 EfficientNet blocks unfrozen → ~5M vs 19M params)
- MixUp augmentation (alpha=0.4) for better ordinal interpolation
- Increased weight decay (1e-5 → 5e-5)
- UMAP latent space visualisation integrated into training
- **Goal:** Reduce overfitting gap while maintaining or improving middle-class recall

![User uploaded media 1]()

![User uploaded media 1]()

| Change          | v2.3                    | v2.4                                      |
| --------------- | ----------------------- | ----------------------------------------- |
| Backbone freeze | All 19M params unfrozen | **Last 2 blocks only (~5M params)** |
| Augmentation    | Standard transforms     | **+ MixUp (alpha=0.4)**             |
| Weight decay    | 1e-5                    | **5e-5**                            |

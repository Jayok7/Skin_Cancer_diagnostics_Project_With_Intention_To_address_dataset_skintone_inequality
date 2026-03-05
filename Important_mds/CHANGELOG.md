# FairFace Skin Tone Classifier — Changelog

Version history for the FairFace skin tone classifier (MST and legacy Fitzpatrick).

---

## v3.0 — MST-5 *(2026-03-04)*

**Focus:** Paradigm shift from Fitzpatrick (L*-only) to MST-5 (ITA-based) labelling

### Diagnosis

Fitzpatrick centroids are fundamentally broken for Types I-III: MSKCC actual medians are 65.1, 63.9, and 65.1 L* respectively — within 1.2 L* of each other (noise). After 3 iterations of loss-only fixes (v2.4–v2.6), the Type IV attractor persisted. The problem is upstream in the labels, not in the model.

### Paradigm Shift

- **Old:** L*-only → nearest Fitzpatrick centroid → 6 classes (5 L* gaps)
- **New:** ITA = arctan((L*-50)/b*) × 180/π → MST thresholds → 5 merged classes (from MST-10)

ITA uses both L* (lightness) AND b* (yellow-blue), giving much better separation in medium tones. Based on the [beyond-fitzpatrick](https://github.com/ssitaru/beyond-fitzpatrick) approach. MST-10 merged to MST-5 with undersampling of Dark/Medium to reduce extreme oversampling needs.

### Changes

- **[NEW] `compute_mst_labels.py`** — ITA-based labelling with published MST-10 thresholds
- **`fairface_dataset.py`** — supports MST-10, MST-5, Fitzpatrick-6, Fitzpatrick-3 via `num_classes` arg
- **`train_fairface.py`** — clean OrdinalCE (sigma=1.0, gamma=2.0), default 5 classes, MixUp re-enabled
- **`submit_csf.sh`** — 2-step pipeline (ITA labels → training), output `FairFace-Model-3.0-mst5`

### Results

| Metric            | Value            |
| ----------------- | ---------------- |
| Macro Accuracy    | **0.7704** |
| Macro Precision   | 0.6951           |
| Macro Recall      | 0.7857           |
| Macro F1          | **0.7223** |
| Weighted Accuracy | 0.77             |

| Class                | Precision      | Recall         | F1   | Support |
| -------------------- | -------------- | -------------- | ---- | ------- |
| Very Dark (MST 9-10) | 0.61           | **0.90** | 0.73 | 1,263   |
| Dark (MST 7-8)       | 0.77           | **0.84** | 0.80 | 4,793   |
| Medium (MST 5-6)     | **0.96** | 0.65           | 0.77 | 4,094   |
| Light (MST 3-4)      | 0.54           | **0.81** | 0.65 | 463     |
| Very Light (MST 1-2) | 0.59           | **0.74** | 0.66 | 341     |

**Observations:** Highest macro accuracy (77%). **No collapse** — all 5 classes have meaningful recall (≥65%). This is the breakthrough version. The latent space (UMAP) shows clean ordinal gradient with distinct clusters. Training curves show steady convergence after fine-tune start with no val loss divergence.

---

## v2.6

**Focus:** Break the Type IV attractor with class-weighted loss

### Diagnosis

v2.5's wider sigma (1.5) backfired — by giving 25% credit to adjacent classes, the model hedged harder into Type IV (82% of Type III swallowed, up from 69%).

### Changes

- **Class-weighted loss** — inverse-frequency weights (Type II ×2.38, Type III ×1.67)
- **Tight sigma** `0.7` (was 1.5) — only ~7% credit for adjacent classes
- **Stronger focal gamma** `3.0` (was 2.0) — more focus on hard samples
- **MixUp disabled** — `alpha=0.0` (was 0.2)

### Results

**Extreme underfitting + continued Type IV collapse.** Type III 87% swallowed by IV, Type II 77%. Class weights addressed the wrong problem — the labels themselves were noisy.

### Diagnosis

v2.4's partial freeze was too aggressive — train acc (47%) fell *below* val acc (63%), causing underfitting. Type IV absorbed 69% of Type III and 53% of Type II predictions.

### Changes

- **Wider backbone unfreeze** — blocks 4–7 unfrozen (~10M trainable, was blocks 6–7 / ~5M)
- **Wider ordinal smoothing** — `sigma=1.5` (was 1.0) → adjacent classes get ~25% credit (was ~15%)
- **Reduced MixUp** — `alpha=0.2` (was 0.4) → less blurring of ambiguous middle-class boundaries
- **Lower weight decay** — `1e-5` (was 5e-5) → let the model learn more (underfitting, not overfitting)
- **Higher fine-tune LR** — `1.2e-5` (was 8e-6) → faster convergence with more trainable params

### Results

| Metric          | Value                             |
| --------------- | --------------------------------- |
| Macro Accuracy  | **0.6730** (up from 0.6331) |
| Macro F1        | 0.46                              |
| Type III Recall | 1% (down from 3.8%)               |
| Type II Recall  | 1% (down from 8.7%)               |

**Observations:** Extremes improved strongly (Type VI 92%, V 72%, I 87%) and macro accuracy rose to 67%. But wider sigma made Type IV attractor worse — 82% of Type III and 68% of Type II swallowed by Type IV. Backbone unfreeze and LR changes were beneficial; sigma increase was harmful.

### Files Modified

- `train_fairface.py` — sigma, unfreeze_from, lr, weight_decay, mixup_alpha
- `submit_csf.sh` — output dir `FairFace-Model-2.5`, 16h time limit

---

## v2.4

**Focus:** Regularisation for cross-domain generalisation

### Changes

- **Partial backbone freeze** — only last 2 EfficientNet-B4 blocks (6-7) unfrozen (~5M trainable vs 19M)
- **MixUp augmentation** (alpha=0.4) — interpolates image pairs and ordinal labels during fine-tuning
- **Weight decay** increased from `1e-5` → `5e-5`
- **NaN val_loss fix** — moved loss computation outside `autocast` context in `evaluate()`; added logit clamping
- **UMAP latent space visualisation** added to post-training evaluation
- **`umap-learn`** added to CSF dependencies

### Results

| Metric          | Value            |
| --------------- | ---------------- |
| Macro Accuracy  | **0.6331** |
| Macro F1        | 0.46             |
| Type III Recall | 3.8%             |
| Type II Recall  | 8.7%             |

**Observations:** Overfitting was reduced (train/val loss gap narrower than v2.3), but the partial freeze overcorrected — model was underfitting (train acc 47% < val acc 63%). UMAP shows smooth ordinal gradient but heavy overlap in middle types. Type IV became a massive attractor class.

### Files Modified

- `train_fairface.py` — `partial_unfreeze_backbone()`, `mixup_data()`, `OrdinalCrossEntropy` float32 fix
- `fairface_dataset.py` — unchanged from v2.1
- `submit_csf.sh` — output dir `FairFace-Model-2.4`, 60 fine-tune epochs

---

## v2.3

**Focus:** Let the model train longer with better tracking

### Changes

- **Best model tracked by val_acc** instead of val_loss (ordinal loss can plateau while accuracy improves)
- **Early stopping patience** `10` → `20`
- **ReduceLROnPlateau patience** `5` → `8`
- **Fine-tune LR** `5e-6` → `8e-6`

### Results

| Metric          | Value            |
| --------------- | ---------------- |
| Macro Accuracy  | **0.6901** |
| Macro F1        | 0.52             |
| Type III Recall | 0.4%             |
| Type II Recall  | 0.5%             |

**Observations:** Highest overall accuracy so far. Model trained all 70 epochs. However, clear overfitting (train loss ~0.35 vs val loss ~0.70) and middle classes still collapsed. Confirmed that the model architecture was sound but regularisation was needed.

---

## v2.2

**Focus:** Ordinal-aware loss to fix middle-class collapse

### Changes

- **`OrdinalCrossEntropy`** replaced `FocalLoss` — soft Gaussian labels (sigma=1.0) give adjacent classes ~15% credit
- **`ReduceLROnPlateau`** replaced `CosineAnnealingWarmRestarts` — reduces LR only when val_loss stalls
- **Fine-tune epochs** `40` → `60`
- **Best model tracked by val_loss** (later reverted in v2.3)

### Results

| Metric          | Value                      |
| --------------- | -------------------------- |
| Macro Accuracy  | 0.6340                     |
| Macro F1        | **0.4893**           |
| Type III Recall | **8.8%** (was 0.7%)  |
| Type II Recall  | **19.3%** (was 0.9%) |

**Observations:** Ordinal loss successfully broke the middle-class collapse — Types II and III went from near-zero to meaningful recall. However, early stopping triggered at epoch 28 because val_loss was tracked (it plateaued while val_acc was still rising). Overall accuracy dropped due to the accuracy/spread trade-off.

---

## v2.1

**Focus:** Roll back overcorrections from v2.0

### Changes

- **Removed per-class focal alpha** — `WeightedRandomSampler` is now the sole class-balancing mechanism
- **Scalar focal alpha** `0.25` (was per-class `[0.1…0.9]`)
- **Fine-tune LR** reverted `1e-5` → `5e-6`
- **Early stopping patience** `15` → `10`
- **Minority augmentations toned down** — removed `RandomRotation(15°)` and `RandomPerspective(0.15)`, reduced `ColorJitter` and `RandomErasing` intensity

### Results

| Metric          | Value            |
| --------------- | ---------------- |
| Macro Accuracy  | **0.6661** |
| Macro F1        | 0.4517           |
| Type III Recall | 0.7%             |
| Type II Recall  | 0.9%             |

**Observations:** Significant improvement over v2.0. Val loss much healthier with no wild divergence. Extremes (Type VI: 90%, Type I: 90%) recovered strongly. Middle classes still near-zero — identified need for ordinal-aware approach.

---

## v2.0

**Focus:** Address class imbalance with aggressive rebalancing

### Changes

- **`WeightedRandomSampler`** for balanced epoch sampling
- **Per-class focal alpha** (normalised to [0.1, 0.9] range)
- **Stronger minority augmentations** — `RandomRotation`, `RandomPerspective`, heavy `ColorJitter`
- **Fine-tune LR** `5e-6` → `1e-5`
- **Widened L\* centroids** for label assignment

### Results

| Metric            | Value           |
| ----------------- | --------------- |
| Macro Accuracy    | ~0.55           |
| Effective classes | 3-4 (collapsed) |

**Observations:** Severe overfitting — val loss increased, train/val accuracy gap widened. Model collapsed into 3-4 effective classes. Root cause: double-correction from combining `WeightedRandomSampler` with per-class focal alpha, plus aggressive LR. Led to rollback in v2.1.

---

## v1.0

**Focus:** Baseline 6-way classifier

### Architecture

- **EfficientNet-B4** (ImageNet pretrained, 380×380 input)
- **2-stage training:** head-only → full fine-tune
- **FocalLoss** (gamma=2.0)
- Standard augmentations (flip, jitter, blur, erasing)

### Results

- Baseline accuracy established
- Class imbalance identified as primary issue for future versions

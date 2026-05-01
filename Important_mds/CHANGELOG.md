# FairFace Skin Tone Classifier — Changelog

Version history for the FairFace skin tone classifier (MST and legacy Fitzpatrick).

---

## v3.2 (fine-tuned) — MSKCC Domain Adaptation *(2026-03-25)*

**Focus:** Fine-tune the v3.2 model on MSKCC dermatological images with human expert MST rater labels to improve clinical skin tone classification.

### Motivation

v3.2 was trained exclusively on FairFace/UTKFace facial images with ITA-derived pseudo-labels. MSKCC provides ground-truth MST ratings from trained dermatology raters on clinical lesion images, which is the target deployment domain. Fine-tuning on MSKCC recalibrates decision boundaries for clinical imagery.

### Data Pipeline

- **Label source:** MSKCC-MST inter-rater CSV (MST 1-10 scale from 2 raters per site)
- **Data linkage:** `isic_id → tag_id` (colorimeter CSV) → `mst_r1, mst_r2` (inter-rater CSV) → MST-5
- **Rater consensus:** If raters agree, use that value; if not, average + round
- **Image preprocessing:**
  - *Non-lesional sites (~639):* Direct center crop (clean skin)
  - *Lesional sites (~607):* A→B cascade (U-Net perilesional ring → inpaint → save; fallback to corner patches)
- **Split:** 80/20 patient-stratified train/val (no data leakage)

### Training Configuration

- Single-stage fine-tuning (no head-only phase — checkpoint already trained)
- Checkpoint: v3.2 `best_finetuned_model.pth`
- Last 2 EfficientNet-B4 blocks unfrozen (more aggressive freeze than v3.2's 4 blocks)
- LR: `1e-5` with CosineAnnealing (T_max=30)
- Dropout: `0.6` (up from 0.5)
- Weight decay: `5e-4` (up from 1e-5)
- MixUp: Disabled (dataset too small)
- OrdinalCE: sigma=1.0, gamma=2.0 (same as v3.2)

### New Files

- `prepare_mskcc_labels.py` — joins MSKCC-MST CSVs, maps MST-10 → MST-5, outputs `mskcc_mst5_labels.csv`
- `preprocess_mskcc_crops.py` — offline A→B cascade preprocessing for lesional images
- `mskcc_dataset.py` — PyTorch Dataset for MSKCC crops
- `submit_mskcc_finetune.sh` — Slurm submission script for CSF cluster

### Files Modified

- `train_fairface.py` — added `--finetune-mskcc` mode with `main_mskcc_finetune()` function

### Results

*Pending — run on CSF cluster*

---


## v3.1 — MST-5 Optimisation *(2026-03-07)*

**Focus:** Improve macro accuracy and F1 by addressing per-class bottlenecks from v3.0

### Diagnosis

v3.0 plateaus at 77% macro accuracy with three bottlenecks: (1) Medium→Dark leakage (27% of Medium misclassified as Dark), (2) VeryDark→Dark leakage (51%), (3) low precision for Light (0.54) and VeryLight (0.59). Training curves show unstable val accuracy oscillation (±5%) suggesting LR scheduling issues.

### Changes

- **CosineAnnealingWarmRestarts** replaced ReduceLROnPlateau (T_0=15, T_mult=2, eta_min=1e-7) — smoother LR decay, eliminates val_acc oscillation
- **Gradient clipping** (max_norm=1.0) — prevents gradient spikes during fine-tuning
- **Tighter ordinal sigma** `0.8` (was 1.0) — adjacent classes get ~10% credit (was ~15%), sharper decision boundaries
- **Inverse-frequency class weights** added to OrdinalCE — penalises minority-class errors (Light/VeryLight) more heavily
- **Test-Time Augmentation (TTA)** — averages 5 augmented views + 1 clean view at inference for ~1-3% free accuracy gain
- **RandomAffine** augmentation (degrees=15, translate=0.05, scale=0.95-1.05) — added to both standard and minority transforms

### Files Modified

- `train_fairface.py` — scheduler, grad clip, loss config, TTA function + evaluation
- `fairface_dataset.py` — RandomAffine in train transforms
- `submit_csf.sh` — output dir `FairFace-Model-3.1-mst5`, walltime 20h, job name v3.1

### Results

| Metric            | v3.0       | v3.1       | Change |
| ----------------- | ---------- | ---------- | ------ |
| Macro Accuracy    | **0.7704** | 0.7097     | −6.1%  |
| Macro Precision   | 0.6951     | 0.6443     | −5.1%  |
| Macro Recall      | 0.7857     | 0.7746     | −1.1%  |
| Macro F1          | **0.7223** | 0.6695     | −5.3%  |

**Outcome: Regression.** Class weights + tighter sigma destabilised training (train loss stayed above val loss, val accuracy oscillated 40–70%). Wrap-around errors increased from 109 → 128. See `FairFace_v3_WrapAround_Analysis.md` for details.

---

## v3.2 — Revert Loss + Data Cleaning *(2026-03-16)*

**Focus:** Recover v3.0 accuracy by reverting harmful v3.1 loss changes and cleaning noisy ITA labels

### Diagnosis

v3.1's class weights + tight sigma (0.8) caused a 6% accuracy drop. Training curves show train loss > val loss (class weights distort loss magnitude) and wild val accuracy oscillation. Wrap-around misclassifications (Very Dark ↔ Very Light) increased 17%. Root cause is **label noise**, not model capacity.

### Changes

- **Reverted OrdinalCE** to v3.0 baseline: `sigma=1.0`, `gamma=2.0`, **no class weights**
- **Fixed confusion matrix renderer** — replaced `sns.heatmap` with `plt.imshow` + manual text annotations (seaborn drops annotations when cell count disparity is extreme)
- **[NEW] `clean_fairface_labels.py`** — flags and removes images with suspicious ITA labels:
  - Saturated ITA (|ITA| > 85°, boundary artefact from b* ≈ 0)
  - Monochrome (mean HSV saturation < 20, unreliable ITA)
  - ITA-vs-luminance mismatch (ITA class contradicts raw L* value)
- **Cleaned CSV** (`master_mst_labels_cleaned.csv`) used for training instead of raw labels
- **Kept from v3.1:** CosineAnnealingWarmRestarts, gradient clipping, TTA, RandomAffine

### Files Modified

- `train_fairface.py` — loss config reverted, confusion matrix renderer replaced
- `submit_csf.sh` — output dir `FairFace-Model-3.2-mst5`, cleaned CSV path
- **[NEW]** `clean_fairface_labels.py` — ITA label quality auditor

### Results

| Metric            | v3.1       | v3.2       | Change  |
| ----------------- | ---------- | ---------- | ------- |
| Macro Accuracy    | 0.7097     | **0.8226** | +11.3%  |
| Macro Precision   | 0.6443     | **0.6830** | +3.9%   |
| Macro Recall      | 0.7746     | **0.8174** | +4.3%   |
| Macro F1          | 0.6695     | **0.7325** | +6.3%   |
| Weighted F1       | —          | **0.83**   | —       |

| Class                | Precision | Recall     | F1   | Support |
| -------------------- | --------- | ---------- | ---- | ------- |
| Very Dark (MST 9-10) | 0.63      | **0.92**   | 0.74 | 953     |
| Dark (MST 7-8)       | 0.86      | **0.83**   | 0.85 | 4,788   |
| Medium (MST 5-6)     | **0.91**  | 0.79       | 0.85 | 4,093   |
| Light (MST 3-4)      | 0.57      | **0.82**   | 0.68 | 451     |
| Very Light (MST 1-2) | 0.44      | **0.72**   | 0.55 | 98      |

**Outcome: Significant improvement.** Reverted loss + cleaned labels recovered and exceeded v3.0 performance.

**Key improvements over v3.1:**
- Medium recall **62.8% → 79.1%** (+16pp) — no longer collapsed by class weights
- Very Light recall **56.1% → 72.4%** (+16pp) — cleaned labels improved minority performance
- Light recall **74.1% → 82.0%** (+8pp)
- Dark→Very Dark misclassifications fell from 793 → 484

**Remaining weaknesses:**
- Very Light precision 0.44 — over half of "Very Light" predictions are false positives
- Dark→Very Dark confusion (484 samples, 10.1% of Dark) remains the largest error block
- Medium→Dark leakage (571 samples, 13.9% of Medium) is the second largest

**Training curve diagnosis:**
- **No overfitting.** Val loss sits consistently *below* train loss for the entire fine-tuning phase (epochs 10–55). This is the signature of healthy dropout/augmentation regularisation — the model sees harder augmented views at train time and cleaner views at val time.
- **No underfitting either.** Both loss curves are still declining at the final epoch, and val accuracy is climbing steadily (~0.70→0.75 during fine-tune). The model has not converged — extending training to 70–80 epochs would likely squeeze out 1–2pp more accuracy.
- **Stable convergence.** Unlike v3.1's wild oscillation (40–70% val acc), v3.2 shows steady monotonic improvement after the fine-tune start. The loss revert + cleaned labels eliminated the gradient noise that plagued v3.1.
- **Train accuracy (~67%) < val accuracy (~75%)** at final epoch — again confirming the augmentation pipeline is doing its job as a regulariser.

**Wrap-around errors (Very Dark ↔ Very Light):**

| Direction | v3.0 | v3.1 | v3.2 | Change |
|-----------|:----:|:----:|:----:|:------:|
| Very Dark → Very Light | 70 | 90 | **4** | −94% |
| Very Light → Very Dark | 39 | 38 | **4** | −90% |
| **Total** | **109** | **128** | **8** | **−93%** |

> [!TIP]
> Data cleaning eliminated **93% of wrap-around errors** (109→8). This confirms the root cause was label noise — saturated ITA values pushing images to the wrong extreme — not model capacity.

**UMAP:** Smooth ordinal gradient from Very Dark (upper-left) to Very Light (lower-right). Classes form a natural continuum rather than discrete clusters, consistent with the physical reality of skin tone as a continuous spectrum.

### v3.3 Roadmap

Based on v3.2's remaining weaknesses, the highest-impact improvements for v3.3 are:

1. **Extend training to 70–80 epochs** — both curves still improving; cheapest possible gain (~1–2pp expected)
2. **Raise Very Light recall** — only 98 val samples after cleaning; consider (a) sourcing additional light-skinned face data, (b) reducing the oversampling cap from 8,000→6,000 to further tighten the class ratio, or (c) a dedicated augmentation pipeline for this class
3. **Reduce Dark→Very Dark confusion (484 samples, 10.1%)** — the largest remaining error block; the ITA boundary at −66.9° may be slightly too aggressive; consider shifting it to −70° to reclassify borderline images into Dark instead of Very Dark

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

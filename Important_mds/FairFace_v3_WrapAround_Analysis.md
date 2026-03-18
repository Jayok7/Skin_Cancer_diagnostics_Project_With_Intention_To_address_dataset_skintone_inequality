# FairFace v3.0 — "Wrap-Around" Misclassification Analysis

**Date:** 2026-03-09
**Notebook:** `FairFace_v3_Extreme_Misclassifications.ipynb`

---

## Overview

The v3.0 confusion matrix reveals an unusual pattern where the model confuses the two *most distant* classes:

| Direction | Count | % of True Class |
|-----------|-------|----------------|
| Very Dark (MST 9-10) → predicted Very Light (MST 1-2) | 70 | 5.5% of 1,263 |
| Very Light (MST 1-2) → predicted Very Dark (MST 9-10) | 39 | 11.4% of 341 |
| **Total** | **109** | **1.0% of 10,954 val samples** |

---

## Root Causes Identified (Visual Inspection)

### 1. Monochrome / Colour-Shifted Images

Many misclassified images have strong blue, green, or sepia colour casts. These monochrome artefacts confuse both the ITA labelling pipeline (which computes skin tone from pixel colour) and the model (which relies on colour features).

- **Very Light → Very Dark**: Images with heavy blue/green tints appear "dark" to the model despite the subject having light skin
- **Very Dark → Very Light**: Overexposed or washed-out images make dark skin appear light

### 2. Cropping Issues

Several images are extreme close-ups of skin patches, partial faces, or off-centre crops where the visible skin area is minimal. The ITA label was computed from a centre-crop that may capture hair, background, or clothing rather than representative skin.

### 3. Inappropriate Labels (ITA Noise)

Some images are clearly mislabelled by the ITA pipeline:
- Light-skinned subjects labelled as "Very Dark" (ITA values around -85° to -90°)
- Dark-skinned subjects labelled as "Very Light" (ITA values around +85° to +90°)

These saturated ITA values (near ±90°) indicate the ITA computation hit boundary conditions — likely from non-skin pixels dominating the centre crop.

### 4. Non-Standard Subjects

A small number of images contain:
- Infants/babies (skin tone appearance differs significantly)
- Heavy accessories (sunglasses, hats) obscuring skin
- Artistic/filtered photos with unnatural colour grading

---

## Why This Is NOT a Mathematical Wraparound

The ordinal cross-entropy loss penalises far-off errors more during training (via Gaussian soft labels), but the model's **softmax output layer has no ordinal constraint at inference**. The 5 logits are independent — the model has no architectural notion that class 0 (Very Dark) is "far" from class 4 (Very Light). On ambiguous or noisy images, any logit can become the argmax.

This contrasts with true ordinal regression (cumulative logits), which would force monotone decision boundaries and make extreme-to-extreme jumps structurally impossible.

---

## Confidence Patterns

The model is generally **moderately confident** in these wrong predictions (30–65% confidence), not overwhelmingly certain. This suggests the images are genuinely ambiguous to the model rather than strongly activating the wrong class features.

---

## Implications for v3.1

These 109 samples (1.0%) represent a noise floor that is unlikely to be eliminated through hyperparameter tuning alone. The v3.1 optimisations (tighter sigma, class weights, gradient clipping) may slightly reduce the count, but systematic fixes would require:

1. **Data cleaning** — remove or relabel images with clear colour artefacts or wrong ITA labels
2. **ITA robustness** — improve the ITA computation to exclude non-skin pixels (e.g., skin segmentation before ITA)
3. **Ordinal regression head** — replace softmax with cumulative logits to structurally prevent extreme-to-extreme jumps

---

## v3.1 Results: Did It Help?

**No — it made things worse.** The v3.1 model (tighter sigma=0.8, inverse-frequency class weights) increased the wrap-around count from 109 → 128:

| Direction | v3.0 | v3.1 | Change |
|-----------|:----:|:----:|:------:|
| Very Dark → Very Light | 70 | 90 | +29% ↑ |
| Very Light → Very Dark | 39 | 38 | −3% ≈ |
| **Total** | **109** | **128** | **+17% ↑** |

**Why?** The class weights told the model to pay *more* attention to minority classes (Light, Very Light). But because many of these extreme misclassifications stem from **noisy labels** (images that genuinely look ambiguous), amplifying the gradient signal on noisy samples made the model *more* confused, not less. The tighter sigma (0.8) also reduced the ordinal "cushion" between adjacent classes, making the model more aggressive in jumping to extreme predictions.

> [!IMPORTANT]
> **Lesson learned:** Class weights are counterproductive when label noise is the bottleneck. The 109 wrap-around errors in v3.0 are best addressed through data cleaning, not loss function engineering.

### v3.1 Training Curve Evidence

The v3.1 training curves confirm the distortion:

- **Train loss stayed ABOVE val loss** for the entire fine-tuning stage — class weights inflated the training loss artificially on majority classes (Dark, Medium), making the metric misleading
- **Val accuracy oscillated 40–70%** with no convergence trend — the model never found a stable solution because the weighted gradients kept pulling it in different directions each epoch
- **Best checkpoint selected at ~70% val acc** — but this was fragile, and the model's predictions on extreme classes were unreliable

This pattern is the exact signature of an overly aggressive loss reweighting strategy applied to noisy labels.

---

## Recommended Next Steps

### Short-Term (v3.2) — ✅ Implemented
- ✅ **Reverted loss to v3.0 baseline** (sigma=1.0, no class weights)
- ✅ **Kept** CosineAnnealingWarmRestarts and gradient clipping
- ✅ **Fixed confusion matrix** (matplotlib `imshow` replaces seaborn `heatmap`)
- ✅ **[NEW] `clean_fairface_labels.py`** — flags saturated ITA, monochrome, ITA/L* mismatch images and removes them from training

### Medium-Term (Data Quality)
- **Label audit:** Flag and review images with ITA values near ±90° (boundary conditions)
- **Colour normalisation:** Apply white-balance correction before ITA computation to reduce monochrome artefact influence
- **Skin segmentation:** Use a face/skin mask before computing ITA to exclude hair, background, and accessories

### Long-Term (Architecture)
- **Ordinal regression head:** Replace the 5-way softmax with cumulative logits (CORAL / CORN framework) to structurally enforce ordinality at inference — making Very Dark → Very Light jumps architecturally impossible
- **Post-hoc constraint:** Reject predictions that skip more than 2 classes from the second-highest logit



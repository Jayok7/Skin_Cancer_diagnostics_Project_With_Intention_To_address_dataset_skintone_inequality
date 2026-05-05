
# Equitable Skin Cancer Diagnostics

> An pipeline for **equitable skin cancer classification** that addresses the under-representation of darker skin tones in dermatological AI. The project spans skin-tone classification, dataset augmentation, diagnostic model training, Vision-Language Model (VLM) clinical reasoning, and a clinical decision-support GUI.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline Stages](#pipeline-stages)
   - [Stage 1 — Skin Tone Classification](#stage-1--skin-tone-classification)
   - [Stage 2 — Skin Tone Augmentation](#stage-2--skin-tone-augmentation)
   - [Stage 3 — Diagnostic Model Training](#stage-3--diagnostic-model-training)
   - [Stage 4 — VLM Clinical Reasoning](#stage-4--vlm-clinical-reasoning)
   - [Stage 5 — Clinical GUI](#stage-5--clinical-gui)
   - [Stage 6 — External Validation](#stage-6--external-validation)
4. [Experimental &amp; Legacy Scripts](#experimental--legacy-scripts)
5. [Slurm / HPC Scripts](#slurm--hpc-scripts)
6. [Datasets](#datasets)
7. [Installation &amp; Requirements](#installation--requirements)
8. [Hardware Requirements](#hardware-requirements)
9. [Reproducibility](#reproducibility)
10. [Citation](#citation)
11. [License](#license)

---

## Overview

Dermatological AI models are overwhelmingly trained on light-skinned populations, leading to disparate performance on darker skin tones. This project builds a **fairness-aware pipeline** that:

1. **Classifies skin tone** on dermoscopic images using a lesion-aware cascade (ITA analytical + EfficientNet-B4 CNN).
2. **Augments under-represented tones** via Reinhard LAB colour transfer, Poisson deep blending, U-Net lesion-aware transplantation, and neural style transfer.
3. **Trains diagnostic models** (EfficientNet-B3) on balanced datasets (HAM10000 & ISIC 2019).
4. **Fine-tunes a Vision-Language Model** (Qwen2.5-VL-7B) with SFT + DPO to produce grounded clinical reasoning from both the image and its Grad-CAM heatmap.
5. **Deploys a clinical GUI** (Flask + HTML/CSS/JS) with real-time Grad-CAM explainability and VLM diagnostic reasoning.

---

## Repository Structure

```
.
├── skin-tone-classifier/          # Stage 1: Skin tone classification
│   ├── train_fairface.py          #   EfficientNet-B4 trainer (FairFace + MSKCC fine-tune)
│   ├── classify_skin_tone.py      #   Lesion-aware A+B+C cascade inference pipeline
│   ├── evaluate_predictions_vs_mst_raters.py  # Eval against MSKCC human raters
│   ├── fairface_dataset.py        #   PyTorch Dataset for FairFace (class-aware augmentation)
│   └── mskcc_dataset.py           #   PyTorch Dataset for MSKCC (3-tier augmentation)
│
├── skin-tone-augmentation/        # Stage 2: Dataset balancing
│   ├── skin_tone_augmentation.py  #   V1: Reinhard CIE-LAB + Poisson deep blending (λ-controlled)
│   ├── skin_tone_augmentation_v2.py #  V2: U-Net lesion-aware transplant + Poisson blending
│   ├── skin_tone_transfer.py      #   Reinhard-only colour transfer (standalone)
│   └── submit_skin_augmentation_v2_*.sh  # Slurm scripts for λ sweep (0.0, 0.3, 0.7, 1.0)
│
├── diagnostics/                   # Stage 3: Diagnostic model training & evaluation
│   ├── skin_cancer_diagnostics.py #   HAM10000 EfficientNet-B3 trainer + Grad-CAM
│   ├── skin_cancer_diagnostics_isic2019.py  # ISIC 2019 EfficientNet-B3 trainer
│   ├── evaluate_isic2019_test.py  #   Multi-model test-set evaluation framework
│   └── *.ipynb                    #   Exploratory notebooks
│
├── vlm-finetune/                  # Stage 4: Vision-Language Model fine-tuning
│   ├── generate_cnn_outputs.py    #   Generate CNN predictions + Grad-CAM for teacher input
│   ├── teacher_generation.py      #   Teacher reasoning via OpenAI GPT-4o API
│   ├── teacher_generation_free.py #   Teacher reasoning via local Qwen2.5-VL-7B-Instruct
│   ├── quality_control.py         #   5-stage QC filter (keywords, spatial, safety, length)
│   ├── qc_reference.py            #   Earlier 3-stage QC reference implementation
│   ├── format_dataset.py          #   Convert cleaned data → chat-template JSONL
│   ├── generate_dpo_pairs.py      #   Build preference pairs for DPO training
│   ├── train_sft.py               #   QLoRA supervised fine-tuning
│   ├── train_dpo.py               #   DPO alignment training
│   ├── evaluate.py                #   SFT vs DPO evaluation across quality dimensions
│   └── summary.md                 #   VLM pipeline overview & cost estimates
│
├── gui/                           # Stage 5: Clinical decision-support interface
│   ├── app.py                     #   Flask backend (CNN + Grad-CAM + VLM inference)
│   ├── index.html                 #   Frontend markup
│   ├── style.css                  #   Premium UI styling (glassmorphism, dark mode)
│   ├── script.js                  #   Frontend logic (upload, display, API calls)
│   ├── requirements.txt           #   GUI-specific dependencies
│   ├── CNNs/                      #   Diagnostic model checkpoints directory
│   └── checkpoints/               #   VLM checkpoint directory
│
├── External-validation/           # Stage 6: Empirical validation battery
│   ├── ISIC2019_Internal_baseline/  # Internal baseline tone-stratified evaluation
│   │   ├── evaluate_tone_stratified.py  # Tests 2–5 on ISIC 2019 test set
│   │   └── submit_tone_stratified_eval_novig.sh  # Vignette-sensitivity Slurm script
│   ├── Fitzpatrick17k/              # External validation on Fitzpatrick17k
│   │   ├── generate_fitzpatrick_predictions.py  # Stage 1: GPU inference per model
│   │   ├── evaluate_fitzpatrick_stratified.py   # Stage 2: Tests 2–5 evaluation
│   │   └── submit_fitzpatrick_stratified.sh     # Slurm script (both mapping arms)
│   ├── MILK10K/                     # External validation on MILK10K
│   │   ├── generate_milk10k_predictions.py  # Stage 1: GPU inference per model
│   │   ├── evaluate_milk10k_stratified.py   # Stage 2: Tests 2–5 evaluation
│   │   └── submit_milk10k_stratified.sh     # Slurm script (both mapping arms)
│   └── Tests.md                     # Full test battery specification (Tests 1–6)
│
├── csf/                           # Slurm job scripts for CSF3 HPC cluster (20 scripts)
├── neural_style_transfer.py       # Experimental: VGG-19 neural style transfer augmentation
├── requirements.txt               # Root-level Python dependencies
├── .gitignore                     # Excludes datasets, models, outputs, venvs
└── README.md                      # Original README
```

> **Note:** Several experimental/legacy scripts exist in the project root (see [Experimental &amp; Legacy Scripts](#experimental--legacy-scripts)). These are not part of the core pipeline but are retained for reproducibility of exploratory work.

---

## Pipeline Stages

### Stage 1 — Skin Tone Classification

**Goal:** Assign a Monk Skin Tone (MST) label to every dermoscopic image so the dataset can be stratified and balanced.

#### Training (`train_fairface.py`)

| Detail                      | Value                                                                |
| --------------------------- | -------------------------------------------------------------------- |
| **Backbone**          | EfficientNet-B4 (ImageNet pretrained)                                |
| **Training data**     | FairFace (86k face images with L\*-based labels)                     |
| **Fine-tune data**    | MSKCC perilesional crops (680 images, MST-rated)                     |
| **Loss**              | Ordinal Cross-Entropy (penalises far-off predictions more heavily)   |
| **Class balancing**   | `WeightedRandomSampler` + 3-tier class-aware augmentation          |
| **Outputs**           | 3-way (Dark / Medium / Light) or 5-way (MST-5) checkpoint            |
| **Key design choice** | Ordinal loss chosen over standard CE because MST is an ordered scale |

**Dataset helpers:**

- `fairface_dataset.py` — PyTorch Dataset with class-aware augmentation (minority classes get stronger transforms).
- `mskcc_dataset.py` — PyTorch Dataset with 3-tier augmentation strategy (standard → minority → severe minority) and √-balanced sampling to prevent memorisation.

#### Inference (`classify_skin_tone.py`)

A **lesion-aware A+B+C cascade** that avoids contamination from the lesion and dermoscopic vignette:

| Approach    | Method                                                           | When Used                  |
| ----------- | ---------------------------------------------------------------- | -------------------------- |
| **A** | Perilesional ring (U-Net segmentation → dilate → ring mask)    | Default; requires U-Net    |
| **B** | Multi-patch consensus (corner/edge patches, IQR outlier removal) | Fallback if ring too small |
| **C** | Full-image (vignette-cropped, median-painted border)             | Last resort                |

All approaches are **vignette-aware**: a black-border mask is computed once and propagated through every sampling stage.

Supports both **ITA analytical** (L\*a\*b\* → ITA angle → MST bin) and **CNN** (EfficientNet-B4 softmax) modes.

#### Evaluation (`evaluate_predictions_vs_mst_raters.py`)

Evaluates CNN predictions against MSKCC human raters (Rater 1 & Rater 2), computing:

- Exact accuracy and relaxed accuracy (±1 class)
- Per-method and per-confidence breakdowns
- Confusion matrices and side-by-side MST count bar charts

---

### Stage 2 — Skin Tone Augmentation

**Goal:** Balance the skin-tone distribution so diagnostic models train on equitable data.

Three complementary augmentation methods are provided:

#### V1: Reinhard + Deep Blending (`skin_tone_augmentation.py`)

A **λ-controlled hybrid** pipeline:

- **λ = 1.0** → 100% Reinhard CIE-LAB colour transfer (fast, preserves spatial structure)
- **λ = 0.0** → 100% Poisson deep blending (`cv2.seamlessClone`, preserves lesion gradients)
- **λ = 0.7** (default) → 70/30 mix

Includes a **quality gate** that rejects unrealistic augmentations by checking:

- Skin hue plausibility (HSV border + centre analysis)
- Saturation bounds (rejects neon/oversaturated)
- LAB a\*/b\* range (rejects colour casts)
- Colour variance (rejects patchy multi-colour artifacts)

Uses **iterative generate→filter→retry cycles** (up to 4) with 1.5× oversampling to compensate for QC rejections. Tracks λ drift per class and warns if actual ratio deviates >5%.

#### V2: U-Net Lesion-Aware Transplant (`skin_tone_augmentation_v2.py`)

The refined pipeline for the final experiments:

- U-Net segments the lesion from a source image
- The lesion is transplanted onto a different-skin-tone reference image using Poisson blending
- Preserves lesion morphology (ABCD features) while adopting the reference skin colour/texture
- Slurm scripts provided for λ sweep: `submit_skin_augmentation_v2_{00,03,07,10}.sh`

#### Standalone Reinhard Transfer (`skin_tone_transfer.py`)

A lightweight Reinhard-only tool for quick colour transfer:

- Computes per-class LAB statistics from a predictions CSV
- Supports manual transfer (source class → target class), auto-balance, and preview mode
- Multi-reference averaging (samples N references to reduce per-image noise)

#### Neural Style Transfer (`neural_style_transfer.py`) *(Experimental)*

VGG-19 Gatys et al. style transfer applied to dermoscopic images:

- Transfers skin texture/colour from dark-skin references onto light-skin lesion images
- Uses LBFGS optimisation (300 steps default, α=1 content, β=1e6 style)
- Optional CSV filtering to enforce Light/Medium → Dark direction
- **Status:** Experimental. Not used in the final pipeline due to higher computational cost and less controllable output compared to Reinhard/Poisson methods.

---

### Stage 3 — Diagnostic Model Training

**Goal:** Train a skin cancer classifier and evaluate fairness across skin tones.

#### HAM10000 (`diagnostics/skin_cancer_diagnostics.py`)

| Detail             | Value                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| **Model**    | EfficientNet-B3 (ImageNet pretrained)                                                                    |
| **Dataset**  | HAM10000 (7 diagnostic categories)                                                                       |
| **Head**     | BatchNorm → Dense(256) → ReLU → Dropout(0.4) → Dense(7)                                              |
| **Training** | 2-stage: 20 epochs head-only (lr=1e-3) + 20 epochs full fine-tune (lr=1e-5)                              |
| **Loss**     | CrossEntropy with label smoothing (0.1)                                                                  |
| **Outputs**  | Training curves, confusion matrix, classification report, confidence statistics, Grad-CAM visualisations |

#### ISIC 2019 (`diagnostics/skin_cancer_diagnostics_isic2019.py`)

Same architecture adapted for the larger ISIC 2019 dataset (8 diagnostic categories, 25k+ images). Includes augmented-data integration for fairness experiments.

#### Multi-Model Evaluation (`diagnostics/evaluate_isic2019_test.py`)

Comprehensive evaluation framework that:

- Compares multiple model checkpoints on the ISIC 2019 test set
- Generates per-class metrics, confidence statistics, and comparison reports
- Produces publication-ready plots and LaTeX-compatible tables

---

### Stage 4 — VLM Clinical Reasoning

**Goal:** Fine-tune Qwen2.5-VL-7B to produce grounded clinical reasoning that explains *why* the CNN made its prediction, anchored to both the dermoscopic image and the Grad-CAM heatmap.

#### Pipeline Flow

```
CNN Predictions + Grad-CAM  ──►  Teacher Generation  ──►  Quality Control
       │                              │                         │
  generate_cnn_outputs.py    teacher_generation*.py      quality_control.py
                                                                │
                                                          ▼
                                                    format_dataset.py
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                     train_sft.py   generate_dpo_pairs.py
                                          │               │
                                          ▼               ▼
                                    SFT checkpoint   train_dpo.py
                                                          │
                                                          ▼
                                                   DPO checkpoint
                                                          │
                                                     evaluate.py
```

#### Step 1: Generate CNN Outputs (`generate_cnn_outputs.py`)

Runs the diagnostic CNN on all images, saving predictions, confidence scores, Grad-CAM heatmaps, and spatial activation regions.

#### Step 2: Teacher Generation

Two approaches provided:

- **`teacher_generation.py`** — Uses OpenAI GPT-4o API. Higher quality but costs ~$50–75 for 2,500 samples.
- **`teacher_generation_free.py`** *(recommended)* — Uses local Qwen2.5-VL-7B-Instruct. Free, stylistically consistent with the student model, Apache 2.0 licensed.

Both generate structured clinical reasoning that references specific visual features and their spatial locations relative to the Grad-CAM activation pattern.

#### Step 3: Quality Control (`quality_control.py`)

A **5-stage filtering pipeline** (expected discard rate: 20–30%):

| Stage | Check                | Purpose                                                                       |
| ----- | -------------------- | ----------------------------------------------------------------------------- |
| 1     | Keyword constraint   | Catches gross factual errors (e.g., melanoma prediction described as angioma) |
| 2     | Indeterminate filter | Discards hedged non-answers ("cannot be determined")                          |
| 3     | Spatial alignment    | Verifies text spatial references match Grad-CAM activation regions            |
| 4     | Length bounds        | Ensures 60–200 words per response                                            |
| 5     | Safety language      | Requires hedging phrases, rejects definitive diagnostic claims                |

An earlier **3-stage reference implementation** is preserved in `qc_reference.py`.

#### Step 4: Format Dataset (`format_dataset.py`)

Converts cleaned training data into chat-template JSONL for VLM fine-tuning, with multi-image input (original + Grad-CAM overlay).

#### Step 5: Training

| Phase         | Script           | Method                             | Key Config                                                            |
| ------------- | ---------------- | ---------------------------------- | --------------------------------------------------------------------- |
| **SFT** | `train_sft.py` | QLoRA (4-bit NF4, rank 64, α 128) | Token masking on assistant responses only; bf16 mixed precision       |
| **DPO** | `train_dpo.py` | Direct Preference Optimisation     | β=0.1; preference pairs from QC scores via `generate_dpo_pairs.py` |

#### Step 6: Evaluation (`evaluate.py`)

Compares SFT and DPO checkpoints across diagnostic quality dimensions using automated metrics and side-by-side analysis.

---

### Stage 5 — Clinical GUI

**Goal:** A clinician-facing web interface for real-time skin lesion analysis with explainable AI.

#### Architecture

| Component            | Technology                                                                     |
| -------------------- | ------------------------------------------------------------------------------ |
| **Backend**    | Flask (`app.py`) — loads CNN models, generates Grad-CAM, runs VLM inference |
| **Frontend**   | HTML (`index.html`) + CSS (`style.css`) + JS (`script.js`)               |
| **CNN Models** | Loaded from `gui/CNNs/` directory (multiple checkpoint support)              |
| **VLM**        | Qwen2.5-VL-7B loaded from `gui/checkpoints/` (DPO checkpoint)                |

#### Features

- **Multi-model support:** Select from available diagnostic checkpoints via dropdown
- **Three-panel results view:** Original image, Grad-CAM heatmap, diagnostic reasoning
- **Classification scores:** All class probabilities displayed with confidence bars
- **Dark mode / light mode:** Theme toggle with premium glassmorphism styling
- **Drag-and-drop upload:** Supported image formats: PNG, JPG, JPEG
- **Medical disclaimer:** Prominent clinical decision-support disclaimer

#### Running the GUI

```bash
cd gui
pip install -r requirements.txt
python app.py
# Open http://localhost:5000 in your browser
```

> **VLM Setup:** Set `os.environ["HF_HOME"]` in `app.py` to a valid local directory for model caching. Place the DPO checkpoint in `gui/checkpoints/`. The VLM requires ~16 GB VRAM (GPU) or will fall back to CPU with degraded performance.

---

### Stage 6 — External Validation

**Goal:** Rigorously evaluate whether the augmentation pipeline's fairness improvements generalise beyond the ISIC 2019 training distribution, using independent external datasets with ground-truth skin tone labels.

The validation battery implements **Tests 2–5** from the empirical-rigour specification (`Tests.md`), applied consistently across three evaluation arms:

#### Evaluation Arms

| Arm                         | Dataset                                       | Tone Labels                                       | Purpose                                                                             |
| --------------------------- | --------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Internal baseline** | ISIC 2019 test set                            | CNN-predicted MST (3-class)                       | Confirms augmentation effects on held-out data from the training distribution       |
| **Fitzpatrick17k**    | Fitzpatrick17k (clinical photos, 16k+ images) | Fitzpatrick Scale I–VI (human-rated, two raters) | External generalisability on a dataset with dermatologist-assigned skin type labels |
| **MILK10K**           | MILK10K (Harvard, 10k+ images)                | 6-level skin tone class (inverted scale)          | Second independent external dataset with diverse imaging sources                    |

#### Two-Stage Architecture

Each arm follows the same two-stage pattern:

**Stage 1 — Prediction Generation (GPU):** Run all five EfficientNet-B3 model variants (baseline + four λ-augmented) on the target dataset. Outputs per-model prediction CSVs with full softmax probabilities.

- `generate_fitzpatrick_predictions.py` — Fitzpatrick17k predictions with HIERARCHY or STRICT label-mapping arms
- `generate_milk10k_predictions.py` — MILK10K predictions with DIAGNOSIS3 or SIMPLIFIED label-mapping arms
- ISIC 2019 internal baseline reuses existing predictions from Stage 3

**Stage 2 — Stratified Evaluation (CPU):** Consume prediction CSVs and run the full test battery.

#### Test Battery

| Test         | Name                                   | Method                                                                                                         | Purpose                                                                                           |
| ------------ | -------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **2**  | Tone-stratified diagnostics            | Per-model × per-tone-tertile: macro F1, per-class F1, per-class recall, macro AUC                             | Headline fairness result — the 5 × 3 grid of performance across skin tones                      |
| **3**  | Bootstrap confidence intervals         | 1,000-iteration resampling → 95% CI on macro F1                                                               | Quantifies uncertainty on small subsets (especially Dark); error bars for claims                  |
| **4**  | McNemar's paired test                  | 2 × 2 contingency table per (baseline vs λ=k) pair, per tone                                                 | Tests whether augmentation shifts*which* images are correctly classified, not just reshuffling  |
| **5a** | Class composition audit                | Diagnostic-class distribution by tone tertile                                                                  | Detects confound: if Dark subset is 80% naevus, class-specific gains masquerade as fairness gains |
| **5b** | Source/site confounder                 | Image source distribution by tone tertile                                                                      | Detects imaging-protocol confound (e.g., Dark images clustering from one clinical site)           |
| **5c** | Confidence/rater-agreement sensitivity | Re-run Tests 2–3 at stricter tone confidence ≥ 0.8 (ISIC 2019) or on rater-agreement subset (Fitzpatrick17k) | Controls for tone-label noise                                                                     |
| **5d** | Label-mapping sensitivity              | Re-run with alternate diagnosis → ISIC-class mapping                                                          | Tests robustness to taxonomy choices                                                              |

#### Key Design Choices

- **Dual-mapping arms:** Both Fitzpatrick17k (HIERARCHY vs STRICT) and MILK10K (DIAGNOSIS3 vs SIMPLIFIED) run two independent label-mapping strategies, testing whether conclusions survive taxonomy differences.
- **Vignette sensitivity (ISIC 2019):** A `--require-no-vignette` flag excludes images with dermoscopic vignette artifacts, controlling for the site/equipment confound.
- **McNemar's paired test:** Provides a stronger claim than unpaired bootstrap CIs — it tests on the *same images* whether model A and model B disagree systematically.
- **Either outcome is defensible:** If CIs overlap, the project argues that existing test data cannot adequately evaluate fairness — itself a central thesis.

---

## Experimental & Legacy Scripts

The following scripts in the project root are **not part of the core pipeline** but are retained for reproducibility:

| Script                          | Purpose                                                 |
| ------------------------------- | ------------------------------------------------------- |
| `neural_style_transfer.py`    | VGG-19 Gatys-style transfer (experimental augmentation) |
| `classify_isic2020.py`        | ISIC 2020 skin tone classification (earlier dataset)    |
| `label_skin_tone.py`          | Legacy labelling script (3-class)                       |
| `label_isic2018.py`           | ISIC 2018 labelling utility                             |
| `diagnose_dark_medium.py`     | Diagnostic analysis by skin tone subgroup               |
| `diagnose_mskcc_finetune.py`  | MSKCC fine-tuning diagnostics                           |
| `evaluate_mskcc_cnn.py`       | MSKCC CNN evaluation                                    |
| `extract_embeddings.py`       | Feature embedding extraction                            |
| `visualise_latent_space.py`   | t-SNE/UMAP latent space plots                           |
| `prepare_mskcc_labels.py`     | MSKCC label preparation                                 |
| `preprocess_mskcc_crops.py`   | MSKCC perilesional crop preprocessing                   |
| `train_fitzpatrick.py`        | Legacy Fitzpatrick classifier trainer                   |
| `style_transfer_test.py`      | Neural style transfer experiments                       |
| `stylegan_augmentation.py`    | StyleGAN-based augmentation (prototype)                 |
| `MSKCC_Advanced_Eval_Cell.py` | Advanced evaluation utilities                           |
| `analyze_dataset.py`          | Dataset statistics                                      |

**Notebooks** (various `.ipynb` files) contain exploratory analysis, visualisation, and prototyping work.

---

## Slurm / HPC Scripts

All Slurm job submission scripts are in `csf/`, targeting the University of Manchester CSF3 cluster:

| Script                             | Pipeline Stage                       |
| ---------------------------------- | ------------------------------------ |
| `submit_classify_isic2020.sh`    | Skin tone classification (ISIC 2020) |
| `submit_classify_skin_tone.sh`   | Skin tone classification (general)   |
| `submit_finetune_unet.sh`        | U-Net fine-tuning on ISIC 2018       |
| `submit_mskcc_finetune.sh`       | EfficientNet-B4 fine-tuning on MSKCC |
| `submit_skin_augmentation.sh`    | Skin tone augmentation               |
| `submit_diagnostics_isic2019.sh` | ISIC 2019 diagnostic training        |
| `submit_skin_diagnostics.sh`     | HAM10000 diagnostic training         |
| `submit_eval_isic2019_test.sh`   | ISIC 2019 test evaluation            |
| `submit_generate_cnn_outputs.sh` | CNN output generation for VLM        |
| `submit_teacher_generation.sh`   | VLM teacher generation               |
| `submit_vlm_qc_format.sh`        | QC + dataset formatting              |
| `submit_vlm_sft.sh`              | VLM supervised fine-tuning           |
| `submit_vlm_dpo.sh`              | VLM DPO training                     |
| `submit_vlm_evaluate.sh`         | VLM evaluation                       |
| `submit_label_isic2018.sh`       | ISIC 2018 labelling                  |
| `submit_label_isic2019.sh`       | ISIC 2019 labelling                  |
| `submit_label_mskcc.sh`          | MSKCC labelling                      |
| `submit_utkface_labels.sh`       | UTKFace labelling                    |
| `submit_mskcc_diagnostic.sh`     | MSKCC diagnostic analysis            |
| `submit_gui.sh`                  | GUI deployment                       |

Additional Slurm scripts within `External-validation/`:

| Script                                                              | Pipeline Stage                                     |
| ------------------------------------------------------------------- | -------------------------------------------------- |
| `Fitzpatrick17k/submit_fitzpatrick_stratified.sh`                 | Fitzpatrick17k prediction + evaluation (both arms) |
| `MILK10K/submit_milk10k_stratified.sh`                            | MILK10K prediction + evaluation (both arms)        |
| `ISIC2019_Internal_baseline/submit_tone_stratified_eval_novig.sh` | ISIC 2019 vignette-sensitivity evaluation          |

---

## Datasets

All raw datasets are **excluded from the repository** via `.gitignore`. Download them from their respective sources:

| Dataset                  | Source                                                  | Used For                                            |
| ------------------------ | ------------------------------------------------------- | --------------------------------------------------- |
| **FairFace**       | [GitHub](https://github.com/joojs/fairface)                | Skin tone classifier training (86k faces)           |
| **UTKFace**        | [UTKFace](https://susanqq.github.io/UTKFace/)              | Supplementary skin tone labels                      |
| **MSKCC**          | Memorial Sloan Kettering                                | Fine-tuning + ground truth (MST-rated)              |
| **HAM10000**       | [ISIC Archive](https://www.isic-archive.com/)              | Diagnostic model training (7 classes)               |
| **ISIC 2019**      | [ISIC Challenge](https://challenge.isic-archive.com/data/) | Diagnostic model training (8 classes, 25k+ images)  |
| **ISIC 2018**      | [ISIC Challenge](https://challenge.isic-archive.com/data/) | U-Net segmentation fine-tuning                      |
| **Fitzpatrick17k** | [GitHub](https://github.com/mattgroh/fitzpatrick17k)       | External validation (FST-stratified, dual-rater)    |
| **MILK10K**        | Harvard Dataverse                                       | External validation (6-level tone, diverse sources) |

---

## Installation & Requirements

### Root Environment (training & augmentation)

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

**Key dependencies:**

- `torch >= 2.0`
- `torchvision`
- `transformers >= 4.40`
- `peft >= 0.10`
- `accelerate`
- `opencv-python`
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- `trl` (for DPO training)

### GUI Environment

```bash
cd gui
pip install -r requirements.txt
```

**Additional GUI dependencies:**

- `flask`
- `tensorflow` (for legacy model support)
- `qwen-vl-utils`

---

## Hardware Requirements

| Task                                 | Minimum         | Recommended     |
| ------------------------------------ | --------------- | --------------- |
| Skin Tone Classification (inference) | CPU             | Any GPU         |
| Skin Tone Classifier Training        | 1× GPU (8 GB)  | 1× A100 40 GB  |
| Augmentation Pipeline                | CPU (slow)      | GPU recommended |
| Diagnostic Training (ISIC 2019)      | 1× GPU (12 GB) | 1× A100 40 GB  |
| VLM Teacher Generation               | 1× GPU (24 GB) | 1× A100 80 GB  |
| VLM SFT (QLoRA)                      | 1× GPU (24 GB) | 1× A100 40 GB  |
| VLM DPO                              | 1× GPU (40 GB) | 1× A100 80 GB  |
| GUI (with VLM)                       | 1× GPU (16 GB) | 1× GPU (24 GB) |
| GUI (CNN only)                       | CPU             | Any GPU         |

---

## Reproducibility

All experiments use **fixed random seeds** (default: 42). Key reproducibility notes:

- **Skin Tone Classification:** `train_fairface.py` sets `torch.manual_seed`, `np.random.seed`, and `random.seed`.
- **Augmentation:** Both V1 and V2 pipelines accept `--seed` argument.
- **Diagnostics:** `train_test_split(..., random_state=42)` ensures consistent data splits.
- **VLM Training:** SFT and DPO scripts use fixed seeds in training arguments.
- **Hyperparameter rationale:**
  - Ordinal CE loss (classifier) — MST is ordered; penalising far-off errors more heavily improves calibration.
  - λ=0.7 (augmentation V1) — 70% Reinhard chosen for speed; 30% deep blending for diversity.
  - QLoRA rank=64, α=128 (VLM SFT) — balances parameter efficiency with expressiveness for a 7B model.
  - β=0.1 (DPO) — standard setting following the DPO paper.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{idrissou2026equitable,
  title   = {Equitable Skin Cancer Diagnostics with VLM-Guided Clinical Reasoning},
  author  = {Idrissou, Jamal},
  year    = {2026},
  url     = {https://github.com/jayok7/skin-cancer-diagnostics}
}
```

---

## License

This project is for **academic and research purposes**. See individual dependency licences for third-party components. The VLM component (Qwen2.5-VL) is available under Apache 2.0.

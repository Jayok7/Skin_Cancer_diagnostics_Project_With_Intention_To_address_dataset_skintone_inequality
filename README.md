# Equitable Skin Cancer Diagnostics with VLM-Guided Clinical Reasoning

> A reproducible pipeline for building fair, interpretable skin cancer diagnostic systems — from skin tone classification through augmented training to VLM-based clinical reasoning.

**Author:** Jamal Idrissou

---

## Overview

This project addresses **diagnostic bias in dermatology AI** by building a complete pipeline that:

1. **Classifies skin tone** using the Monk Skin Tone (MST) scale via an EfficientNet-B4 model trained on a merged FairFace and UTKFace dataset
2. **Augments underrepresented skin tones** using UNET-Skin transplant + Poisson blending (configurable λ ratio)
3. **Trains diagnostic models** (EfficientNet-B3) on balanced datasets and evaluates fairness across skin tones
4. **Generates clinical reasoning** using a Vision-Language Model (Qwen2.5-VL-7B) fine-tuned with SFT + DPO
5. **Provides a clinical GUI** for real-time lesion analysis with Grad-CAM visualisation and VLM reasoning

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  ISIC 2019   │────▶│  Skin Tone   │────▶│  Tone-Aware  │
  │  25k images  │     │  Classifier  │     │ Augmentation │
  └──────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
  ┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
  │  Clinical    │◀────│  SFT + DPO   │◀────│  Diagnostic  │
  │  GUI App     │     │  Fine-Tuning │     │  EfficientNet│
  └──────────────┘     └──────────────┘     └──────────────┘
```

---

## Repository Structure

```
├── datasets/
│   ├── compute_fairface_labels.py            # Computes MST labels for FairFace via ITA
│   ├── compute_utkface_mst_labels.py         # Computes MST labels for UTKFace via ITA
│   ├── clean_fairface_labels.py              # Cleans erroneous label computations
│   └── merge_mst_datasets.py                 # Merges FairFace and UTKFace labels into master
│
├── skin-tone-classifier/
│   ├── fairface_dataset.py                   # PyTorch dataset for FairFace
│   ├── mskcc_dataset.py                      # PyTorch dataset for MSKCC
│   ├── train_fairface.py                     # Train MST classifier (EfficientNet-B4)
│   ├── classify_skin_tone.py                 # Inference: label images with skin tone
│   └── evaluate_predictions_vs_mst_raters.py # Evaluate against human raters
│
├── skin-augmentation/
│   ├── skin_tone_augmentation.py             # Reinhard + Poisson blending with QC filter
│   ├── skin_tone_augmentation_v2.py          # Advanced augmentation using U-Net (fine-tuned on ISIC 2018)
│   └── skin_tone_transfer.py                 # Core colour transfer algorithms
│
├── diagnostics/
│   ├── skin_cancer_diagnostics_isic2019.py   # Train EfficientNet-B3 (8-class ISIC 2019)
│   └── evaluate_isic2019_test.py             # Compare original vs augmented models
│
├── vlm-finetune/
│   ├── generate_cnn_outputs.py               # EfficientNet inference + Grad-CAM
│   ├── teacher_generation_free.py            # Qwen2.5-VL teacher reasoning (free)
│   ├── quality_control.py                    # 5-stage QC pipeline
│   ├── format_dataset.py                     # Convert to JSONL for SFT
│   ├── train_sft.py                          # QLoRA SFT on Qwen2.5-VL-7B
│   ├── generate_dpo_pairs.py                 # Preference pair generation
│   ├── train_dpo.py                          # DPO alignment training
│   └── evaluate.py                           # SFT vs DPO comparison
│
├── gui/
│   ├── app.py                                # Flask backend (inference + Grad-CAM)
│   ├── index.html                            # Clinical interface
│   ├── style.css                             # Styling
│   ├── script.js                             # Frontend logic
│   └── requirements.txt                      # GUI-specific dependencies
│
├── csf/                                      # Slurm submission scripts
│   ├── submit_skin_augmentation.sh
│   ├── submit_diagnostics_isic2019.sh
│   ├── submit_teacher_generation.sh
│   ├── submit_vlm_sft.sh
│   ├── submit_vlm_dpo.sh
│   └── ...
│
└── README.md
```

---

## Pipeline

### Stage 1: Dataset Generation and Skin Tone Classification

To ensure generalizability, we build a robust, varied skin tone dataset by merging FairFace and UTKFace, labeling them algorithmically using Individual Typology Angle (ITA) mapped to the Monk Skin Tone (MST) scale.

**1. Label Computing & Dataset Merging:**

```bash
# Compute MST labels for FairFace
python compute_fairface_labels.py

# Compute MST labels for UTKFace
python compute_utkface_mst_labels.py

# Clean erroneous FairFace labels
python clean_fairface_labels.py

# Merge the datasets into master_mst_labels.csv
python merge_mst_datasets.py
```

**2. Train Skin Tone Classifier:**
Trains an EfficientNet-B4 model on the merged MST labels using Ordinal Cross-Entropy loss. Supports fine-tuning on MSKCC. This stage also leverages a U-Net model fine-tuned on ISIC 2018 for lesion segmentation to assist in classification and fine-tuning.

```bash
# Train the classifier
python train_fairface.py \
    --data-csv datasets/master_mst_labels.csv \
    --image-root datasets/ \
    --output-dir outputs/fairface/

# Label ISIC 2019 images with skin tone
python classify_skin_tone.py \
    --model outputs/fairface_mskcc_best.pth \
    --images datasets/ISIC_2019_Training_Input \
    --output outputs/isic2019_skin_tone_labels.csv
```

**Key finding**: ISIC 2019 distribution — Light: 41%, Medium: 58%, **Dark: 0.4%**

---

### Stage 2: Skin Tone Augmentation

Generates synthetic dark-skin-tone images to balance the training set using two methods:

| Method                         | λ Control         | What it does                                                             |
| ------------------------------ | ------------------ | ------------------------------------------------------------------------ |
| **UNET Skin Transplant** | λ proportion      | Lesion-aware segmentation - augments healthy skin, preserves core lesion |
| **Poisson Blending**     | 1 − λ proportion | Gradient-domain - better texture, slower                                |

```bash
# Generate augmented images (λ=0.7 → 70% Skin Transplant, 30% Poisson)
python skin-augmentation/skin_tone_augmentation_v2.py \
    --source-dir datasets/ISIC_2019_Training_Input \
    --labels outputs/isic2019_skin_tone_labels.csv \
    --output-dir datasets/ISIC_2019_Augmented \
    --lambda-ratio 0.7
```

**Quality filter**: Dual-region (border + central) realism gate checking hue, saturation, and patchiness. Iterates up to 4 cycles with 1.5× oversampling to meet targets despite rejections.

**Advanced Augmentation (v2)**: The pipeline includes an advanced script (`skin-augmentation/skin_tone_augmentation_v2.py`) which uses a U-Net segmentation model fine-tuned on the ISIC 2018 dataset. This ensures that only healthy perilesional skin is augmented while keeping the core lesion intact, further preserving critical diagnostic morphology.

---

### Stage 3: Diagnostic Model Training

Trains EfficientNet-B3 on ISIC 2019 (8-class) with optional augmented data integration.

```bash
# Train on original data only
python diagnostics/skin_cancer_diagnostics_isic2019.py \
    --gt datasets/ISIC_2019_Training_GroundTruth.csv

# Compare original vs augmented models on official test set (8238 images)
python diagnostics/evaluate_isic2019_test.py \
    --model-orig  outputs/isic2019_orig/best_efficientnet_b3_isic2019.pth \
    --model-aug07 outputs/isic2019_aug07/best_efficientnet_b3_isic2019.pth \
    --model-aug03 outputs/isic2019_aug03/best_efficientnet_b3_isic2019.pth
```

Generates per-model confusion matrices and a side-by-side comparison chart.

---

### Stage 4: VLM Clinical Reasoning

Generates and refines clinical diagnostic reasoning using a vision-language model.

**Teacher–Student setup:**

- **Teacher**: Qwen2.5-VL-7B-Instruct (open-source, generates training data).
  *Note: A commercial GPT-4o teacher pipeline is also available for higher-quality reference generations. Deploying GPT-4o as a teacher costs roughly $0.005 to $0.01 per lesion depending on the context length and resolution.*
- **Student**: Qwen2.5-VL-7B-Instruct (fine-tuned via SFT → DPO)

The VLM receives **both** the original dermoscopic image and the Grad-CAM heatmap overlay, ensuring clinical reasoning is grounded in lesion morphology and model attention patterns.

```bash
# 1. Generate CNN outputs + Grad-CAM heatmaps
python vlm-finetune/generate_cnn_outputs.py

# 2. Generate teacher reasoning (Qwen2.5-VL, free, local GPU)
python vlm-finetune/teacher_generation_free.py

# 3. Quality control — 5-stage filter
#    (keyword, indeterminate, spatial/Grad-CAM, length, safety)
python vlm-finetune/quality_control.py

# 4. Format for training
python vlm-finetune/format_dataset.py

# 5. SFT training (QLoRA, 4-bit on Qwen2.5-VL-7B)
python vlm-finetune/train_sft.py

# 6. Generate DPO preference pairs
python vlm-finetune/generate_dpo_pairs.py

# 7. DPO alignment training
python vlm-finetune/train_dpo.py

# 8. Evaluate SFT vs DPO
python vlm-finetune/evaluate.py
```

---

### Stage 5: Clinical GUI

A web-based clinical interface for real-time lesion analysis.

**Three-panel display:**

1. **Original Image** — the uploaded dermoscopic photograph
2. **Grad-CAM Overlay** — heatmap showing model attention regions
3. **VLM Reasoning** — clinical diagnostic paragraph from the fine-tuned VLM

```bash
cd gui/
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

#### Integrating VLM Output

The GUI currently uses a placeholder for VLM text. To connect the fine-tuned VLM:

1. **Load the model** in `app.py` alongside the diagnostic model:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

vlm_processor = AutoProcessor.from_pretrained("./checkpoints/dpo/final")
vlm_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
vlm_model = PeftModel.from_pretrained(vlm_base, "./checkpoints/dpo/final")
vlm_model.eval()
```

2. **Add a VLM inference function** in `app.py`:

```python
clascdef generate_vlm_reasoning(original_b64, gradcam_b64, prediction, confidence, all_scores):
    """Generate clinical reasoning from the fine-tuned VLM."""
    messages = [
        {"role": "system", "content": "You are a clinical AI assistant..."},
        {"role": "user", "content": [
            {"type": "image", "image": original_b64},
            {"type": "image", "image": gradcam_b64},
            {"type": "text", "text": f"Predicted: {prediction} ({confidence:.1%})..."}
        ]}
    ]
    inputs = vlm_processor.apply_chat_template(messages, return_tensors="pt")
    outputs = vlm_model.generate(**inputs, max_new_tokens=256)
    return vlm_processor.decode(outputs[0], skip_special_tokens=True)
```

3. **Replace the mock text** in `script.js` (line 197):

```diff
- const mockedVLM = "Upon dermoscopic evaluation...";
- document.getElementById('vlmOutput').textContent = mockedVLM;
+ document.getElementById('vlmOutput').textContent = data.vlm_reasoning;
```

4. **Add VLM output to the `/predict` response** in `app.py`:

```diff
  return jsonify({
      'success': True,
      ...
+     'vlm_reasoning': vlm_text,
  })
```

---

## Datasets

| Dataset             | Size           | Source                                                                                            | Used for                           |
| ------------------- | -------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **ISIC 2019** | 25,331 images  | [ISIC Challenge](https://challenge.isic-archive.com/data/#2019)                                      | Diagnostic training + augmentation |
| **HAM10000**  | 10,015 images  | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | VLM teacher generation             |
| **FairFace**  | 108,501 images | [GitHub](https://github.com/joojs/fairface)                                                          | Skin tone classifier pre-training  |
| **UTKFace**   | 23,705 images  | [GitHub](https://github.com/aicip/UTKFace)                                                           | Skin tone classifier pre-training  |
| **MSKCC**     | 4,880 images   | Clinical dataset                                                                                  | Skin tone classifier fine-tuning   |

> **Note**: Datasets are not included in this repository due to size and licensing. Download from the links above and place in `datasets/`.

---

## Requirements

```
# Core
torch>=2.0
torchvision
transformers  # Install from source for Qwen3-VL: pip install git+https://github.com/huggingface/transformers.git
accelerate
bitsandbytes
peft
trl>=1.0
datasets
pandas

# Image processing
opencv-python
Pillow
scikit-learn

# Visualisation
matplotlib
seaborn

# VLM-specific
qwen-vl-utils

# GUI
flask
tensorflow  # For the clinical GUI backend
```

### Hardware

| Task                     | Minimum GPU        | VRAM     |
| ------------------------ | ------------------ | -------- |
| Skin tone classification | Any CUDA GPU       | 4 GB     |
| Augmentation             | CPU (multicore)    | 8 GB RAM |
| Diagnostic training      | NVIDIA GPU         | 8 GB     |
| VLM teacher generation   | NVIDIA L40S / A100 | 24+ GB   |
| VLM SFT/DPO training     | NVIDIA L40S / A100 | 24+ GB   |
| GUI (inference only)     | CPU or GPU         | 4+ GB    |

---

## Reproducibility

All experiments use `seed=42`. Slurm submission scripts in `csf/` are configured for the University of Manchester CSF3 cluster but can be adapted to any Slurm environment. The entire pipeline, from labeling via individual typology angle (ITA) upwards, uses static seeds and deterministic operations where possible.

### Key Hyperparameters

| Parameter               | Value     | Rationale                                          |
| ----------------------- | --------- | -------------------------------------------------- |
| λ (augmentation ratio) | 0.7 / 0.3 | Controls Reinhard vs Poisson blend proportion      |
| QLoRA rank (r)          | 64        | Balances expressivity vs parameter count           |
| QLoRA α                | 128       | α/r = 2.0 amplifies LoRA updates for domain shift |
| SFT learning rate       | 2e-4      | Standard for QLoRA                                 |
| SFT epochs              | 3         | Sufficient for ~1750 clean examples                |
| DPO β (KL penalty)     | 0.1       | Moderate constraint on policy divergence           |
| QC min pass score       | 3/5       | Expected 20-30% discard rate                       |
| Image resolution (CNN)  | 300×300  | EfficientNet-B3 input size                         |

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

## Licence

MIT Licence. See [LICENCE](LICENCE) for details.

> **Medical disclaimer**: This system is for research purposes only. It does not provide medical diagnoses and should not be used as a substitute for professional medical advice.

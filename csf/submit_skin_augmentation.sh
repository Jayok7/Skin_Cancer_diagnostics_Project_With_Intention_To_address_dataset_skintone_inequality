#!/bin/bash
#SBATCH --job-name=skin_aug
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/skin_augmentation_%j.out
#SBATCH --error=logs/skin_augmentation_%j.err

# ================================================================
# Skin Tone Augmentation — Reinhard CIE-LAB + Deep Blending
# ================================================================
# Balances the ISIC 2019 skin-tone distribution using:
#   - Reinhard CIE-LAB colour transfer (λ proportion)
#   - Deep Blending / Poisson image editing (1-λ proportion)
#   - MSKCC images as the reference style pool
#
# CPU only — no GPU required (uses OpenCV, not deep learning)
#
# Prerequisites:
#   - ISIC 2019 images at datasets/ISIC_2019_Training_Input/
#   - MSKCC images at datasets/MSKCC-images/
#   - Label CSVs at outputs/
# ================================================================

echo "=== Skin Tone Augmentation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs

# ── Configuration ──────────────────────────────────────────
ISIC_CSV="outputs/isic2019_skin_tone_labels.csv"
MSKCC_CSV="outputs/mskcc_skin_tone_labels.csv"
ISIC_IMAGES="datasets/ISIC_2019_Training_Input"
MSKCC_IMAGES="datasets/MSKCC-images"
OUTPUT_DIR="datasets/ISIC_2019_Augmented"

# λ = 0.7 means 70% Reinhard, 30% Deep Blending
LAMBDA=0.7

# Target: ~8500 per class for roughly even distribution
TARGET=8500

echo ""
echo "  ISIC CSV:     $ISIC_CSV"
echo "  MSKCC CSV:    $MSKCC_CSV"
echo "  Lambda:       $LAMBDA"
echo "  Target/class: $TARGET"
echo ""

# ── Validation ─────────────────────────────────────────────
if [ ! -f "$ISIC_CSV" ]; then
    echo "ERROR: ISIC labels CSV not found at $ISIC_CSV"
    exit 1
fi

if [ ! -f "$MSKCC_CSV" ]; then
    echo "ERROR: MSKCC labels CSV not found at $MSKCC_CSV"
    exit 1
fi

if [ ! -d "$ISIC_IMAGES" ]; then
    echo "ERROR: ISIC images directory not found at $ISIC_IMAGES"
    exit 1
fi

if [ ! -d "$MSKCC_IMAGES" ]; then
    echo "ERROR: MSKCC images directory not found at $MSKCC_IMAGES"
    exit 1
fi

# ── Run augmentation ───────────────────────────────────────
python skin_tone_augmentation.py \
    --isic-csv "$ISIC_CSV" \
    --mskcc-csv "$MSKCC_CSV" \
    --isic-images "$ISIC_IMAGES" \
    --mskcc-images "$MSKCC_IMAGES" \
    --output-dir "$OUTPUT_DIR" \
    --lambda-ratio $LAMBDA \
    --target-per-class $TARGET \
    --seed 42

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"

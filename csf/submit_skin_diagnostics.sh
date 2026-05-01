#!/bin/bash
#SBATCH --job-name=skin_diag
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/skin_diagnostics_%j.out
#SBATCH --error=logs/skin_diagnostics_%j.err

# ================================================================
# Combined Skin Cancer Diagnostics — EfficientNet-B3 + Grad-CAM
# ================================================================
# Trains EfficientNet-B3 on HAM10000 (40 epochs: 20 head + 20 fine-tune),
# evaluates with confusion matrix and classification report,
# then generates Grad-CAM visualisations.
# Saves best model checkpoint based on validation accuracy.
#
# Prerequisites:
#   Transfer HAM10000 data to CSF:
#     datasets/Ham10000/HAM10000_metadata.csv
#     datasets/Ham10000/HAM10000_images_part_1/
#     datasets/Ham10000/HAM10000_images_part_2/
#
# Usage:
#   sbatch submit_skin_diagnostics.sh
# ================================================================

echo "=== Skin Cancer Diagnostics ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs

source activate skin_tone_env 2>/dev/null || conda activate skin_tone_env
pip install 'numpy<2' --quiet 2>/dev/null

# ── Paths (matching CSF directory layout) ──
BASE_PATH="datasets/Ham10000"
CSV_PATH="datasets/Ham10000/HAM10000_metadata.csv"
OUTPUT_DIR="outputs/skin_cancer_diagnostics"

echo ""
echo "Base path: $BASE_PATH"
echo "CSV:       $CSV_PATH"
echo "Output:    $OUTPUT_DIR"
echo ""

# ── Verify prerequisites ──
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: HAM10000 metadata CSV not found at $CSV_PATH"
    exit 1
fi

if [ ! -d "$BASE_PATH/HAM10000_images_part_1" ] || [ ! -d "$BASE_PATH/HAM10000_images_part_2" ]; then
    echo "ERROR: HAM10000 image directories not found."
    echo "  Expected at: $BASE_PATH/HAM10000_images_part_1/ and HAM10000_images_part_2/"
    exit 1
fi

python skin_cancer_diagnostics.py \
    --base-path "$BASE_PATH" \
    --csv-path "$CSV_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"

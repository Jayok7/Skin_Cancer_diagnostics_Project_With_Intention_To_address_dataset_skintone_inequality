#!/bin/bash
#SBATCH --job-name=isic2019
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/isic2019_%j.out
#SBATCH --error=logs/isic2019_%j.err

# ================================================================
# ISIC 2019 Diagnostics — EfficientNet-B3 (Augmented)
# ================================================================
# Trains EfficientNet-B3 on ISIC 2019 with optional skin-tone
# augmented images. 40 epochs (20 head + 20 fine-tune).
#
# Prerequisites:
#   - ISIC 2019 images at datasets/ISIC_2019_Training_Input/
#   - Ground truth at datasets/ISIC_2019_Training_GroundTruth.csv
#   - (Optional) Augmented images at datasets/ISIC_2019_Augmented/
# ================================================================

echo "=== ISIC 2019 Diagnostics Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs outputs/isic2019_aug03

# ── Environment ────────────────────────────────────────────
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pip install "numpy<2.0" 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip install matplotlib seaborn scikit-learn Pillow pandas opencv-python-headless 2>/dev/null

echo ""
nvidia-smi
echo ""

# ── Configuration ──────────────────────────────────────────
IMAGE_DIR="datasets/ISIC_2019_Training_Input"
GT_CSV="datasets/ISIC_2019_Training_GroundTruth.csv"
AUG_DIR="datasets/ISIC_2019_Augmented_03"
AUG_MANIFEST="datasets/ISIC_2019_Augmented_03/augmentation_manifest.csv"
OUTPUT_DIR="outputs/isic2019_aug03"

# ── Validate ───────────────────────────────────────────────
if [ ! -f "$GT_CSV" ]; then
    echo "ERROR: Ground truth CSV not found at $GT_CSV"
    echo "  Download from: https://challenge.isic-archive.com/data/#2019"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found at $IMAGE_DIR"
    exit 1
fi

# Check for augmented data (optional)
AUG_FLAGS=""
if [ -f "$AUG_MANIFEST" ]; then
    echo "  Augmented data found — including in training"
    AUG_FLAGS="--aug-dir $AUG_DIR --aug-manifest $AUG_MANIFEST"
else
    echo "  No augmented data — training on original ISIC 2019 only"
fi

# ── Run ────────────────────────────────────────────────────
python skin_cancer_diagnostics_isic2019.py \
    --image-dir "$IMAGE_DIR" \
    --gt-csv "$GT_CSV" \
    $AUG_FLAGS \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"

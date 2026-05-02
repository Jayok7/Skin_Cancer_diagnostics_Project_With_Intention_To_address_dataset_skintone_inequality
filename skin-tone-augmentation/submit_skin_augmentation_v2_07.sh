#!/bin/bash
#SBATCH --job-name=skin_aug_v2
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/skin_aug_v2_%j.out
#SBATCH --error=logs/skin_aug_v2_%j.err

# ================================================================
# Skin Tone Augmentation v2 -- U-Net Transplant + Deep Blending
# ================================================================
# Replaces Reinhard CIE-LAB with U-Net lesion transplant.
# lambda = 0.7 means 70% U-Net transplant, 30% deep blend.
# Requires GPU for U-Net segmentation.
#
# Prerequisites:
#   - ISIC 2019 images + skin tone labels
#   - MSKCC images + skin tone labels
#   - Pytorch-UNet with trained checkpoint
# ================================================================

echo "=== Skin Tone Augmentation v2 ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs

# -- Environment --
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pip install "numpy<2.0" 2>/dev/null
pip install opencv-python-headless matplotlib pandas tqdm Pillow 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null

echo ""
nvidia-smi
echo ""

# -- Configuration --
ISIC_CSV="outputs/isic2019_skin_tone_labels.csv"
MSKCC_CSV="outputs/mskcc_skin_tone_labels.csv"
ISIC_IMAGES="datasets/ISIC_2019_Training_Input"
MSKCC_IMAGES="datasets/MSKCC-images"
UNET_DIR="Pytorch-UNet"
UNET_WEIGHTS="Pytorch-UNet/checkpoints/checkpoint_epoch50.pth"

# Lambda: 0.7 = 70% transplant, 30% deep blend
LAMBDA=0.7
TARGET=8500
OUTPUT_DIR="datasets/ISIC_2019_Augmented_v2_07"

echo "  ISIC CSV:     $ISIC_CSV"
echo "  MSKCC CSV:    $MSKCC_CSV"
echo "  U-Net:        $UNET_WEIGHTS"
echo "  Lambda:       $LAMBDA"
echo "  Target/class: $TARGET"
echo "  Output:       $OUTPUT_DIR"
echo ""

# -- Validate --
if [ ! -f "$ISIC_CSV" ]; then
    echo "ERROR: ISIC labels CSV not found at $ISIC_CSV"
    exit 1
fi
if [ ! -f "$MSKCC_CSV" ]; then
    echo "ERROR: MSKCC labels CSV not found at $MSKCC_CSV"
    exit 1
fi
if [ ! -d "$ISIC_IMAGES" ]; then
    echo "ERROR: ISIC images not found at $ISIC_IMAGES"
    exit 1
fi
if [ ! -d "$MSKCC_IMAGES" ]; then
    echo "ERROR: MSKCC images not found at $MSKCC_IMAGES"
    exit 1
fi
if [ ! -f "$UNET_WEIGHTS" ]; then
    echo "ERROR: U-Net weights not found at $UNET_WEIGHTS"
    exit 1
fi

# -- Run --
python skin_tone_augmentation_v2.py \
    --isic-csv "$ISIC_CSV" \
    --mskcc-csv "$MSKCC_CSV" \
    --isic-images "$ISIC_IMAGES" \
    --mskcc-images "$MSKCC_IMAGES" \
    --unet-dir "$UNET_DIR" \
    --unet-weights "$UNET_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --lambda-ratio $LAMBDA \
    --target-per-class $TARGET \
    --max-cycles 4 \
    --unet-threshold 0.5 \
    --pad-pixels 10 \
    --seed 42 \
    --device cuda

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"
#!/bin/bash
# =========================================================================
# CSF Submission: Fine-tune U-Net on ISIC 2018 Lesion Segmentation
# =========================================================================
# This script:
#   1. Clones milesial/Pytorch-UNet (if not present)
#   2. Prepares the ISIC 2018 data into the expected /data/imgs and /data/masks format
#   3. Downloads the pretrained Carvana weights for transfer learning
#   4. Trains the model and saves checkpoints
#
# Usage:  sbatch submit_finetune_unet.sh
# =========================================================================

#SBATCH --job-name=finetune_unet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/finetune_unet_%j.log
#SBATCH --error=logs/finetune_unet_%j.err

# Load required modules
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

# Set environment path
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

mkdir -p logs

# 1. Setup U-Net Repository
if [ ! -d "Pytorch-UNet" ]; then
    echo "Cloning milesial/Pytorch-UNet..."
    git clone https://github.com/milesial/Pytorch-UNet.git
fi

cd Pytorch-UNet
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 2. Prepare Data Structure
echo "Preparing dataset structure for U-Net..."
mkdir -p data/imgs data/masks

# Find where the ISIC data actually lives
ISIC_IMG_DIR=""
ISIC_MASK_DIR=""
for candidate in "../datasets/ISIC2018_Task1-2_Training_Input" "../datasets/ISIC2018/ISIC2018_Task1-2_Training_Input"; do
    if [ -d "$candidate" ]; then
        ISIC_IMG_DIR="$candidate"
        break
    fi
done
for candidate in "../datasets/ISIC2018_Task1_Training_GroundTruth" "../datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth"; do
    if [ -d "$candidate" ]; then
        ISIC_MASK_DIR="$candidate"
        break
    fi
done

if [ -z "$ISIC_IMG_DIR" ] || [ -z "$ISIC_MASK_DIR" ]; then
    echo "ERROR: Could not find ISIC 2018 data!"
    echo "  Searched from: $(pwd)"
    echo "  Contents of ../datasets/:"
    ls -la ../datasets/ 2>/dev/null || echo "  ../datasets/ not found"
    exit 1
fi

echo "Found ISIC images in: $ISIC_IMG_DIR ($(ls "$ISIC_IMG_DIR"/*.jpg 2>/dev/null | wc -l) files)"
echo "Found ISIC masks in:  $ISIC_MASK_DIR ($(ls "$ISIC_MASK_DIR"/*_segmentation.png 2>/dev/null | wc -l) files)"

# Copy ISIC 2018 images to expected directory
# Only copy if no .jpg files present (ls -A is fooled by .gitkeep)
if [ -z "$(ls data/imgs/*.jpg 2>/dev/null)" ]; then
    echo "Copying training images..."
    cp "$ISIC_IMG_DIR"/*.jpg data/imgs/
fi

# Copy ISIC 2018 masks to expected directory
if [ -z "$(ls data/masks/*.png 2>/dev/null)" ]; then
    echo "Copying ground truth masks..."
    for mask in "$ISIC_MASK_DIR"/*_segmentation.png; do
        if [ -f "$mask" ]; then
            filename=$(basename -- "$mask")
            new_filename="${filename/_segmentation.png/.png}"
            cp "$mask" "data/masks/$new_filename"
        fi
    done
fi
echo "Data preparation complete. Found $(ls data/imgs | wc -l) images and $(ls data/masks | wc -l) masks."

# 3. Download Pretrained Carvana Weights
mkdir -p checkpoints
if [ ! -f "checkpoints/carvana_weights.pth" ]; then
    echo "Downloading pretrained Carvana weights for transfer learning..."
    wget -q https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth -O checkpoints/carvana_weights.pth
fi

# 4. Install training specific dependencies
echo "Verifying / Installing training dependencies..."
pip install tqdm wandb

# 5. Patch train.py and data_loading.py for CSF + ISIC compatibility
# a) Handle missing 'mask_values' key in raw pretrained weights
sed -i "s/del state_dict\['mask_values'\]/state_dict.pop('mask_values', None)/" train.py
# b) Use SLURM-allocated CPUs, not full node (os.cpu_count() sees all 48 cores, we only have 4)
sed -i "s/num_workers=os.cpu_count()/num_workers=min(os.cpu_count(), int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))/" train.py
# c) ISIC images have varying sizes:  resize all to fixed 256x256 so they can be batched
sed -i 's/newW, newH = int(scale \* w), int(scale \* h)/newW, newH = 256, 256/' utils/data_loading.py

# 6. Execute Training
echo "=========================================="
echo "Training U-Net from scratch on ISIC 2018"
echo "  No transfer learning (Carvana too different)"
echo "  Epochs: 50"
echo "  LR: 1e-4, Batch: 4, AMP: on"
echo "=========================================="

# Run offline to avoid wandb login prompts
export WANDB_MODE="offline"

python train.py \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --scale 0.5 \
    --validation 10

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

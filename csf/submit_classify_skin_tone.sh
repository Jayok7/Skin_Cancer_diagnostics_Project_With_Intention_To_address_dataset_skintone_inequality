#!/bin/bash
# =========================================================================
# CSF Job: Lesion-Aware Skin Tone Classification (A+B Cascade)
# =========================================================================
# Purpose: Classify MSKCC *clinical close-up* images using perilesional
#          ring (A) with multi-patch consensus fallback (B).
#          Uses milesial/Pytorch-UNet for segmentation.
#          Uses FairFace v3.2 for CNN predicting and validation.
#
# Usage:  sbatch submit_classify_skin_tone.sh
# =========================================================================

#SBATCH --job-name=mskcc_closeup_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/classify_skin_tone_%j.log
#SBATCH --error=logs/classify_skin_tone_%j.err

# Load modules
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

# Activate conda env
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "=========================================="
echo "MSKCC Skin Tone Classification (A+B Cascade)"
echo "  Filter: clinical close-up only"
echo "  Segmentation: milesial/Pytorch-UNet"
echo "  Started at $(date)"
echo "=========================================="
echo ""

cd ~/skin-cancer

# Create directories
mkdir -p logs
mkdir -p outputs/skin_tone_cascade_closeup/visualisations

# =========================================================================
# Install dependencies
# =========================================================================
echo "Installing dependencies..."

# Do NOT pip install numpy or pandas — the conda env already has versions
# compiled against each other. Overwriting them causes binary incompatibility.
pip install "opencv-python-headless<4.11" tqdm Pillow seaborn scikit-learn

# Clone milesial/Pytorch-UNet if not already present
if [ ! -d "Pytorch-UNet" ]; then
    echo "Cloning milesial/Pytorch-UNet..."
    git clone https://github.com/milesial/Pytorch-UNet.git
fi

echo ""

# =========================================================================
# Run Classification — Clinical Close-Up Only
# =========================================================================
echo "=========================================="
echo "Running A+B cascade on MSKCC close-up images"
echo "  Filtering dermoscopic via metadata.csv"
echo "  U-Net segmentation: Carvana pretrained (auto-download)"
echo "  Running FairFace Model v3.2"
echo "=========================================="
echo ""

python classify_skin_tone.py \
    --image-dir datasets/MSKCC-images/ \
    --metadata-csv datasets/MSKCC-images/metadata.csv \
    --eval-csv datasets/MSKCC-images/metadata.csv \
    --cnn-model outputs/FairFace-Model-3.2-mst5/best_finetuned_model.pth \
    --eval-mode 5-way \
    --unet-dir Pytorch-UNet \
    --unet-weights Pytorch-UNet/checkpoints/checkpoint_epoch50.pth \
    --output-dir outputs/skin_tone_cascade_closeup/ \
    --visualise \
    --margin-px 30 \
    --min-ring-pixels 500

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

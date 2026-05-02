#!/bin/bash
#SBATCH --job-name=style_test
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/style_test_%j.out
#SBATCH --error=logs/style_test_%j.err

# ================================================================
# Neural Style Transfer — Proof of Concept (5 images)
# ================================================================
# Tests Gatys et al. VGG-based style transfer for skin tone
# augmentation. Transfers dark skin tone from MSKCC references
# onto light-skin ISIC lesion images.
#
# ~30s per image on L40S GPU → 5 images ≈ 3 minutes
# ================================================================

echo "=== Neural Style Transfer Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs outputs/style_transfer_test

# ── Environment ────────────────────────────────────────────
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pip install "numpy<2.0" 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip install matplotlib seaborn scikit-learn Pillow pandas 2>/dev/null

echo ""
nvidia-smi
echo ""

# ── Run ────────────────────────────────────────────────────
python style_transfer_test.py \
    --isic-csv outputs/isic2019_skin_tone_labels.csv \
    --mskcc-csv outputs/mskcc_skin_tone_labels.csv \
    --isic-images datasets/ISIC_2019_Training_Input \
    --mskcc-images datasets/MSKCC-images \
    --output-dir outputs/style_transfer_test \
    --num-samples 5 \
    --steps 300 \
    --content-weight 1e5 \
    --style-weight 1e10 \
    --device cuda

echo ""
echo "=== Done ==="
echo "Date: $(date)"
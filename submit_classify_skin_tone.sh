#!/bin/bash
# =========================================================================
# CSF Job: Lesion-Aware Skin Tone Classification (A+B Cascade)
# =========================================================================
# Purpose: Classify MSKCC dermoscopic images using perilesional ring (A)
#          with multi-patch consensus fallback (B). CPU-only job.
#
# Usage:  sbatch submit_classify_skin_tone.sh
# =========================================================================

#SBATCH --job-name=mskcc_skin_tone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=multicore
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/classify_skin_tone_%j.log

echo "=========================================="
echo "MSKCC Skin Tone Classification (A+B Cascade)"
echo "Started at $(date)"
echo "=========================================="
echo ""

cd ~/skin-cancer

# Create directories
mkdir -p logs
mkdir -p outputs/skin_tone_cascade/visualisations

# =========================================================================
# Install dependencies
# =========================================================================
echo "Installing dependencies..."

# Pin opencv to 4.10.x (compatible with numpy 1.x)
pip install "opencv-python-headless<4.11" tqdm Pillow pandas
# Force numpy 1.x LAST so nothing overrides it
pip install --force-reinstall "numpy==1.26.4"

echo ""

# =========================================================================
# Run Classification (Approach B only — no U-Net for initial test)
# =========================================================================
echo "=========================================="
echo "Running A+B cascade on MSKCC images"
echo "(Using heuristic segmentation — no U-Net weights yet)"
echo "=========================================="
echo ""

python classify_skin_tone.py \
    --image-dir datasets/MSKCC-images/ \
    --output-dir outputs/skin_tone_cascade/ \
    --visualise \
    --margin-px 30 \
    --min-ring-pixels 500 \
    --confidence-threshold 15.0

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

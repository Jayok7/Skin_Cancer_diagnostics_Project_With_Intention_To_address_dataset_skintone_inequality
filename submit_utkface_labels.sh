#!/bin/bash
# =========================================================================
# CSF GPU Job: Download + Label UTKFace with MST
# =========================================================================
# Purpose: Download UTKFace dataset and compute ITA-based MST labels.
#          This is a lightweight CPU job (no GPU needed).
#
# Usage:  sbatch submit_utkface_labels.sh
# =========================================================================

#SBATCH --job-name=utkface_mst_labels
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=multicore
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/utkface_labels_%j.log

echo "=========================================="
echo "UTKFace MST Labelling Job"
echo "Started at $(date)"
echo "=========================================="
echo ""

cd ~/skin-cancer

# Create directories
mkdir -p datasets
mkdir -p logs

# Pin opencv to 4.10.x (compatible with numpy 1.x — 4.13 requires numpy>=2)
pip install "opencv-python-headless<4.11" tqdm Pillow pandas
# Force numpy 1.x LAST so nothing overrides it (system pandas needs 1.x)
pip install --force-reinstall "numpy==1.26.4"

# =========================================================================
# Compute MST labels (UTKFace already in datasets/UTKFace via scp)
# =========================================================================
echo ""
echo "=========================================="
echo "Computing ITA → MST labels for UTKFace"
echo "=========================================="

python compute_utkface_mst_labels.py \
    --image-dir datasets/UTKFace/utkface-aligned-labeled/UTKFace_images \
    --output datasets/utkface_mst_labels.csv \
    --num-classes 10

# =========================================================================
# STEP 3: Print summary comparison with FairFace
# =========================================================================
echo ""
echo "=========================================="
echo "STEP 3: Summary"
echo "=========================================="
echo ""
echo "UTKFace labels saved to: datasets/utkface_mst_labels.csv"
echo "FairFace labels at:      datasets/fairface_mst_labels.csv"
echo ""
echo "To merge and train, update submit_csf.sh to use the merged CSV."

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

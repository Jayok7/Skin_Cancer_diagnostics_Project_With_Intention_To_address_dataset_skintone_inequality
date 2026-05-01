#!/bin/bash
#SBATCH --job-name=isic2019_train
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/isic2019_train_%j.out
#SBATCH --error=logs/isic2019_train_%j.err

echo "=== ISIC 2019 Diagnostics Training (Orig + Aug00) ==="
cd ~/skin-cancer

# ── Environment ──
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

# ── Common Config ──
IMAGE_DIR="datasets/ISIC_2019_Training_Input"
GT_CSV="datasets/ISIC_2019_Training_GroundTruth.csv"

# ==========================================
# RUN 1: ORIGINAL (NO AUGMENTATION)
# ==========================================
echo "=== RUN 1: Original Dataset ==="
OUTPUT_ORIG="outputs/isic2019_orig"
mkdir -p "$OUTPUT_ORIG"

python skin_cancer_diagnostics_isic2019.py \
    --image-dir "$IMAGE_DIR" \
    --gt-csv "$GT_CSV" \
    --output-dir "$OUTPUT_ORIG" \
    --device cuda

# ==========================================
# RUN 2: AUGMENTED (LAMBDA = 0)
# ==========================================
echo "=== RUN 2: Augmented Dataset (Lambda=0) ==="
AUG_DIR="datasets/ISIC_2019_Augmented_00"
AUG_MANIFEST="datasets/ISIC_2019_Augmented_00/augmentation_manifest.csv"
OUTPUT_AUG00="outputs/isic2019_aug00"
mkdir -p "$OUTPUT_AUG00"

python skin_cancer_diagnostics_isic2019.py \
    --image-dir "$IMAGE_DIR" \
    --gt-csv "$GT_CSV" \
    --aug-dir "$AUG_DIR" \
    --aug-manifest "$AUG_MANIFEST" \
    --output-dir "$OUTPUT_AUG00" \
    --device cuda

echo "=== All Training Complete ==="
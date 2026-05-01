#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --error=logs/eval_test_%j.err

# ================================================================
# ISIC 2019 Test Set Evaluation
# ================================================================
# Compares Original vs Augmented (λ=0.7) vs Augmented (λ=0.3)
# models on the official ISIC 2019 test set (8238 images).
#
# Prerequisites:
#   - Test images at datasets/ISIC_2019_Test_Input/
#   - Ground truth at datasets/ISIC_2019_Test_GroundTruth.csv
#   - Trained models at their respective output dirs
# ================================================================

echo "=== ISIC 2019 Test Set Evaluation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs outputs/isic2019_test_eval

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

# ── Configuration ──────────────────────────────────────────
TEST_IMAGES="datasets/ISIC_2019_Test_Input"
TEST_GT="datasets/ISIC_2019_Test_GroundTruth.csv"
OUTPUT_DIR="outputs/isic2019_test_eval"

# Model paths — adjust these to match your training output dirs
MODEL_ORIG="outputs/isic2019_orig/best_efficientnet_b3_isic2019.pth"
MODEL_AUG07="outputs/isic2019_aug07/best_efficientnet_b3_isic2019.pth"
MODEL_AUG03="outputs/isic2019_aug03/best_efficientnet_b3_isic2019.pth"

# ── Validate ───────────────────────────────────────────────
if [ ! -f "$TEST_GT" ]; then
    echo "ERROR: Test ground truth not found at $TEST_GT"
    exit 1
fi

# Build model flags (only include models that exist)
MODEL_FLAGS=""
[ -f "$MODEL_ORIG"  ] && MODEL_FLAGS="$MODEL_FLAGS --model-orig $MODEL_ORIG"
[ -f "$MODEL_AUG07" ] && MODEL_FLAGS="$MODEL_FLAGS --model-aug07 $MODEL_AUG07"
[ -f "$MODEL_AUG03" ] && MODEL_FLAGS="$MODEL_FLAGS --model-aug03 $MODEL_AUG03"

if [ -z "$MODEL_FLAGS" ]; then
    echo "ERROR: No trained models found. Train at least one model first."
    echo "  Expected: $MODEL_ORIG or $MODEL_AUG07 or $MODEL_AUG03"
    exit 1
fi

echo "  Models found:"
[ -f "$MODEL_ORIG"  ] && echo "    ✓ Original:     $MODEL_ORIG"
[ -f "$MODEL_AUG07" ] && echo "    ✓ Aug (λ=0.7):  $MODEL_AUG07"
[ -f "$MODEL_AUG03" ] && echo "    ✓ Aug (λ=0.3):  $MODEL_AUG03"

# ── Run ────────────────────────────────────────────────────
python evaluate_isic2019_test.py \
    --test-images "$TEST_IMAGES" \
    --test-gt "$TEST_GT" \
    $MODEL_FLAGS \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "=== Evaluation Complete ==="
echo "Date: $(date)"

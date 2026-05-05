#!/bin/bash
#SBATCH --job-name=fitz_strat
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/fitz_strat_%j.out
#SBATCH --error=logs/fitz_strat_%j.err

# to not waste time
set -e          # Exit immediately on any command failure
set -o pipefail # Catch failures in pipes too

# =========================================================================
# Two-stage Fitzpatrick17k fairness battery (Tests 2-5):
#   Stage 1 (GPU): generate predictions for all 5 models -> per-tag CSVs
#   Stage 2 (CPU): run Tests 2/3/4/5 on those CSVs
# Both arms are run: HIERARCHY (primary) and STRICT (Test 5d sensitivity).
# =========================================================================

echo "=== Fitzpatrick17k Tone-Stratified Evaluation ==="
echo "Date: $(date) | Node: $(hostname) | Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pip install "numpy<2.0" --quiet 2>/dev/null
pip install matplotlib seaborn pandas Pillow opencv-python-headless scikit-learn statsmodels --quiet 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null

echo ""
nvidia-smi
echo ""

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
FITZ_CSV="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/skin-cancer/datasets/fitzpatrick17k.csv"
FITZ_IMG_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/skin-cancer/datasets/fitzpatrick17k/data/finalfitz17k"

PRED_DIR_HIER="outputs/fitzpatrick_predictions_hierarchy"
PRED_DIR_STRICT="outputs/fitzpatrick_predictions_strict"
EVAL_DIR_HIER="outputs/fitzpatrick_stratified_eval/hierarchy"
EVAL_DIR_STRICT="outputs/fitzpatrick_stratified_eval/strict"

# -------------------------------------------------------------------------
# Build model flags (skip missing files gracefully)
# -------------------------------------------------------------------------
MODEL_FLAGS=""
TAGS=""

F="outputs/isic2019_orig/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model orig:$F"
    TAGS="$TAGS orig"
    echo "FOUND: orig -> $F"
else
    echo "MISSING: orig -> $F"
fi

F="outputs/isic2019_aug00/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda00:$F"
    TAGS="$TAGS lambda00"
    echo "FOUND: lambda00 -> $F"
else
    echo "MISSING: lambda00 -> $F"
fi

F="outputs/isic2019_aug_v2_03/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda03:$F"
    TAGS="$TAGS lambda03"
    echo "FOUND: lambda03 -> $F"
else
    echo "MISSING: lambda03 -> $F"
fi

F="outputs/isic2019_aug_v2_07/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda07:$F"
    TAGS="$TAGS lambda07"
    echo "FOUND: lambda07 -> $F"
else
    echo "MISSING: lambda07 -> $F"
fi

F="outputs/isic2019_aug_v2_10/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda10:$F"
    TAGS="$TAGS lambda10"
    echo "FOUND: lambda10 -> $F"
else
    echo "MISSING: lambda10 -> $F"
fi

if [ -z "$MODEL_FLAGS" ]; then
    echo "ERROR: No model checkpoints found."
    exit 1
fi
echo ""
echo "Tags resolved:$TAGS"
echo ""

# =========================================================================
# Stage 1a: Generate predictions -- HIERARCHY mapping (primary)
# =========================================================================
echo "==== Stage 1a: HIERARCHY mapping predictions ===="
python generate_fitzpatrick_predictions.py \
    --csv        "$FITZ_CSV" \
    --image-dir  "$FITZ_IMG_DIR" \
    --mapping    HIERARCHY \
    --output-dir "$PRED_DIR_HIER" \
    --device     cuda \
    $MODEL_FLAGS

# =========================================================================
# Stage 1b: Generate predictions -- STRICT mapping (Test 5d sensitivity)
# =========================================================================
echo ""
echo "==== Stage 1b: STRICT mapping predictions ===="
python generate_fitzpatrick_predictions.py \
    --csv        "$FITZ_CSV" \
    --image-dir  "$FITZ_IMG_DIR" \
    --mapping    STRICT \
    --output-dir "$PRED_DIR_STRICT" \
    --device     cuda \
    $MODEL_FLAGS

# =========================================================================
# Stage 2a: Run Tests 2-5 on HIERARCHY predictions
# =========================================================================
echo ""
echo "==== Stage 2a: Tests 2-5 (HIERARCHY) ===="
python evaluate_fitzpatrick_stratified.py \
    --pred-dir       "$PRED_DIR_HIER" \
    --tags          $TAGS \
    --baseline-tag   orig \
    --output-dir     "$EVAL_DIR_HIER" \
    --bootstrap-iters 1000

# =========================================================================
# Stage 2b: Run Tests 2-5 on STRICT predictions (5d arm)
# =========================================================================
echo ""
echo "==== Stage 2b: Tests 2-5 (STRICT, Test 5d arm) ===="
python evaluate_fitzpatrick_stratified.py \
    --pred-dir       "$PRED_DIR_STRICT" \
    --tags          $TAGS \
    --baseline-tag   orig \
    --output-dir     "$EVAL_DIR_STRICT" \
    --bootstrap-iters 1000

echo ""
echo "=== Fitzpatrick17k stratified evaluation complete ==="
echo "Date: $(date)"
echo ""
echo "HIERARCHY artefacts:"
ls -lh "$EVAL_DIR_HIER" 2>/dev/null
echo ""
echo "STRICT artefacts:"
ls -lh "$EVAL_DIR_STRICT" 2>/dev/null

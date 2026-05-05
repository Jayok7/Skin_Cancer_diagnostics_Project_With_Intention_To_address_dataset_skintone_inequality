#!/bin/bash
#SBATCH --job-name=milk_strat
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/milk_strat_%j.out
#SBATCH --error=logs/milk_strat_%j.err

set -e
set -o pipefail

echo "=== MILK10K Tone-Stratified Evaluation ==="
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
pip install matplotlib pandas Pillow scikit-learn statsmodels --quiet 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet 2>/dev/null

echo ""
nvidia-smi
echo ""

# -------------------------------------------------------------------------
# UPDATE THESE PATHS to match where you placed MILK10K
# -------------------------------------------------------------------------
MILK_CSV="datasets/milk10k_harvard.csv"
MILK_IMG_DIR="datasets/MILK10k_images"

PRED_DIR_D3="outputs/milk10k_predictions_diagnosis3"
PRED_DIR_SIMP="outputs/milk10k_predictions_simplified"
EVAL_DIR_D3="outputs/milk10k_stratified_eval/diagnosis3"
EVAL_DIR_SIMP="outputs/milk10k_stratified_eval/simplified"

if [ ! -f "$MILK_CSV" ]; then
    echo "FATAL: CSV not found at $MILK_CSV"; exit 1
fi
if [ ! -d "$MILK_IMG_DIR" ]; then
    echo "FATAL: image dir not found at $MILK_IMG_DIR"; exit 1
fi
N_IMG=$(ls "$MILK_IMG_DIR" | wc -l)
echo "CSV:        $MILK_CSV"
echo "Image dir:  $MILK_IMG_DIR ($N_IMG files)"

# -------------------------------------------------------------------------
# Build model flags (inline -- bash function form has been flaky)
# -------------------------------------------------------------------------
MODEL_FLAGS=""

F="outputs/isic2019_orig/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model orig:$F"
    echo "FOUND: orig"
else
    echo "MISSING: orig -> $F"
fi

F="outputs/isic2019_aug00/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda00:$F"
    echo "FOUND: lambda00"
else
    echo "MISSING: lambda00 -> $F"
fi

F="outputs/isic2019_aug_v2_03/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda03:$F"
    echo "FOUND: lambda03"
else
    echo "MISSING: lambda03 -> $F"
fi

F="outputs/isic2019_aug_v2_07/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda07:$F"
    echo "FOUND: lambda07"
else
    echo "MISSING: lambda07 -> $F"
fi

F="outputs/isic2019_aug_v2_10/best_efficientnet_b3_isic2019.pth"
if [ -f "$F" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --model lambda10:$F"
    echo "FOUND: lambda10"
else
    echo "MISSING: lambda10 -> $F"
fi

if [ -z "$MODEL_FLAGS" ]; then
    echo "ERROR: no checkpoints found"; exit 1
fi
echo ""

TAGS="orig lambda00 lambda03 lambda07 lambda10"

# =========================================================================
# Stage 1a: DIAGNOSIS3 mapping predictions
# =========================================================================
echo "==== Stage 1a: DIAGNOSIS3 predictions ===="
python generate_milk10k_predictions.py \
    --csv        "$MILK_CSV" \
    --image-dir  "$MILK_IMG_DIR" \
    --mapping    DIAGNOSIS3 \
    --output-dir "$PRED_DIR_D3" \
    --device     cuda \
    $MODEL_FLAGS

# =========================================================================
# Stage 1b: SIMPLIFIED mapping predictions
# =========================================================================
echo ""
echo "==== Stage 1b: SIMPLIFIED predictions ===="
python generate_milk10k_predictions.py \
    --csv        "$MILK_CSV" \
    --image-dir  "$MILK_IMG_DIR" \
    --mapping    SIMPLIFIED \
    --output-dir "$PRED_DIR_SIMP" \
    --device     cuda \
    $MODEL_FLAGS

# =========================================================================
# Stage 2a: Tests 2-5 on DIAGNOSIS3 predictions
# =========================================================================
echo ""
echo "==== Stage 2a: Tests 2-5 (DIAGNOSIS3) ===="
python evaluate_milk10k_stratified.py \
    --pred-dir       "$PRED_DIR_D3" \
    --tags          $TAGS \
    --baseline-tag   orig \
    --output-dir     "$EVAL_DIR_D3" \
    --bootstrap-iters 1000

# =========================================================================
# Stage 2b: Tests 2-5 on SIMPLIFIED predictions
# =========================================================================
echo ""
echo "==== Stage 2b: Tests 2-5 (SIMPLIFIED, Test 5d arm) ===="
python evaluate_milk10k_stratified.py \
    --pred-dir       "$PRED_DIR_SIMP" \
    --tags          $TAGS \
    --baseline-tag   orig \
    --output-dir     "$EVAL_DIR_SIMP" \
    --bootstrap-iters 1000

echo ""
echo "=== MILK10K stratified evaluation complete ==="
echo "Date: $(date)"
echo ""
echo "DIAGNOSIS3 artefacts:"
ls -lh "$EVAL_DIR_D3" 2>/dev/null
echo ""
echo "SIMPLIFIED artefacts:"
ls -lh "$EVAL_DIR_SIMP" 2>/dev/null
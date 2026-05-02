#!/bin/bash
#SBATCH --job-name=eval_test
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --error=logs/eval_test_%j.err

echo "=== ISIC 2019 Test Set Evaluation (Orig vs Aug00) ==="
cd ~/skin-cancer

# ── Environment ──
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

TEST_IMAGES="datasets/ISIC_2019_Test_Input"
TEST_GT="datasets/ISIC_2019_Test_GroundTruth.csv"
OUTPUT_DIR="outputs/isic2019_test_eval"
mkdir -p "$OUTPUT_DIR"

MODEL_ORIG="outputs/isic2019_orig/best_efficientnet_b3_isic2019.pth"
MODEL_AUG00="outputs/isic2019_aug00/best_efficientnet_b3_isic2019.pth"

MODEL_FLAGS=""
[ -f "$MODEL_ORIG"  ] && MODEL_FLAGS="$MODEL_FLAGS --model-orig $MODEL_ORIG"
# Repurposing the aug07 flag slot for aug00 to keep python args happy, or update python args if preferred.
[ -f "$MODEL_AUG00" ] && MODEL_FLAGS="$MODEL_FLAGS --model-aug07 $MODEL_AUG00" 

python evaluate_isic2019_test.py \
    --test-images "$TEST_IMAGES" \
    --test-gt "$TEST_GT" \
    $MODEL_FLAGS \
    --output-dir "$OUTPUT_DIR" \
    --device cuda
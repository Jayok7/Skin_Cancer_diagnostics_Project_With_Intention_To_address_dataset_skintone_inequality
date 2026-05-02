#!/bin/bash
#SBATCH --job-name=cnn_outputs
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/cnn_outputs_%j.out
#SBATCH --error=logs/cnn_outputs_%j.err

# ================================================================
# CNN Output Generation — EfficientNet-B3 + Grad-CAM
# ================================================================
# Generates classification probabilities and Grad-CAM heatmaps
# for every image in HAM10000. Output is consumed by the teacher
# generation script in the VLM fine-tuning pipeline.
#
# Prerequisites:
#   - Trained model at outputs/skin_cancer_diagnostics/
#   - HAM10000 images + metadata
# ================================================================

echo "=== CNN Output Generation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs training_data

# ── Environment ────────────────────────────────────────────
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

pip install "numpy<2.0" 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip install opencv-python-headless Pillow pandas 2>/dev/null

echo ""
nvidia-smi
echo ""

# ── Configuration ──────────────────────────────────────────
MODEL_PATH="outputs/skin_cancer_diagnostics/best_efficientnet_b3_ham10000.pth"
IMAGE_DIR="datasets/Ham10000/HAM10000_images_part_1"
METADATA="datasets/Ham10000/HAM10000_metadata.csv"
OUTPUT_DIR="training_data"

echo "  Model:    $MODEL_PATH"
echo "  Images:   $IMAGE_DIR"
echo "  Metadata: $METADATA"
echo "  Output:   $OUTPUT_DIR"

# Validate 
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$METADATA" ]; then
    echo "ERROR: Metadata CSV not found at $METADATA"
    exit 1
fi

# Run
python generate_cnn_outputs.py \
    --model-path "$MODEL_PATH" \
    --image-dir "$IMAGE_DIR" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"

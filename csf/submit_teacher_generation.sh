#!/bin/bash
#SBATCH --job-name=teacher_gen
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/teacher_gen_%j.out
#SBATCH --error=logs/teacher_gen_%j.err

# ================================================================
# Teacher Generation (FREE) — Qwen2.5-VL-7B-Instruct
# ================================================================
# Generates clinical reasoning using an open-source VLM in 4-bit
# quantisation. Consumes CNN outputs from generate_cnn_outputs.py
# and produces teacher reasoning for SFT training.
#
# Prerequisites:
#   - CNN outputs at training_data/cnn_outputs.json
#   - ~24GB GPU VRAM (4-bit quant of 7B model)
#   - HuggingFace model will be auto-downloaded on first run
# ================================================================

echo "=== Teacher Generation (FREE — Qwen2.5-VL) ==="
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

# Cache HuggingFace models locally
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"

# Upgrade numpy + rebuild scipy/sklearn C extensions for numpy 2.x compatibility
# (accelerate requires numpy 2.x; scipy/sklearn must match)
pip install "numpy>=2.0" "scipy>=1.14" "scikit-learn>=1.5" --upgrade --force-reinstall 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip install transformers accelerate bitsandbytes qwen-vl-utils Pillow --upgrade 2>/dev/null

# Diagnostic: verify compatible versions
python -c "import numpy; print(f'numpy={numpy.__version__}'); import scipy; print(f'scipy={scipy.__version__}'); import accelerate; print(f'accelerate={accelerate.__version__}')"

echo ""
nvidia-smi
echo ""

# ── Configuration ──────────────────────────────────────────
CNN_OUTPUTS="training_data/cnn_outputs.json"
OUTPUT="training_data/teacher_outputs_free.json"
MAX_SAMPLES=2500
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

echo "  CNN Outputs: $CNN_OUTPUTS"
echo "  Output:      $OUTPUT"
echo "  Max samples: $MAX_SAMPLES"
echo "  Model:       $MODEL"

# ── Validate ───────────────────────────────────────────────
if [ ! -f "$CNN_OUTPUTS" ]; then
    echo "ERROR: CNN outputs not found at $CNN_OUTPUTS"
    echo "  Run submit_generate_cnn_outputs.sh first."
    exit 1
fi

# ── Run ────────────────────────────────────────────────────
python teacher_generation_free.py \
    --cnn-outputs "$CNN_OUTPUTS" \
    --output "$OUTPUT" \
    --max-samples $MAX_SAMPLES \
    --model "$MODEL"

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"

#!/bin/bash
#SBATCH --job-name=vlm_sft
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/vlm_sft_%j.out
#SBATCH --error=logs/vlm_sft_%j.err

# ================================================================
# VLM Pipeline Step 5: SFT Training (QLoRA on Qwen3-VL)
# ================================================================
# Supervised fine-tuning using QLoRA 4-bit quantisation.
# Trains on formatted JSONL conversations from Step 4.
#
# Prerequisites:
#   - training_data/formatted/train.jsonl (from Step 4)
#   - ~24GB GPU VRAM (4-bit quant)
#
# After this job: submit submit_vlm_dpo.sh
# ================================================================

echo "=== VLM SFT Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs checkpoints/sft

# ── Environment ────────────────────────────────────────────
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HOME/.cache/huggingface"

pip install "numpy<2.0" 2>/dev/null
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null
pip install transformers accelerate bitsandbytes peft datasets 2>/dev/null
pip install -U "trl>=1.2.0" 2>/dev/null
pip install qwen-vl-utils 2>/dev/null

echo ""
nvidia-smi
echo ""

# ── Validate ───────────────────────────────────────────────
if [ ! -f "training_data/formatted/train.jsonl" ]; then
    echo "ERROR: Training data not found."
    echo "  Run submit_vlm_qc_format.sh first."
    exit 1
fi

echo "  Train data: $(wc -l < training_data/formatted/train.jsonl) examples"
echo "  Val data:   $(wc -l < training_data/formatted/validation.jsonl) examples"

# ── Run SFT ────────────────────────────────────────────────
python train_sft.py

echo ""
echo "=== SFT Training Complete ==="
echo "Next: sbatch submit_vlm_dpo.sh"
echo "Date: $(date)"

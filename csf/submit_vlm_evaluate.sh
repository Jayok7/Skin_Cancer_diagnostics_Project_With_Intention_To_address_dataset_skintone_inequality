#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/vlm_eval_%j.out
#SBATCH --error=logs/vlm_eval_%j.err

# ================================================================
# VLM Pipeline Step 8: Evaluate SFT vs DPO
# ================================================================
# Runs pairwise evaluation comparing SFT and DPO model outputs.
# Produces a win-rate report and per-metric breakdown.
#
# Prerequisites:
#   - SFT model at checkpoints/sft/final/
#   - DPO model at checkpoints/dpo/final/
#   - Test data at training_data/formatted/test.jsonl
# ================================================================

echo "=== VLM Evaluation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs

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
pip install transformers accelerate bitsandbytes peft trl datasets 2>/dev/null
pip install qwen-vl-utils rouge-score nltk 2>/dev/null

echo ""
nvidia-smi
echo ""

# ── Validate ───────────────────────────────────────────────
for d in "checkpoints/sft/final" "checkpoints/dpo/final"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Model not found at $d"
        exit 1
    fi
done

# ── Evaluate ───────────────────────────────────────────────
python evaluate.py

echo ""
echo "=== Evaluation Complete ==="
echo "Report: training_data/evaluation_report.json"
echo "Date: $(date)"

#!/bin/bash
#SBATCH --job-name=vlm_dpo
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/vlm_dpo_%j.out
#SBATCH --error=logs/vlm_dpo_%j.err

# ================================================================
# VLM Pipeline Steps 6+7: DPO Pair Generation + DPO Training
# ================================================================
# Step 6: generate_dpo_pairs.py — creates preference pairs from QC
# Step 7: train_dpo.py — DPO alignment training
#
# Prerequisites:
#   - SFT model at checkpoints/sft/final/ (from Step 5)
#   - test.jsonl for DPO pair generation
#
# After this job: submit submit_vlm_evaluate.sh
# ================================================================

echo "=== VLM DPO Pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs checkpoints/dpo

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
if [ ! -d "checkpoints/sft/final" ]; then
    echo "ERROR: SFT model not found at checkpoints/sft/final/"
    echo "  Run submit_vlm_sft.sh first."
    exit 1
fi

# ── Step 6: Generate DPO Pairs ──
echo ""
echo "═══ Step 6: Generate DPO Pairs ═══"
python generate_dpo_pairs.py

# ── Step 7: DPO Training ──
echo ""
echo "═══ Step 7: DPO Training ═══"
python train_dpo.py

echo ""
echo "=== DPO Pipeline Complete ==="
echo "Next: sbatch submit_vlm_evaluate.sh"
echo "Date: $(date)"

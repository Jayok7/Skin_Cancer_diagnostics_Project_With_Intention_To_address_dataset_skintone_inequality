#!/bin/bash
#SBATCH --job-name=isic2020_classify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/isic2020_classify_%j.out
#SBATCH --error=outputs/isic2020_classify_%j.err

# ═══════════════════════════════════════════════════════════════
# ISIC 2020 Skin Tone Classification - Slurm Submission
# ═══════════════════════════════════════════════════════════════
# Uses the 3-class fine-tuned model to label ~33k ISIC 2020 images.
#
# Usage:
#   sbatch submit_classify_isic2020.sh
# ═══════════════════════════════════════════════════════════════

echo "=== ISIC 2020 Classification Job ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ── Environment ──
module load anaconda3/2024.06
source activate skin_tone

# ── Paths (adjust as needed) ──
CHECKPOINT="outputs/FairFace-Model-3.2-finetuned-v5-3class/fairface_mskcc_best.pth"
METADATA="datasets/challenge-2020-training_metadata_2026-04-04.csv"
IMAGE_ROOT="datasets/ISIC-2020-images/"
OUTPUT="datasets/isic2020_skin_tone_predictions.csv"

echo "Checkpoint: $CHECKPOINT"
echo "Metadata:   $METADATA"
echo "Image root: $IMAGE_ROOT"
echo "Output:     $OUTPUT"
echo ""

# ── Run ──
python classify_isic2020.py \
    --checkpoint "$CHECKPOINT" \
    --metadata "$METADATA" \
    --image-root "$IMAGE_ROOT" \
    --output "$OUTPUT" \
    --batch-size 64 \
    --num-workers 4 \
    --num-classes 3

echo ""
echo "=== Job Complete ==="
echo "Date: $(date)"

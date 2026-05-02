#!/bin/bash
#SBATCH --job-name=vlm_qc_format
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/vlm_qc_format_%j.out
#SBATCH --error=logs/vlm_qc_format_%j.err

# ================================================================
# VLM Pipeline Steps 3+4: Quality Control + Dataset Formatting
# ================================================================
# CPU-only job. Runs after teacher generation completes.
#   Step 3: quality_control.py → filters teacher reasoning
#   Step 4: format_dataset.py  → converts to JSONL for SFT
#
# After this job: submit submit_vlm_sft.sh
# ================================================================

echo "=== VLM QC + Format Pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs training_data/formatted

module load apps/binapps/anaconda3/2021.11
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"

pip install "numpy<2.0" scikit-learn 2>/dev/null

#  Step 3: Quality Control  
echo ""
echo "═══ Step 3: Quality Control ═══"

TEACHER_INPUT="training_data/teacher_outputs_free.json"
QC_OUTPUT="training_data/clean_dataset.json"
QC_REPORT="training_data/qc_report.json"

if [ ! -f "$TEACHER_INPUT" ]; then
    echo "ERROR: Teacher outputs not found at $TEACHER_INPUT"
    echo "  Run submit_teacher_generation.sh first."
    exit 1
fi

python quality_control.py \
    --input "$TEACHER_INPUT" \
    --output "$QC_OUTPUT" \
    --report "$QC_REPORT"

if [ ! -f "$QC_OUTPUT" ]; then
    echo "ERROR: QC failed to produce output"
    exit 1
fi

echo "  QC report: $QC_REPORT"

#  Step 4: Format Dataset 
echo ""
echo "═══ Step 4: Format Dataset ═══"

python format_dataset.py

echo ""
echo "=== QC + Format Complete ==="
echo "Next: sbatch submit_vlm_sft.sh"
echo "Date: $(date)"

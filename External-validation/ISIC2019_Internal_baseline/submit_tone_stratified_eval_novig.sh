#!/bin/bash
#SBATCH --job-name=tone_strat_novig
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/tone_eval_novig_%j.out
#SBATCH --error=logs/tone_eval_novig_%j.err

# =========================================================================
# Vignette-stratified sensitivity arm of Tests 2-5.
# Re-runs the full evaluation with vignette-flagged images excluded,
# to test whether the tone-fairness conclusions survive when the
# site/equipment confound is removed.
# =========================================================================

echo "=== Vignette-stratified Tone Evaluation ==="
echo "Date: $(date) | Node: $(hostname) | Job ID: $SLURM_JOB_ID"

cd ~/skin-cancer
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

pip install "numpy<2.0" --quiet 2>/dev/null
pip install statsmodels --quiet 2>/dev/null

TONE_CSV="outputs/isic2019_test_tone_audit_cascade/skin_tone_predictions.csv"
GT_CSV="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/skin-cancer/datasets/ISIC_2019_Test_GroundTruth.csv"
PRED_DIR="outputs/diagnostic_predictions"
META_CSV="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/skin-cancer/datasets/ISIC_2019_Test_Input/ISIC_2019_Test_Metadata.csv"
OUT_DIR="outputs/tone_stratified_eval"

# Run the no-vignette sensitivity at conf>=0.0 (where Dark N is largest)
echo ""
echo "==== NO-VIGNETTE SENSITIVITY (tone conf>=0.0, vignette-flagged excluded) ===="
python evaluate_tone_stratified.py \
    --tone-csv     "$TONE_CSV" \
    --gt-csv       "$GT_CSV" \
    --pred-dir     "$PRED_DIR" \
    --metadata-csv "$META_CSV" \
    --output-dir   "$OUT_DIR/conf000" \
    --tone-conf 0.0 \
    --tone-conf-strict 0.8 \
    --bootstrap-iters 1000 \
    --model-classes "AK,BCC,BKL,DF,MEL,NV,SCC,UNK,VASC" \
    --eval-classes  "AK,BCC,BKL,DF,MEL,NV,SCC,VASC" \
    --tags orig lambda00 lambda03 lambda07 lambda10 \
    --baseline-tag orig \
    --require-no-vignette

# Also at conf>=0.5 for additional coverage on Light/Medium
echo ""
echo "==== NO-VIGNETTE SENSITIVITY (tone conf>=0.5, vignette-flagged excluded) ===="
python evaluate_tone_stratified.py \
    --tone-csv     "$TONE_CSV" \
    --gt-csv       "$GT_CSV" \
    --pred-dir     "$PRED_DIR" \
    --metadata-csv "$META_CSV" \
    --output-dir   "$OUT_DIR/conf050" \
    --tone-conf 0.5 \
    --tone-conf-strict 0.8 \
    --bootstrap-iters 1000 \
    --model-classes "AK,BCC,BKL,DF,MEL,NV,SCC,UNK,VASC" \
    --eval-classes  "AK,BCC,BKL,DF,MEL,NV,SCC,VASC" \
    --tags orig lambda00 lambda03 lambda07 lambda10 \
    --baseline-tag orig \
    --require-no-vignette

echo ""
echo "=== No-vignette sensitivity complete ==="
echo "Date: $(date)"
ls -lh "$OUT_DIR/conf000/no_vignette/" "$OUT_DIR/conf050/no_vignette/" 2>/dev/null
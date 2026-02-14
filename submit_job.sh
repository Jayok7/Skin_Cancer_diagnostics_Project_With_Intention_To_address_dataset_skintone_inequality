#!/bin/bash
#SBATCH --job-name=fitzpatrick_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/fitzpatrick_%j.log
#SBATCH --error=logs/fitzpatrick_%j.err

# Load required modules
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

# Set environment path and add to PATH
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

# Verify we're using the correct Python
echo "Python location: $(which python)"
python --version
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "=========================================="

# Print GPU information
nvidia-smi

# Create output directories
mkdir -p outputs
mkdir -p logs

# Run training script
python train_fitzpatrick.py \
    --data-dir ./datasets/MSKCC-images \
    --csv-path ./datasets/mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv \
    --output-dir ./outputs \
    --use-3way \
    --batch-size 8 \
    --epochs-head 15 \
    --epochs-finetune 60 \
    --image-size 260

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

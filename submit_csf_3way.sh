#!/bin/bash
#SBATCH --job-name=fairface_3way
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/fairface_3way_%j.log
#SBATCH --error=logs/fairface_3way_%j.err

# Load required modules
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

# Set environment path and add to PATH (Manual activation for robustness)
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

# Create output directories (specific for 3-way)
mkdir -p outputs/fairface_3way
mkdir -p logs

# Install missing dependencies for visualization and metrics
# (Including tqdm and Pillow for safety)
pip install matplotlib seaborn scikit-learn tqdm Pillow
# Install PyTorch (headless)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Note: Ensure paths to CSV and images are correct relative to where you submit from
python train_fairface.py \
    --data-csv datasets/fairface_lstar_labels.csv \
    --image-root datasets/fairface-img-margin025-trainval/ \
    --output-dir outputs/fairface_3way \
    --batch-size 32 \
    --epochs-head 10 \
    --epochs-finetune 40 \
    --num-classes 3

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=fairface_label
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --output=logs/labeling_%j.log
#SBATCH --error=logs/labeling_%j.err

# Load modules
module load apps/binapps/anaconda3/2021.11

# Set up environment (same as training)
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"

# Run labeling script
# Install missing OpenCV dependency (headless for server)
pip install opencv-python-headless


python compute_fairface_labels.py \
    --csv-dir datasets/ \
    --image-root datasets/fairface-img-margin025-trainval/ \
    --output datasets/fairface_lstar_labels.csv

echo "Labeling complete."

 #!/bin/bash
#SBATCH --job-name=fairface_label
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/labeling_%j.log
#SBATCH --error=logs/labeling_%j.err
#SBATCH --partition=short

# Load modules
module load apps/binapps/anaconda3/2021.11

# Set up environment (same as training)
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"

# Run labeling script
echo "Starting labeling with MSKCC-aligned centroids..."

python compute_fairface_labels.py \
    --csv-dir datasets/ \
    --image-root datasets/fairface-img-margin025-trainval/ \
    --output datasets/fairface_lstar_labels.csv

echo "Labeling complete."

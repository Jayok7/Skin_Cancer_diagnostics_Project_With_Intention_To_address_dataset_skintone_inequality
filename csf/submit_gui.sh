#!/bin/bash
#SBATCH --job-name=derma-gui
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gui_%j.log

# Load modules and environment
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
ENV_PATH="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/.conda/envs/tf_gpu"
export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

# Make sure we have the VLM packages installed in your conda env
pip install transformers qwen-vl-utils accelerate

echo "=========================================================="
echo "GUI Server running on Compute Node: $(hostname)"
echo "=========================================================="

# Run the Flask app
python app.py
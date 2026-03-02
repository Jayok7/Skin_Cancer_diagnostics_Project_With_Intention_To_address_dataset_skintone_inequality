#!/bin/bash
#SBATCH --job-name=umap-plot
#SBATCH --partition=multicore
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/umap_plot_%j.log

# =========================================================================
# UMAP Plot from Embeddings — no GPU needed, just numpy/matplotlib/umap
# =========================================================================

cd ~/skin-cancer
mkdir -p logs

# Fix numpy to match system matplotlib (compiled with numpy 1.x)
pip install --force-reinstall "numpy==1.26.4"
pip install umap-learn scikit-learn

# Plot from existing embeddings
python visualise_latent_space.py \
    --embeddings outputs/FairFace-Model-2.4/embeddings.npz \
    --output outputs/FairFace-Model-2.4/fairface_latent_space.png

echo "Done — $(date)"

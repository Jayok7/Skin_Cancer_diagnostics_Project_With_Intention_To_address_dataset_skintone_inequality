#!/usr/bin/env python3
"""
Latent Space Visualisation — Local Script (memory-safe)
=======================================================
Loads embeddings from .npz, subsamples if needed, and produces UMAP plot.

Usage:
    python visualise_latent_space.py \
        --embeddings outputs/FairFace-Model-2.4/embeddings.npz \
        --output outputs/FairFace-Model-2.4/fairface_latent_space.png
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_latent_space(embeddings, labels, class_names, output_path, max_samples=3000):
    """Subsample, reduce to 2D with UMAP/t-SNE, and plot."""

    # Subsample for memory — stratified by class
    if len(labels) > max_samples:
        print(f"  Subsampling {len(labels):,} -> {max_samples:,} (stratified)...")
        rng = np.random.RandomState(42)
        indices = []
        per_class = max_samples // len(class_names)
        for c in range(len(class_names)):
            cls_idx = np.where(labels == c)[0]
            n = min(per_class, len(cls_idx))
            indices.extend(rng.choice(cls_idx, n, replace=False))
        indices = np.array(indices)
        rng.shuffle(indices)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Keep as float32 to save memory
    embeddings = embeddings.astype(np.float32)

    # PCA pre-reduction (in-place friendly)
    print(f"  Running incremental PCA (-> 50 dims)...")
    from sklearn.decomposition import IncrementalPCA
    pca = IncrementalPCA(n_components=50, batch_size=500)
    embeddings = pca.fit_transform(embeddings)

    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                       metric="cosine", random_state=42, low_memory=True)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42,
                       n_iter=1000)
        method = "t-SNE"

    print(f"  Running {method} on {embeddings.shape[0]:,} samples...")
    coords = reducer.fit_transform(embeddings)

    # Skin-tone palette (dark -> light)
    skin_colors = [
        "#3B2219",  # Type VI
        "#6B4226",  # Type V
        "#A67B5B",  # Type IV
        "#C8A882",  # Type III
        "#E8C9A0",  # Type II
        "#F5DEB3",  # Type I
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(len(class_names)):
        mask = labels == i
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=skin_colors[i], label=str(class_names[i]),
                   alpha=0.5, s=12, edgecolors="none")

    ax.set_title(f"Latent Space ({method}) - FairFace Skin Tone Classifier",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(markerscale=3, fontsize=10, framealpha=0.9, edgecolor="gray")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True,
                   help="Path to .npz file from extract_embeddings.py")
    p.add_argument("--output", default="fairface_latent_space.png")
    p.add_argument("--max-samples", type=int, default=3000,
                   help="Max samples to plot (default 3000, saves memory)")
    args = p.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"]
    class_names = list(data["class_names"])

    print(f"Loaded {embeddings.shape[0]:,} embeddings ({embeddings.shape[1]} dims)")
    print(f"Classes: {class_names}")

    plot_latent_space(embeddings, labels, class_names, args.output,
                      max_samples=args.max_samples)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MSKCC PyTorch Dataset for Fine-Tuning
=======================================
Loads preprocessed MSKCC crops with MST-5 labels for fine-tuning
the FairFace skin tone classifier.

Reuses transform factories from fairface_dataset.py for consistency.

Stratified sampling strategy for extreme class imbalance:
─────────────────────────────────────────────────────────
The MSKCC training set is heavily skewed:
  Very Light:  239 (35%)   ← majority
  Light:       186 (27%)
  Medium:      204 (30%)
  Dark:         29 ( 4%)   ← severe minority
  Very Dark:    22 ( 3%)   ← severe minority

Naive inverse-frequency sampling would oversample Very Dark ~11x per epoch,
causing memorisation of those 22 images. Instead we use:

  1. √-balanced sampling:  weight_i = 1/√count_i  (not 1/count_i)
     → Very Dark oversampled ~3.3x instead of ~10.9x
  2. Capped oversampling:  max 5x repetition factor per class per epoch
  3. 3-tier augmentation:
     - Standard:         Medium, Light, Very Light  (≥15% of data)
     - Minority:         classes <15% but ≥5%       (stronger color jitter)
     - Severe minority:  classes <5% of data         (heavy aug + larger crop
                          pool + more erasing to maximise visual diversity)
"""

import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from fairface_dataset import (
    get_train_transforms,
    get_minority_train_transforms,
    get_val_transforms,
)


MST5_NAMES = [
    "Very Dark (MST 9-10)",    # class 0
    "Dark (MST 7-8)",          # class 1
    "Medium (MST 5-6)",        # class 2
    "Light (MST 3-4)",         # class 3
    "Very Light (MST 1-2)",    # class 4
]

MST3_NAMES = [
    "Dark (MST 7-10)",         # class 0  ← merged Very Dark + Dark
    "Medium (MST 3-6)",        # class 1  ← merged Medium + Light
    "Light (MST 1-2)",         # class 2  ← Very Light
]

# Mapping: mst5_class → mst3_class
MST5_TO_MST3 = {
    0: 0,   # Very Dark → Dark
    1: 0,   # Dark      → Dark
    2: 1,   # Medium    → Medium
    3: 1,   # Light     → Medium
    4: 2,   # Very Light → Light
}


# ── Additional transform tier for severe minorities ──────────

def get_severe_minority_transforms(image_size: int = 380):
    """
    Heavy augmentation for classes with <5% of training data.

    Key differences from standard minority transforms:
    - Larger crop pool (+48px vs +40px) -> more spatial diversity
    - Heavier color jitter (0.4 brightness/contrast) -> more appearance variety
    - Higher RandomErasing probability (0.25 vs 0.20) -> forces invariance
    - Random rotation up to 20 deg (vs 15) -- less concern about face warping
      since these are lesion crops, not portraits
    - RandomPerspective for additional geometric variety
    """
    return transforms.Compose([
        transforms.Resize(image_size + 48),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=20, translate=(0.08, 0.08), scale=(0.90, 1.10)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.10,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20)),
    ])


class MSKCCDataset(Dataset):
    """
    PyTorch Dataset for MSKCC preprocessed crops.

    Supports both 5-class (MST-5) and 3-class (MST-3) modes.
    In 3-class mode, the 5 MST bins are merged:
        Dark    = Very Dark + Dark    (MST 7-10)
        Medium  = Medium + Light      (MST 3-6)
        Light   = Very Light           (MST 1-2)

    Args:
        csv_path:    Path to mskcc_mst5_labels.csv (must have crop_file column)
        crop_root:   Directory containing preprocessed crops
        split:       'train', 'val', 'test', or 'all'
        merge_classes: 5 for MST-5, 3 for merged MST-3 (default: 5)
        transform:   Custom transform pipeline (if None, uses defaults)
        image_size:  Image size for default transforms (380 for EfficientNet-B4)
        severe_minority_threshold: Classes below this fraction get tier-3 augmentation
        minority_threshold: Classes below this fraction get tier-2 augmentation
    """

    def __init__(
        self,
        csv_path: str,
        crop_root: str,
        split: str = "all",
        merge_classes: int = 5,
        transform=None,
        image_size: int = 380,
        minority_threshold: float = 0.15,
        severe_minority_threshold: float = 0.05,
    ):
        self.crop_root = crop_root
        self.df = pd.read_csv(csv_path)

        # Filter by split
        if split != "all":
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # Ensure crop_file column exists
        if "crop_file" not in self.df.columns:
            warnings.warn("crop_file column not found -- constructing from isic_id. "
                          "Run preprocess_mskcc_crops.py first for best results.")
            self.df["crop_file"] = self.df["isic_id"].apply(lambda x: f"{x}_crop.jpg")

        # Drop rows with missing crop files
        self.df = self.df.dropna(subset=["crop_file", "mst5_class"]).reset_index(drop=True)

        # Validate a sample path
        if len(self.df) > 0:
            sample = os.path.join(crop_root, self.df.iloc[0]["crop_file"])
            if not os.path.isfile(sample):
                alt_sample = os.path.join(crop_root, f"{self.df.iloc[0]['isic_id']}.jpg")
                if os.path.isfile(alt_sample):
                    warnings.warn(f"Crop not found at {sample}, using original images")
                    self.df["crop_file"] = self.df["isic_id"].apply(lambda x: f"{x}.jpg")
                else:
                    warnings.warn(f"Sample image not found: {sample}")

        # ── Class mode: 5-class or merged 3-class ────────────
        self._merge_mode = merge_classes
        raw_labels = self.df["mst5_class"].values.astype(int)

        if merge_classes == 3:
            self.class_names = MST3_NAMES
            self.num_classes = 3
            self._labels = np.array([MST5_TO_MST3[c] for c in raw_labels])
        else:
            self.class_names = MST5_NAMES
            self.num_classes = 5
            self._labels = raw_labels

        # ── Classify each class into augmentation tiers ──────
        counts = np.bincount(self._labels, minlength=self.num_classes)
        total = len(self._labels)
        fractions = counts / total if total > 0 else np.zeros(self.num_classes)

        self._severe_minority_classes = set(
            int(c) for c in range(self.num_classes) if fractions[c] < severe_minority_threshold
        )
        self._minority_classes = set(
            int(c) for c in range(self.num_classes)
            if fractions[c] < minority_threshold and c not in self._severe_minority_classes
        )

        # ── Assign transforms (3-tier system) ────────────────
        if transform is not None:
            self.transform = transform
            self._minority_transform = transform
            self._severe_minority_transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
            self._minority_transform = get_minority_train_transforms(image_size)
            self._severe_minority_transform = get_severe_minority_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)
            self._minority_transform = self.transform
            self._severe_minority_transform = self.transform

        # ── Log info ─────────────────────────────────────────
        print(f"  MSKCCDataset [{split}]: {len(self.df):,} images, "
              f"{self.num_classes} classes")
        for c in range(self.num_classes):
            n = int(counts[c])
            pct = fractions[c] * 100
            if c in self._severe_minority_classes:
                tag = " <-- severe minority (tier-3 aug)"
            elif c in self._minority_classes:
                tag = " <-- minority (tier-2 aug)"
            else:
                tag = ""
            print(f"    {self.class_names[c]}: {n:4d} ({pct:5.1f}%){tag}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop_path = os.path.join(self.crop_root, row["crop_file"])
        label = int(self._labels[idx])  # use remapped labels (supports 3-class mode)

        try:
            img = Image.open(crop_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load {crop_path}: {e}")
            img = Image.new("RGB", (380, 380), (0, 0, 0))

        # 3-tier class-aware augmentation
        if label in self._severe_minority_classes:
            img = self._severe_minority_transform(img)
        elif label in self._minority_classes:
            img = self._minority_transform(img)
        else:
            img = self.transform(img)

        return img, label

    def get_sample_weights(self, strategy="temperature", temperature=0.5, max_oversample=5.0):
        """
        Compute per-sample weights for WeightedRandomSampler.

        Strategies:
          'inverse':      weight_i = 1 / count_i
                          Full inverse-frequency. Oversamples Very Dark ~11x.

          'sqrt':         weight_i = 1 / sqrt(count_i)
                          Moderate rebalancing but still heavy oversampling.

          'temperature':  weight_i = (1/count_i)^temperature     [DEFAULT]
                          temperature=0 → uniform (no rebalancing)
                          temperature=0.5 → moderate (~3x for minorities)
                          temperature=1.0 → full inverse-frequency (~11x)

          'capped':       weight_i = 1 / count_i, capped at max_oversample.

        Args:
            strategy:       'inverse', 'sqrt', 'temperature', or 'capped'
            temperature:    exponent for temperature strategy (default: 0.5)
            max_oversample: max repetition factor per epoch (for 'capped')

        Returns:
            torch.FloatTensor of per-sample weights
        """
        counts = np.bincount(self._labels, minlength=self.num_classes).astype(float)
        counts = np.maximum(counts, 1)  # avoid div-by-zero

        if strategy == "inverse":
            class_weights = 1.0 / counts

        elif strategy == "sqrt":
            class_weights = 1.0 / np.sqrt(counts)

        elif strategy == "temperature":
            # w_i = (1/n_i)^t  — smoothly interpolates between uniform and inverse
            class_weights = np.power(1.0 / counts, temperature)

        elif strategy == "capped":
            class_weights = 1.0 / counts
            max_count = counts.max()
            for c in range(self.num_classes):
                target_samples = class_weights[c] * max_count
                effective_oversample = target_samples / max(counts[c], 1)
                if effective_oversample > max_oversample:
                    class_weights[c] = max_oversample * counts[c] / max_count
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Normalise so weights sum to 1 (convention)
        class_weights /= class_weights.sum()

        # Per-sample weight
        weights = class_weights[self._labels]

        # Log effective oversampling ratios
        natural_frac = counts / counts.sum()
        weighted_frac = class_weights / class_weights.sum()
        print(f"\n  Sampling strategy: {strategy}")
        print(f"  {'Class':<28s}  {'Natural':>8s}  {'Sampled':>8s}  {'Oversample':>10s}")
        for c in range(self.num_classes):
            ratio = weighted_frac[c] / natural_frac[c] if natural_frac[c] > 0 else 0
            print(f"    {self.class_names[c]:<26s}  {natural_frac[c]:7.1%}  "
                  f"{weighted_frac[c]:7.1%}  {ratio:9.1f}x")

        return torch.from_numpy(weights).float()

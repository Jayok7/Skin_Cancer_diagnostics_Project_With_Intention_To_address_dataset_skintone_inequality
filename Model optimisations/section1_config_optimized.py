# ========================================================================
# SECTION 1: CONFIGURATION (OPTIMIZED FOR FITZPATRICK CLASSIFICATION)
# ========================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ========================================================================
# KEY CONFIGURATION: TOGGLE BETWEEN 3-WAY AND 6-WAY CLASSIFICATION
# ========================================================================
USE_3WAY_CLASSIFICATION = True  # Set to False for 6-way classification

# Paths
BASE_PATH = r'D:\skin cancer project\datasets\MSKCC-images'
CSV_PATH = os.path.join(r'D:\skin cancer project\datasets', 'mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv')

# Image settings
IMAGE_SIZE = 260
BATCH_SIZE = 8  # Reduced for small dataset

# Training epochs (increased for small dataset)
FEATURE_EXTRACTION_EPOCHS = 15
FINE_TUNING_EPOCHS = 60

# Learning rates
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE_INITIAL = 5e-6  # Reduced for stability

# Regularization
DROPOUT_RATE = 0.5  # Increased from 0.4
LABEL_SMOOTHING = 0.1

# Focal loss parameters (for handling imbalance)
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

AUTOTUNE = tf.data.AUTOTUNE

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled:", tf.keras.mixed_precision.global_policy().name)

print(f"\n{'='*60}")
print(f"CONFIGURATION: {'3-WAY' if USE_3WAY_CLASSIFICATION else '6-WAY'} FITZPATRICK CLASSIFICATION")
print(f"{'='*60}")

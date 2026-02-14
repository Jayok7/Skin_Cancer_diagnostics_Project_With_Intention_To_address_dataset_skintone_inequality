#!/usr/bin/env python3
"""
MSKCC Fitzpatrick Classifier - Training Script for GPU Cluster
Converted from Jupyter notebook for CSF/HPC execution
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ========================================================================
# CONFIGURATION
# ========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Fitzpatrick Skin Type Classifier')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to MSKCC-images directory')
    parser.add_argument('--csv-path', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory for models and plots')
    parser.add_argument('--use-3way', action='store_true', help='Use 3-way classification (Light/Medium/Dark)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs-head', type=int, default=15, help='Epochs for head training')
    parser.add_argument('--epochs-finetune', type=int, default=60, help='Epochs for fine-tuning')
    parser.add_argument('--image-size', type=int, default=260, help='Image size')
    return parser.parse_args()

# ========================================================================
# FOCAL LOSS
# ========================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal_loss_val = alpha * focal_weight * ce
        return tf.reduce_sum(focal_loss_val, axis=-1)
    return loss_fn

# ========================================================================
# DATA LOADING
# ========================================================================

def group_fitzpatrick(skin_type):
    """Group Fitzpatrick types into 3 broader categories"""
    if skin_type in ['I', 'II']:
        return 'Light'
    elif skin_type in ['III', 'IV']:
        return 'Medium'
    else:  # V, VI
        return 'Dark'

def load_and_prepare_data(csv_path, base_path, use_3way):
    print("\n--- Loading data ---")
    df = pd.read_csv(csv_path)
    df['path'] = df['isic_id'].apply(lambda x: os.path.join(base_path, x + '.jpg'))
    df = df.dropna(subset=['fitzpatrick_skin_type', 'path'])
    
    if use_3way:
        df['target_label'] = df['fitzpatrick_skin_type'].apply(group_fitzpatrick)
        print("✓ Using 3-way classification")
    else:
        df['target_label'] = df['fitzpatrick_skin_type']
        print("✓ Using 6-way classification")
    
    label_columns = sorted(df['target_label'].unique())
    df = pd.concat([df, pd.get_dummies(df['target_label'], dtype='float32')], axis=1)
    
    return df, label_columns

def patient_based_split(df):
    print("\n--- Patient-based split ---")
    unique_patients = df['patient_id'].unique()
    
    train_patients, temp_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
    val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=42)
    
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    validation_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"Train: {len(train_patients)} patients, {len(train_df)} images")
    print(f"Val: {len(val_patients)} patients, {len(validation_df)} images")
    print(f"Test: {len(test_patients)} patients, {len(test_df)} images")
    
    return train_df, validation_df, test_df, train_patients, val_patients, test_patients

# ========================================================================
# DATA PIPELINE
# ========================================================================

def parse_image(filepath, label, image_size):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return image, label

def skin_tone_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

def build_dataset(df, label_columns, image_size, batch_size, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((df['path'].values, df[label_columns].values))
    if is_training:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.map(lambda x, y: parse_image(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.map(skin_tone_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# ========================================================================
# MAIN TRAINING
# ========================================================================

def main():
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Enable mixed precision
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Load data
    df, label_columns = load_and_prepare_data(args.csv_path, args.data_dir, args.use_3way)
    train_df, val_df, test_df, train_patients, val_patients, test_patients = patient_based_split(df)
    
    # Compute class weights
    train_labels = train_df['target_label'].values
    unique_classes = np.unique(train_labels)
    class_weights_array = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
    label_to_index = {label: idx for idx, label in enumerate(label_columns)}
    class_weight_dict = {label_to_index[cls]: weight for cls, weight in zip(unique_classes, class_weights_array)}
    
    print("\nClass weights:", class_weight_dict)
    
    # Build datasets
    train_dataset = build_dataset(train_df, label_columns, args.image_size, args.batch_size, is_training=True)
    val_dataset = build_dataset(val_df, label_columns, args.image_size, args.batch_size, is_training=False)
    
    # Build model
    print("\n--- Building model ---")
    base_model = EfficientNetV2B2(include_top=False, weights='imagenet', 
                                   input_shape=(args.image_size, args.image_size, 3))
    base_model.trainable = False
    
    inputs = Input(shape=(args.image_size, args.image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(label_columns), activation='softmax', dtype='float32')(x)
    model = Model(inputs, outputs)
    
    # Stage 1: Train head
    print("\n--- Stage 1: Training head ---")
    model.compile(optimizer=AdamW(learning_rate=1e-3),
                  loss=focal_loss(gamma=2.0, alpha=0.25),
                  metrics=['accuracy'])
    
    checkpoint_head = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_head_model.keras'),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    
    history_head = model.fit(
        train_dataset,
        epochs=args.epochs_head,
        validation_data=val_dataset,
        callbacks=[checkpoint_head],
        class_weight=class_weight_dict
    )
    
    model.load_weights(os.path.join(args.output_dir, 'best_head_model.keras'))
    
    # Stage 2: Fine-tune
    print("\n--- Stage 2: Fine-tuning ---")
    base_model.trainable = True
    
    steps_per_epoch = len(train_df) // args.batch_size
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=5e-6,
        first_decay_steps=steps_per_epoch * 10,
        t_mul=2.0, m_mul=0.9
    )
    
    model.compile(optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
                  loss=focal_loss(gamma=2.0, alpha=0.25),
                  metrics=['accuracy'])
    
    checkpoint_ft = ModelCheckpoint(
        os.path.join(args.output_dir, 'best_finetuned_model.keras'),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    
    history_ft = model.fit(
        train_dataset,
        epochs=args.epochs_head + args.epochs_finetune,
        initial_epoch=args.epochs_head,
        validation_data=val_dataset,
        callbacks=[early_stop, checkpoint_ft],
        class_weight=class_weight_dict
    )
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'final_model.keras'))
    print(f"\n✓ Training complete! Models saved to {args.output_dir}")

if __name__ == '__main__':
    main()

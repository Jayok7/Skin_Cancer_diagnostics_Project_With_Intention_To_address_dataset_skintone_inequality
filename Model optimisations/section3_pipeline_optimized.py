# ========================================================================
# SECTION 3: OPTIMIZED DATA PIPELINE (SKIN-TONE FOCUSED AUGMENTATION)
# ========================================================================

print(f"\n--- Building tf.data pipelines (IMAGE_SIZE={IMAGE_SIZE})...")

def parse_image(filepath, label):
    """Load and resize image"""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.BILINEAR)
    return image, label

def skin_tone_augment(image, label):
    """
    Skin-tone-specific augmentation.
    Focus on COLOR variations (critical for skin classification).
    NO CutMix - it creates unrealistic mixed skin tones!
    """
    # Spatial augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Color augmentation (MORE IMPORTANT for skin tone classification)
    image = tf.image.random_hue(image, max_delta=0.05)  # Subtle hue variations
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)  # Saturation changes
    image = tf.image.random_brightness(image, max_delta=0.15)  # Lighting conditions
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Contrast variations
    
    return image, label

def build_dataset(df, is_training=True):
    """Build tf.data pipeline"""
    ds = tf.data.Dataset.from_tensor_slices((df['path'].values, df[label_columns].values))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(df))
    
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        # Apply skin-tone-specific augmentation (NO CutMix!)
        ds = ds.map(skin_tone_augment, num_parallel_calls=AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
    return ds

train_dataset = build_dataset(train_df, is_training=True)
validation_dataset = build_dataset(validation_df, is_training=False)
test_dataset = build_dataset(test_df, is_training=False)

print("✓ Data pipelines created (with skin-tone-focused augmentation)")
print("✓ CutMix removed (prevents unrealistic skin tone mixing)")

# ========================================================================
# SECTION 4: MODEL BUILDING (UNCHANGED - ALREADY OPTIMAL)
# ========================================================================

print(f"\n--- Building transfer learning model with EfficientNetV2B2...")

base_model = EfficientNetV2B2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
base_model.trainable = False

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)  # Increased dropout
outputs = Dense(len(label_columns), activation='softmax', dtype='float32')(x)

model = Model(inputs, outputs)
model.summary()

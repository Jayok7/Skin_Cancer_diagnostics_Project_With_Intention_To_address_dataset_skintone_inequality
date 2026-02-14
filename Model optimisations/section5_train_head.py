# ========================================================================
# SECTION 5: STAGE 1 - TRAINING HEAD WITH FOCAL LOSS & CLASS WEIGHTS
# ========================================================================

print(f"\n--- STAGE 1: Training the classification head (with focal loss) ---")

model.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE_HEAD),
    loss=focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA),
    metrics=['accuracy']
)

checkpoint_head = ModelCheckpoint(
    'best_head_model_fitzpatrick.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

history_head = model.fit(
    train_dataset, 
    epochs=FEATURE_EXTRACTION_EPOCHS,
    validation_data=validation_dataset, 
    callbacks=[checkpoint_head],
    class_weight=class_weight_dict  # Apply class weights!
)

model.load_weights('best_head_model_fitzpatrick.keras')
print("✓ Best head model loaded")

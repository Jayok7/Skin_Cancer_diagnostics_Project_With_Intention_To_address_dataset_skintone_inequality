# ========================================================================
# SECTION 6: STAGE 2 - FINE-TUNING WITH LOWER LEARNING RATE
# ========================================================================

print(f"\n--- STAGE 2: Fine-tuning the entire model ---")

# Load best head weights
try:
    model.load_weights('best_head_model_fitzpatrick.keras')
    print("Successfully loaded best head model weights.")
except Exception as e:
    print(f"Error loading best head model weights: {e}")

base_model.trainable = True  # Unfreeze base model

# Recompile with lower learning rate
steps_per_epoch_ft = len(train_df) // BATCH_SIZE
lr_schedule_ft = CosineDecayRestarts(
    initial_learning_rate=LEARNING_RATE_FINE_TUNE_INITIAL,
    first_decay_steps=steps_per_epoch_ft * 10, 
    t_mul=2.0, 
    m_mul=0.9
)

optimizer_ft = AdamW(learning_rate=lr_schedule_ft, weight_decay=1e-5)

model.compile(
    optimizer=optimizer_ft,
    loss=focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA),
    metrics=['accuracy']
)

checkpoint_ft = ModelCheckpoint(
    'best_fine_tuned_model_fitzpatrick.keras', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stop_ft = EarlyStopping(
    monitor='val_loss', 
    patience=15,  # Increased patience for small dataset
    verbose=1
)

history_fine_tune = model.fit(
    train_dataset,
    epochs=FEATURE_EXTRACTION_EPOCHS + FINE_TUNING_EPOCHS,
    initial_epoch=FEATURE_EXTRACTION_EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stop_ft, checkpoint_ft],
    class_weight=class_weight_dict  # Continue using class weights
)

model.load_weights('best_fine_tuned_model_fitzpatrick.keras')
print("✓ Best fine-tuned model loaded")

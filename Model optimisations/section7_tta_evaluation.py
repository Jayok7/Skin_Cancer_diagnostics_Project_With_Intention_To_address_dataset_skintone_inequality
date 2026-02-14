# ========================================================================
# SECTION 7: VISUALIZATION & TTA EVALUATION
# ========================================================================
def plot_history(history_head, history_fine_tune):
    acc = history_head.history['accuracy'] + history_fine_tune.history['accuracy']
    val_acc = history_head.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
    loss = history_head.history['loss'] + history_fine_tune.history['loss']
    val_loss = history_head.history['val_loss'] + history_fine_tune.history['val_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1); plt.plot(epochs_range, acc, label='Training Accuracy'); plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history_head.epoch)-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='lower right'); plt.title('Training and Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2); plt.plot(epochs_range, loss, label='Training Loss'); plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history_head.epoch)-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='upper right'); plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()

print("\n--- Plotting combined training history...")
plot_history(history_head, history_fine_tune)

def predict_with_tta(model, image_path, target_size):
    img = load_img(image_path, target_size=target_size); img_array = img_to_array(img)
    img_array_expanded = tf.expand_dims(img_array, axis=0)
    flipped_img_array = tf.image.flip_left_right(img_array_expanded)
    pred_original = model.predict(img_array_expanded, verbose=0)
    pred_flipped = model.predict(flipped_img_array, verbose=0)
    return np.mean([pred_original, pred_flipped], axis=0)[0]

# Evaluate on VALIDATION set with TTA
print("\n--- Evaluating on VALIDATION SET with Test-Time Augmentation (TTA) ---")
correct_tta_predictions_val = 0
true_labels_val = validation_df[label_columns].values
for i, (_, row) in enumerate(tqdm(validation_df.iterrows(), total=len(validation_df), desc="Validation TTA")):
    tta_pred = predict_with_tta(model, row['path'], (IMAGE_SIZE, IMAGE_SIZE))
    if np.argmax(tta_pred) == np.argmax(true_labels_val[i]): 
        correct_tta_predictions_val += 1

tta_accuracy_val = correct_tta_predictions_val / len(validation_df)

# Evaluate on TEST set with TTA
print("\n--- Evaluating on TEST SET with Test-Time Augmentation (TTA) ---")
test_dataset = build_dataset(test_df, is_training=False)
correct_tta_predictions_test = 0
true_labels_test = test_df[label_columns].values
for i, (_, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Test TTA")):
    tta_pred = predict_with_tta(model, row['path'], (IMAGE_SIZE, IMAGE_SIZE))
    if np.argmax(tta_pred) == np.argmax(true_labels_test[i]): 
        correct_tta_predictions_test += 1

tta_accuracy_test = correct_tta_predictions_test / len(test_df)

# Print final results
final_val_accuracy = max(history_fine_tune.history.get('val_accuracy', [0]))
print(f"\n{'='*60}")
print(f"FINAL RESULTS (CONTROL - Patient-Based Split)")
print(f"{'='*60}")
print(f"Best Validation Accuracy (during training): {final_val_accuracy:.4f}")
print(f"\nTest-Time Augmentation Results:")
print(f"  Validation Set TTA Accuracy: {tta_accuracy_val:.4f} ({len(val_patients)} patients, {len(validation_df)} images)")
print(f"  Test Set TTA Accuracy:       {tta_accuracy_test:.4f} ({len(test_patients)} patients, {len(test_df)} images)")
print(f"{'='*60}")

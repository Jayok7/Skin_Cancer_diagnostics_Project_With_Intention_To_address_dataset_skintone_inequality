# ========================================================================
# SECTION 7: COMPREHENSIVE EVALUATION WITH PER-CLASS METRICS
# ========================================================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_history(history_head, history_fine_tune):
    """Plot training history"""
    acc = history_head.history['accuracy'] + history_fine_tune.history['accuracy']
    val_acc = history_head.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
    loss = history_head.history['loss'] + history_fine_tune.history['loss']
    val_loss = history_head.history['val_loss'] + history_fine_tune.history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history_head.epoch)-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history_head.epoch)-1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

print("\n--- Plotting combined training history...")
plot_history(history_head, history_fine_tune)

def predict_with_tta(model, image_path, target_size):
    """Test-Time Augmentation with horizontal flip"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded = tf.expand_dims(img_array, axis=0)
    flipped_img_array = tf.image.flip_left_right(img_array_expanded)
    pred_original = model.predict(img_array_expanded, verbose=0)
    pred_flipped = model.predict(flipped_img_array, verbose=0)
    return np.mean([pred_original, pred_flipped], axis=0)[0]

def evaluate_with_metrics(model, df, dataset_name, patient_list):
    """Comprehensive evaluation with per-class metrics"""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name}")
    print(f"{'='*60}")
    
    # Collect predictions
    predictions = []
    true_labels_onehot = df[label_columns].values
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dataset_name} TTA"):
        tta_pred = predict_with_tta(model, row['path'], (IMAGE_SIZE, IMAGE_SIZE))
        predictions.append(tta_pred)
    
    predictions = np.array(predictions)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels_onehot, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(pred_classes == true_classes)
    
    # Classification report (per-class metrics)
    print(f"\nClassification Report:")
    print(classification_report(
        true_classes, 
        pred_classes, 
        target_names=label_columns,
        digits=3
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_columns, 
                yticklabels=label_columns)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\n{dataset_name} Summary:")
    print(f"  Patients: {len(patient_list)}")
    print(f"  Images: {len(df)}")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    
    return accuracy

# Evaluate on validation set
val_accuracy = evaluate_with_metrics(
    model, 
    validation_df, 
    "VALIDATION SET", 
    val_patients
)

# Evaluate on test set
test_accuracy = evaluate_with_metrics(
    model, 
    test_df, 
    "TEST SET", 
    test_patients
)

# Final summary
best_val_acc_training = max(history_fine_tune.history.get('val_accuracy', [0]))

print(f"\n{'='*60}")
print(f"FINAL RESULTS - {('3-WAY' if USE_3WAY_CLASSIFICATION else '6-WAY')} CLASSIFICATION")
print(f"{'='*60}")
print(f"Best Validation Accuracy (during training): {best_val_acc_training:.4f}")
print(f"\nTest-Time Augmentation Results:")
print(f"  Validation Set TTA Accuracy: {val_accuracy:.4f}")
print(f"  Test Set TTA Accuracy:       {test_accuracy:.4f}")
print(f"{'='*60}")

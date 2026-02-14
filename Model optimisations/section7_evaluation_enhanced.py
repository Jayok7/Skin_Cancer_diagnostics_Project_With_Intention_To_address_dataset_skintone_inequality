# ========================================================================
# SECTION 7: ENHANCED EVALUATION WITH ADVANCED METRICS
# ========================================================================

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
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

def plot_precision_recall_curves(y_true_onehot, y_pred_probs, class_names):
    """Plot Precision-Recall curves for each class"""
    n_classes = len(class_names)
    
    # Calculate number of subplots needed
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_classes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        
        axes[i].plot(recall, precision, linewidth=2)
        axes[i].set_xlabel('Recall', fontsize=10)
        axes[i].set_ylabel('Precision', fontsize=10)
        axes[i].set_title(f'Precision-Recall: {class_name}', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1])
        
        # Add baseline (random classifier)
        support = y_true_onehot[:, i].sum() / len(y_true_onehot)
        axes[i].axhline(y=support, color='r', linestyle='--', alpha=0.5, label=f'Baseline ({support:.2f})')
        axes[i].legend(loc='lower left', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_classification_errors(df, predictions, true_labels_onehot, class_names, dataset_name, max_errors=12):
    """Display misclassified images with predicted vs true labels"""
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels_onehot, axis=1)
    
    # Find all errors
    error_indices = np.where(pred_classes != true_classes)[0]
    
    if len(error_indices) == 0:
        print(f"\n🎉 No errors found in {dataset_name}!")
        return
    
    # Sample errors if too many
    if len(error_indices) > max_errors:
        error_indices = np.random.choice(error_indices, max_errors, replace=False)
    
    print(f"\n--- Error Analysis: {dataset_name} ---")
    print(f"Total errors: {len(np.where(pred_classes != true_classes)[0])} out of {len(df)}")
    
    # Calculate errors per class
    print("\nErrors by true class:")
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        class_errors = np.sum((pred_classes != true_classes) & class_mask)
        class_total = np.sum(class_mask)
        if class_total > 0:
            error_rate = class_errors / class_total * 100
            print(f"  {class_name:10s}: {class_errors:3d}/{class_total:3d} ({error_rate:5.1f}% error rate)")
    
    # Display sample errors
    n_cols = 4
    n_rows = (len(error_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5*n_rows))
    if len(error_indices) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    df_indexed = df.reset_index(drop=True)
    
    for idx, ax in enumerate(axes):
        if idx < len(error_indices):
            error_idx = error_indices[idx]
            img_path = df_indexed.loc[error_idx, 'path']
            
            true_class = class_names[true_classes[error_idx]]
            pred_class = class_names[pred_classes[error_idx]]
            confidence = predictions[error_idx, pred_classes[error_idx]] * 100
            
            # Load and display image
            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with true vs predicted
            title = f"True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)"
            ax.set_title(title, fontsize=9, color='red', fontweight='bold')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'Sample Misclassifications - {dataset_name}', fontsize=14, fontweight='bold', y=1.00)
    plt.show()

def class_wise_accuracy_by_fitzpatrick(df, predictions, true_labels_onehot, class_names):
    """Cross-tabulation: predicted class accuracy by original Fitzpatrick type"""
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels_onehot, axis=1)
    
    # Get original Fitzpatrick types (before grouping)
    fitzpatrick_types = df['fitzpatrick_skin_type'].values
    
    print("\n--- Class-wise Accuracy by Original Fitzpatrick Type ---")
    
    # Create cross-tabulation
    fitzpatrick_unique = sorted(df['fitzpatrick_skin_type'].unique())
    
    # Build accuracy table
    results = []
    for fitz_type in fitzpatrick_unique:
        fitz_mask = fitzpatrick_types == fitz_type
        fitz_count = np.sum(fitz_mask)
        
        if fitz_count == 0:
            continue
        
        row = {'Fitzpatrick': fitz_type, 'Total': fitz_count}
        
        # Calculate accuracy for each predicted class
        for i, class_name in enumerate(class_names):
            class_correct = np.sum((pred_classes == i) & (true_classes == i) & fitz_mask)
            class_total = np.sum((true_classes == i) & fitz_mask)
            
            if class_total > 0:
                accuracy = class_correct / class_total * 100
                row[f'{class_name}'] = f"{class_correct}/{class_total} ({accuracy:.1f}%)"
            else:
                row[f'{class_name}'] = "N/A"
        
        # Overall accuracy for this Fitzpatrick type
        overall_correct = np.sum((pred_classes == true_classes) & fitz_mask)
        overall_acc = overall_correct / fitz_count * 100
        row['Overall'] = f"{overall_correct}/{fitz_count} ({overall_acc:.1f}%)"
        
        results.append(row)
    
    # Create DataFrame and display
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Visualize as heatmap
    heatmap_data = []
    heatmap_labels = []
    
    for fitz_type in fitzpatrick_unique:
        fitz_mask = fitzpatrick_types == fitz_type
        fitz_count = np.sum(fitz_mask)
        
        if fitz_count == 0:
            continue
        
        accuracies = []
        for i in range(len(class_names)):
            class_correct = np.sum((pred_classes == i) & (true_classes == i) & fitz_mask)
            class_total = np.sum((true_classes == i) & fitz_mask)
            
            if class_total > 0:
                accuracies.append(class_correct / class_total * 100)
            else:
                accuracies.append(0)
        
        heatmap_data.append(accuracies)
        heatmap_labels.append(fitz_type)
    
    if heatmap_data:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='.1f', 
            cmap='RdYlGn',
            xticklabels=class_names,
            yticklabels=heatmap_labels,
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Accuracy (%)'}
        )
        plt.title('Accuracy by Fitzpatrick Type and Predicted Class')
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Fitzpatrick Type')
        plt.tight_layout()
        plt.show()

def evaluate_with_metrics(model, df, dataset_name, patient_list):
    """Comprehensive evaluation with all metrics"""
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
    
    # 1. Classification report
    print(f"\n1. CLASSIFICATION REPORT:")
    print(classification_report(
        true_classes, 
        pred_classes, 
        target_names=label_columns,
        digits=3
    ))
    
    # 2. Confusion matrix
    print(f"\n2. CONFUSION MATRIX:")
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
    
    # 3. Precision-Recall curves
    print(f"\n3. PRECISION-RECALL CURVES:")
    plot_precision_recall_curves(true_labels_onehot, predictions, label_columns)
    
    # 4. Error analysis
    print(f"\n4. ERROR ANALYSIS:")
    show_classification_errors(df, predictions, true_labels_onehot, label_columns, dataset_name)
    
    # 5. Class-wise accuracy by Fitzpatrick type
    print(f"\n5. ACCURACY BY ORIGINAL FITZPATRICK TYPE:")
    class_wise_accuracy_by_fitzpatrick(df, predictions, true_labels_onehot, label_columns)
    
    # Summary
    print(f"\n{dataset_name} SUMMARY:")
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

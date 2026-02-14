# ==================== CELL 5: Precision-Recall Curves ====================
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 7))

for i, label in enumerate(label_columns):
    precision, recall, _ = precision_recall_curve(
        test_labels[:, i], 
        test_predictions[:, i]
    )
    ap = average_precision_score(test_labels[:, i], test_predictions[:, i])
    ax.plot(recall, precision, lw=2, label=f'{label} (AP={ap:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Test Set', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Average Precision Scores:")
for i, label in enumerate(label_columns):
    ap = average_precision_score(test_labels[:, i], test_predictions[:, i])
    print(f"  {label:8s}: {ap:.3f}")


# ==================== CELL 6: Error Analysis with Images ====================
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Find misclassified samples
y_true = test_labels.argmax(axis=1)
y_pred = test_predictions.argmax(axis=1)
errors = y_true != y_pred
error_indices = np.where(errors)[0]

print(f"Total errors: {len(error_indices)} out of {len(test_df)} ({len(error_indices)/len(test_df):.1%})")

# Show first 12 misclassified images
n_to_show = min(12, len(error_indices))
if n_to_show > 0:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, error_idx in enumerate(error_indices[:n_to_show]):
        # Get image info
        img_path = test_df.iloc[error_idx]['path']
        true_label = label_columns[y_true[error_idx]]
        pred_label = label_columns[y_pred[error_idx]]
        confidence = test_predictions[error_idx].max()
        
        # Load and display image
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add title with true/predicted labels
        title = f"True: {true_label}\nPred: {pred_label} ({confidence:.1%})"
        color = 'red'
        axes[idx].set_title(title, fontsize=10, color=color, weight='bold')
    
    # Hide unused subplots
    for idx in range(n_to_show, 12):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Images', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
else:
    print("No errors to display - perfect classification!")


# ==================== CELL 7: Confusion Analysis per Class ====================
import pandas as pd

print("\nDetailed Confusion Analysis:\n")

for i, true_label in enumerate(label_columns):
    # Get samples for this true class
    mask = y_true == i
    true_count = mask.sum()
    
    if true_count == 0:
        continue
    
    print(f"{true_label} (n={true_count}):")
    
    # Count predictions for this true class
    for j, pred_label in enumerate(label_columns):
        pred_count = (y_pred[mask] == j).sum()
        percentage = pred_count / true_count * 100
        
        if pred_count > 0:
            symbol = "✓" if i == j else "✗"
            print(f"  {symbol} Predicted as {pred_label:8s}: {pred_count:3d} ({percentage:5.1f}%)")
    
    print()


# ==================== CELL 8: Class-wise Accuracy by Fitzpatrick Type ====================
print("Accuracy Breakdown by Original Fitzpatrick Type:\n")

# Create crosstab: rows=true Fitzpatrick type, columns=predicted 3-way label
test_df_copy = test_df.copy()
test_df_copy['predicted_label'] = [label_columns[i] for i in y_pred]
test_df_copy['correct'] = y_pred == y_true

# Group by original Fitzpatrick type
for fitz_type in sorted(test_df_copy['fitzpatrick_skin_type'].unique()):
    subset = test_df_copy[test_df_copy['fitzpatrick_skin_type'] == fitz_type]
    accuracy = subset['correct'].mean()
    count = len(subset)
    grouped = subset['target_label'].iloc[0]
    
    print(f"Type {fitz_type} ({grouped}, n={count}): {accuracy:.1%} accuracy")
    
    # Show prediction distribution
    pred_dist = subset['predicted_label'].value_counts()
    for label, cnt in pred_dist.items():
        print(f"  → {label:8s}: {cnt:3d} ({cnt/count*100:5.1f}%)")
    print()

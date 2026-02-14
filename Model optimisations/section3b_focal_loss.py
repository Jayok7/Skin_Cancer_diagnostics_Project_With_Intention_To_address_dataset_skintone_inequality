# ========================================================================
# SECTION 3B: FOCAL LOSS IMPLEMENTATION
# ========================================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard-to-classify examples.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing parameter
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        focal_weight = tf.pow(1 - y_pred, gamma)
        
        # Apply focal loss
        focal_loss_val = alpha * focal_weight * ce
        
        return tf.reduce_sum(focal_loss_val, axis=-1)
    
    return loss_fn

print("\n--- Focal Loss configured ---")
print(f"  Gamma: {FOCAL_LOSS_GAMMA} (focus on hard examples)")
print(f"  Alpha: {FOCAL_LOSS_ALPHA} (balancing parameter)")

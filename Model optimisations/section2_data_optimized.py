# ========================================================================
# SECTION 2: DATA LOADING WITH FITZPATRICK CLASSIFICATION
# ========================================================================

print("\n--- Loading and preparing data (FITZPATRICK SKIN TYPE CLASSIFICATION) ---")
df = pd.read_csv(CSV_PATH)
df['path'] = df['isic_id'].apply(lambda x: os.path.join(BASE_PATH, x + '.jpg'))

# Remove rows with missing Fitzpatrick labels
df = df.dropna(subset=['fitzpatrick_skin_type', 'path'])
print(f"Total images after removing missing labels: {len(df)}")

# Class grouping function
def group_fitzpatrick(skin_type):
    """Group Fitzpatrick types into 3 broader categories"""
    if skin_type in ['I', 'II']:
        return 'Light'
    elif skin_type in ['III', 'IV']:
        return 'Medium'
    else:  # V, VI
        return 'Dark'

# Apply grouping or use original labels based on configuration
if USE_3WAY_CLASSIFICATION:
    df['target_label'] = df['fitzpatrick_skin_type'].apply(group_fitzpatrick)
    print("\n✓ Using 3-way classification: Light / Medium / Dark")
else:
    df['target_label'] = df['fitzpatrick_skin_type']
    print("\n✓ Using 6-way classification: I / II / III / IV / V / VI")

# Show class distribution
print("\nClass distribution:")
class_dist = df.groupby('target_label').agg({
    'patient_id': 'nunique',
    'isic_id': 'count'
}).rename(columns={'patient_id': 'Patients', 'isic_id': 'Images'})
print(class_dist)

# Create one-hot encoded labels
label_columns = sorted(df['target_label'].unique())
df = pd.concat([df, pd.get_dummies(df['target_label'], dtype='float32')], axis=1)

print(f"\nNumber of classes: {len(label_columns)}")
print(f"Classes: {label_columns}")

# ========================================================================
# PATIENT-BASED SPLIT (80/10/10)
# ========================================================================

print("\n--- Splitting by Patient ID (80/10/10) ---")
unique_patients = df['patient_id'].unique()
n_patients = len(unique_patients)
print(f"Total patients: {n_patients}")

# First split: 80% train, 20% temp (for val+test)
train_patients, temp_patients = train_test_split(
    unique_patients, 
    test_size=0.2,
    random_state=42
)

# Second split: Split the 20% into 10% validation and 10% test
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.5,
    random_state=42
)

# Filter dataframe by patient assignments
train_df = df[df['patient_id'].isin(train_patients)].copy()
validation_df = df[df['patient_id'].isin(val_patients)].copy()
test_df = df[df['patient_id'].isin(test_patients)].copy()

# Verification: Ensure no patient overlap
assert len(set(train_patients) & set(val_patients)) == 0, "ERROR: Patient overlap between train and validation!"
assert len(set(train_patients) & set(test_patients)) == 0, "ERROR: Patient overlap between train and test!"
assert len(set(val_patients) & set(test_patients)) == 0, "ERROR: Patient overlap between validation and test!"

print(f"\n✓ Patient-based split complete (no overlap verified):")
print(f"  Training:   {len(train_patients):2d} patients ({len(train_patients)/n_patients*100:.1f}%) → {len(train_df):4d} images")
print(f"  Validation: {len(val_patients):2d} patients ({len(val_patients)/n_patients*100:.1f}%) → {len(validation_df):4d} images")
print(f"  Test:       {len(test_patients):2d} patients ({len(test_patients)/n_patients*100:.1f}%) → {len(test_df):4d} images")

# ========================================================================
# COMPUTE CLASS WEIGHTS FOR IMBALANCED DATA
# ========================================================================

print("\n--- Computing class weights for imbalanced data ---")

# Get all training labels
train_labels = train_df['target_label'].values

# Compute class weights
unique_classes = np.unique(train_labels)
class_weights_array = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=train_labels
)

# Create dictionary mapping class index to weight
# Need to map string labels to indices matching one-hot encoding
label_to_index = {label: idx for idx, label in enumerate(label_columns)}
class_weight_dict = {label_to_index[cls]: weight for cls, weight in zip(unique_classes, class_weights_array)}

print("\nClass weights:")
for label in label_columns:
    idx = label_to_index[label]
    weight = class_weight_dict.get(idx, 1.0)
    count = (train_labels == label).sum()
    print(f"  {label:10s}: {weight:.3f} (n={count})")

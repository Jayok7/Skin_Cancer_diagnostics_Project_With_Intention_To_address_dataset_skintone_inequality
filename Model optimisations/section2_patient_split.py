# ========================================================================
# SECTION 2: DATA LOADING AND PREPARATION (PATIENT-BASED SPLIT)
# ========================================================================
print("\n--- Loading and preparing data (CONTROL - ORIGINAL IMAGES) ---")
df = pd.read_csv(CSV_PATH)
df['path'] = df['isic_id'].apply(lambda x: os.path.join(BASE_PATH, x + '.jpg')) # Simplified path as images are directly in BASE_PATH
df = df.dropna(subset=['path']) # FIX: Also drop NaNs in 'melanocytic' column
label_columns = sorted(df['melanocytic'].unique())
df = pd.concat([df, pd.get_dummies(df['melanocytic'], dtype='float32')], axis=1)

# PATIENT-BASED SPLIT: Split by patient ID to prevent data leakage
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
    test_size=0.5,  # 50% of 20% = 10% of total
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

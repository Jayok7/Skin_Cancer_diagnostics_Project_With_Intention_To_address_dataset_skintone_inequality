import pandas as pd

df = pd.read_csv('datasets/mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv')
print(f'Total rows: {len(df)}')
print(f'Unique patients: {df["patient_id"].nunique()}')
print(f'\nImages per patient stats:')
print(df.groupby('patient_id').size().describe())
print(f'\nSample patient IDs:')
print(df['patient_id'].unique()[:10])

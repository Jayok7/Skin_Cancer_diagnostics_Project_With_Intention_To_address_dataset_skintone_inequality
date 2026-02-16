
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================================================
# SETTINGS
# ========================================================================
MSKCC_CSV = 'datasets/mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv'
IMAGE_ROOT = 'datasets/MSKCC-images'
CROP_RATIO = 0.4

# ========================================================================
# UTILS (Reused from compute_fairface_labels.py)
# ========================================================================

def extract_skin_patch(image_bgr: np.ndarray, crop_ratio: float = 0.4) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    margin_y = int(h * (1 - crop_ratio) / 2)
    margin_x = int(w * (1 - crop_ratio) / 2)
    return image_bgr[margin_y:h - margin_y, margin_x:w - margin_x]

def compute_median_lstar(skin_patch_bgr: np.ndarray) -> float:
    lab = cv2.cvtColor(skin_patch_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    return float(np.median(l_channel))

# ========================================================================
# MAIN ANALYSIS
# ========================================================================

def main():
    print(f"Loading metadata from {MSKCC_CSV}...")
    df = pd.read_csv(MSKCC_CSV)
    
    # Filter for rows with Fitzpatrick labels
    df = df.dropna(subset=['fitzpatrick_skin_type'])
    
    # Filter for closeup images only (exclude dermoscopic)
    print("Filtering for 'clinical: close-up' images...")
    initial_count = len(df)
    df = df[df['image_type'] == 'clinical: close-up']
    print(f"Analyzing {len(df)} closeup images (dropped {initial_count - len(df)} dermoscopic/other)...")

    lstar_values = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['isic_id']
        path = os.path.join(IMAGE_ROOT, f"{img_id}.jpg")
        
        if not os.path.exists(path):
            continue
            
        img = cv2.imread(path)
        if img is None:
            continue
            
        patch = extract_skin_patch(img, CROP_RATIO)
        if patch.size == 0:
            continue
            
        l_val = compute_median_lstar(patch)
        lstar_values.append({
            'fitzpatrick_skin_type': row['fitzpatrick_skin_type'],
            'l_star': l_val
        })

    results_df = pd.DataFrame(lstar_values)
    
    print("\n--- MSKCC L* STATISTICS BY FITZPATRICK TYPE ---")
    stats = results_df.groupby('fitzpatrick_skin_type')['l_star'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(stats)
    
    # Compare with current centroids
    print("\n--- COMPARISON WITH MILK10k CENTROIDS ---")
    current_centroids = {
        '5 (Type I)': 83.0,
        '4 (Type II)': 75.7,
        '3 (Type III)': 66.7,
        '2 (Type IV)': 61.9,
        '1 (Type V)': 53.7,
        '0 (Type VI)': 52.1
    }
    
    print(f"{'Type':<15} {'MSKCC Median':<15} {'Center Value':<15} {'Diff':<10}")
    print("-" * 60)
    
    type_map = {
        'I': '5 (Type I)', 
        'II': '4 (Type II)',
        'III': '3 (Type III)',
        'IV': '2 (Type IV)',
        'V': '1 (Type V)',
        'VI': '0 (Type VI)'
    }
    
    for f_type in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        if f_type in stats.index:
            mskcc_med = stats.loc[f_type, 'median']
            centroid_key = type_map[f_type]
            milk_val = current_centroids[centroid_key]
            diff = mskcc_med - milk_val
            print(f"{f_type:<15} {mskcc_med:<15.1f} {milk_val:<15.1f} {diff:<+10.1f}")

if __name__ == "__main__":
    main()

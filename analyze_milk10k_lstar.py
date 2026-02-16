
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========================================================================
# SETTINGS
# ========================================================================
MILK_CSV = 'datasets/milk10k_harvard.csv'
IMAGE_ROOT = 'datasets/MILK10k_images'
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
    print(f"Loading metadata from {MILK_CSV}...")
    df = pd.read_csv(MILK_CSV)
    
    # Filter for rows with Label
    df = df.dropna(subset=['skin_tone_class'])
    
    # Filter for closeup images only
    print("Filtering for 'clinical: close-up' images...")
    initial_count = len(df)
    # Note: Checking for substring or exact match. Based on MSKCC experience, likely 'clinical: close-up'
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
            'skin_tone_class': row['skin_tone_class'],
            'l_star': l_val
        })

    results_df = pd.DataFrame(lstar_values)
    
    print("\n--- MILK10k L* STATISTICS BY SKIN TONE CLASS ---")
    if results_df.empty:
        print("No images processed! Check paths or filters.")
        return

    stats = results_df.groupby('skin_tone_class')['l_star'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print(stats)
    
    # Compare with "Official" MILK10k Centroids from the script
    # Note: Script uses 5=Type I ... 0=Type VI. 
    # MILK10k CSV likely uses 1=Type I ... 6=Type VI (Need to verify this mapping from stats)
    
    print("\n--- COMPARISON WITH HARDCODED CENTROIDS ---")
    # Mapping based on assumption 1=Type I, 6=Type VI
    # Hardcoded values:
    # 5 (Type I): 83.0
    # 4 (Type II): 75.7
    # 3 (Type III): 66.7
    # 2 (Type IV): 61.9
    # 1 (Type V): 53.7
    # 0 (Type VI): 52.1
    
    hardcoded_centroids = {
        1: 83.0, # Type I
        2: 75.7, # Type II
        3: 66.7, # Type III
        4: 61.9, # Type IV
        5: 53.7, # Type V
        6: 52.1  # Type VI
    }
    
    print(f"{'Class':<10} {'MILK10k Median':<15} {'Hardcoded':<15} {'Diff':<10}")
    print("-" * 55)
    
    for cls in sorted(stats.index):
        if cls in hardcoded_centroids:
            milk_med = stats.loc[cls, 'median']
            hard_val = hardcoded_centroids[cls]
            diff = milk_med - hard_val
            print(f"{cls:<10} {milk_med:<15.1f} {hard_val:<15.1f} {diff:<+10.1f}")
        else:
            print(f"{cls:<10} {stats.loc[cls, 'median']:<15.1f} {'N/A':<15} {'N/A':<10}")

if __name__ == "__main__":
    main()

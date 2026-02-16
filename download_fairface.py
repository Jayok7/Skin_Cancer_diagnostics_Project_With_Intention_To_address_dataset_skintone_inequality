#!/usr/bin/env python3
"""
FairFace Dataset Downloader
===========================
Downloads the FairFace dataset (padding=0.25) and labels from Google Drive
using 'gdown' (pip install gdown).

Files to download:
1. fairface-img-margin025-trainval.zip (Images)
2. fairface_label_train.csv
3. fairface_label_val.csv

Usage:
    pip install gdown
    python download_fairface.py
"""

import os
import zipfile
import subprocess
import shutil

# Check if gdown is installed
try:
    import gdown
except ImportError:
    print("Installing gdown...")
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

# Configuration
DATASETS_DIR = "datasets"
URLS = {
    "fairface-img-margin025-trainval.zip": "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86",
    "fairface_label_train.csv": "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH",
    "fairface_label_val.csv": "1wOdja-ezstMEp81tX1a-EYkFebev4h7D",
}

def download_file(file_name, file_id, output_dir):
    """Download a file from Google Drive."""
    url = f'https://drive.google.com/uc?id={file_id}'
    output_path = os.path.join(output_dir, file_name)
    
    if os.path.exists(output_path):
        print(f"✓ {file_name} already exists. Skipping download.")
        return output_path
        
    print(f"\nDownloading {file_name}...")
    gdown.download(url, output_path, quiet=False)
    return output_path

def extract_zip(zip_path, extract_to):
    """Extract a ZIP file."""
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✓ Extraction complete.")

def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    print(f"{'='*60}")
    print("DOWNLOADING FAIRFACE DATASET")
    print(f"{'='*60}")
    
    # Download CSVs
    for name in ["fairface_label_train.csv", "fairface_label_val.csv"]:
        download_file(name, URLS[name], DATASETS_DIR)
        
    # Download Images ZIP
    zip_name = "fairface-img-margin025-trainval.zip"
    zip_path = download_file(zip_name, URLS[zip_name], DATASETS_DIR)
    
    # Extract Images
    # The ZIP likely contains a folder named 'fairface-img-margin025-trainval'
    # We'll check if the extracted folder exists
    extracted_folder = os.path.join(DATASETS_DIR, "fairface-img-margin025-trainval")
    if not os.path.exists(extracted_folder):
        extract_zip(zip_path, DATASETS_DIR)
    else:
        print(f"✓ Images folder already exists at {extracted_folder}")
        
    print(f"\n{'='*60}")
    print("✓ DOWNLOAD COMPLETE")
    print(f"  Data provided in: {DATASETS_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

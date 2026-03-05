#!/usr/bin/env python3
"""
UTKFace Dataset Downloader
===========================
Downloads the UTKFace aligned & cropped dataset from Google Drive/Kaggle.

UTKFace has ~24k face images named: [age]_[gender]_[race]_[date].jpg
  - age:    0-116
  - gender: 0=male, 1=female
  - race:   0=White, 1=Black, 2=Asian, 3=Indian, 4=Other

Usage:
    pip install gdown
    python download_utkface.py
"""

import os
import subprocess
import tarfile
import zipfile
import glob

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

DATASETS_DIR = "datasets"
UTKFACE_DIR = os.path.join(DATASETS_DIR, "UTKFace")

# UTKFace Google Drive links (aligned & cropped, parts 1-3)
URLS = {
    "UTKFace_part1.tar.gz": "1mbnyBBAVZg6WH8GBMyl9e_4JCwZLfuhM",
    "UTKFace_part2.tar.gz": "1T5KH4MYf5YjQfrMYAdFMHhqvBmfnUqTO",
    "UTKFace_part3.tar.gz": "1vqBRwNPpfBl5OU7PlF-JaYxhXwam2_Ws",
}


def download_file(file_name, file_id, output_dir):
    """Download a file from Google Drive."""
    url = f'https://drive.google.com/uc?id={file_id}'
    output_path = os.path.join(output_dir, file_name)

    if os.path.exists(output_path):
        print(f"✓ {file_name} already exists. Skipping.")
        return output_path

    print(f"\nDownloading {file_name}...")
    gdown.download(url, output_path, quiet=False)
    return output_path


def extract_archive(archive_path, extract_to):
    """Extract a tar.gz or zip archive."""
    print(f"Extracting {os.path.basename(archive_path)}...")
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_to)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    print("✓ Extracted.")


def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    print(f"{'='*60}")
    print("DOWNLOADING UTKFACE DATASET")
    print(f"{'='*60}")

    # Check if already extracted
    if os.path.isdir(UTKFACE_DIR):
        n_imgs = len(glob.glob(os.path.join(UTKFACE_DIR, "*.jpg")))
        if n_imgs > 20000:
            print(f"✓ UTKFace already exists with {n_imgs:,} images. Skipping.")
            return

    # Download parts
    archive_paths = []
    for name, file_id in URLS.items():
        path = download_file(name, file_id, DATASETS_DIR)
        archive_paths.append(path)

    # Extract each part into datasets/ (they all contain a UTKFace/ folder)
    for path in archive_paths:
        if os.path.exists(path):
            extract_archive(path, DATASETS_DIR)

    # Verify
    n_imgs = len(glob.glob(os.path.join(UTKFACE_DIR, "*.jpg")))
    print(f"\n{'='*60}")
    print(f"✓ DOWNLOAD COMPLETE — {n_imgs:,} images in {UTKFACE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge FairFace + UTKFace MST labels into a master dataset.

File paths are prefixed so they all resolve from a common --image-root:
  FairFace: fairface-img-margin025-trainval/train/1.jpg
  UTKFace:  UTKFace/utkface-aligned-labeled/UTKFace_images/filename.jpg
"""

import pandas as pd

MST10_CLASS_NAMES = [
    "MST 10 (deepest)", "MST 9", "MST 8", "MST 7", "MST 6",
    "MST 5", "MST 4", "MST 3", "MST 2", "MST 1 (lightest)",
]
MST5_CLASS_NAMES = [
    "Very Dark (MST 9-10)", "Dark (MST 7-8)", "Medium (MST 5-6)",
    "Light (MST 3-4)", "Very Light (MST 1-2)",
]

# Load both datasets
ff = pd.read_csv("datasets/fairface_mst_labels.csv")
utk = pd.read_csv("datasets/utkface_mst_labels.csv")

# Prefix file paths so they resolve from datasets/ as root
# FairFace: "train/1.jpg" → "fairface-img-margin025-trainval/train/1.jpg"
ff["file"] = "fairface-img-margin025-trainval/" + ff["file"]

# UTKFace: "100_0_0_xxx.jpg" → "UTKFace/utkface-aligned-labeled/UTKFace_images/100_0_0_xxx.jpg"
utk["file"] = "UTKFace/utkface-aligned-labeled/UTKFace_images/" + utk["file"]

# Add source column
ff["source"] = "fairface"
utk["source"] = "utkface"

# Only keep common columns + source
common_cols = ["file", "original_split", "ita", "mst10_class", "mst5_class", "source"]
ff = ff[common_cols]
utk = utk[common_cols]

# Merge
master = pd.concat([ff, utk], ignore_index=True)
master.to_csv("datasets/master_mst_labels.csv", index=False)

print(f"FairFace: {len(ff):,} images")
print(f"UTKFace:  {len(utk):,} images")
print(f"Master:   {len(master):,} images")
print(f"\nSample FairFace path: {master.iloc[0]['file']}")
print(f"Sample UTKFace path:  {master.iloc[len(ff)]['file']}")

# Print distributions
for mode, col, names in [("MST-10", "mst10_class", MST10_CLASS_NAMES),
                          ("MST-5", "mst5_class", MST5_CLASS_NAMES)]:
    print(f"\n{'='*60}")
    print(f"COMBINED CLASS DISTRIBUTION ({mode})")
    print(f"{'='*60}")
    dist = master[col].value_counts().sort_index()
    total = len(master)
    for cls_id in range(len(names)):
        count = dist.get(cls_id, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        name = names[cls_id]
        print(f"  Class {cls_id} ({name:25s}): {count:6,}  ({pct:5.1f}%)  {bar}")

# Show what undersampling at 15k cap would produce for MST-5
print(f"\n{'='*60}")
print(f"AFTER UNDERSAMPLING (cap=15,000/class for MST-5)")
print(f"{'='*60}")
for cls_id in range(5):
    raw = len(master[master["mst5_class"] == cls_id])
    capped = min(raw, 15000)
    oversample = 15000 / capped if capped > 0 else 0
    name = MST5_CLASS_NAMES[cls_id]
    print(f"  {name:25s}: {raw:6,} → {capped:6,}  (oversample: {oversample:.1f}×)")

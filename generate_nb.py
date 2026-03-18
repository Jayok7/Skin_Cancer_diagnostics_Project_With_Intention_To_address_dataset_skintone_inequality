import nbformat as nbf

nb = nbf.v4.new_notebook()

# ----------------- SECTION 1 -----------------
md1 = nbf.v4.new_markdown_cell('''# The Paradigm Shift: From Fitzpatrick to Monk Skin Tone
## Section 1: The FST Baseline and Middle-Class Collapse

Historically, the Fitzpatrick Skin Type (FST) scale was used. However, FST was not designed for computer vision and has a "middle-class collapse" problem. 

Let's look at the distribution of FST in the MSKCC and MILK10k datasets, and see what happens when MSKCC is graded with the Monk Skin Tone (MST) scale instead.''')

code1 = nbf.v4.new_code_cell('''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load datasets
mskcc_df = pd.read_csv('datasets/mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv')
milk_df = pd.read_csv('datasets/milk10k_harvard.csv')
mskcc_mst_df = pd.read_csv('datasets/MSKCC-MST/Analysis of clustering of FST, MST, and Pantone classes in the 2-dimensional CIELAB color space on non-lesional sites..csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MSKCC FST Plot
sns.countplot(
    data=mskcc_df.dropna(subset=['fitzpatrick_skin_type']), 
    x='fitzpatrick_skin_type', 
    order=['I', 'II', 'III', 'IV', 'V', 'VI'],
    ax=axes[0], palette='viridis'
)
axes[0].set_title('MSKCC FST Distribution')
axes[0].set_xlabel('Fitzpatrick Skin Type')

# MSKCC MST Plot
sns.countplot(
    data=mskcc_mst_df.dropna(subset=['mst_r1']), 
    x='mst_r1', 
    ax=axes[1], palette='copper_r'
)
axes[1].set_title('MSKCC MST Distribution (Rater 1)')
axes[1].set_xlabel('Monk Skin Tone (MST-10)')

# MILK Plot
sns.countplot(
    data=milk_df.dropna(subset=['skin_tone_class']), 
    x='skin_tone_class',
    ax=axes[2], palette='viridis'
)
axes[2].set_title('MILK10k (Approximated FST) Distribution')
axes[2].set_xlabel('Skin Tone Class')

plt.tight_layout()
plt.show()''')

# ----------------- SECTION 2 -----------------
md2 = nbf.v4.new_markdown_cell('''## Section 2: The ITA and MST Solution

The solution to middle-class collapse is the **Continuous Individual Typology Angle (ITA)** mapped to the **Monk Skin Tone (MST)** scale. 

Let's visualize how ITA scores distribute across the entire master dataset, colored by their resulting MST-5 class.''')

code2 = nbf.v4.new_code_cell('''# Load MST Master Labels
mst_df = pd.read_csv('datasets/master_mst_labels.csv')

# MST-5 Class Mapping
mst5_mapping = {
    0: 'Very Dark (MST 9-10)',
    1: 'Dark (MST 7-8)',
    2: 'Medium (MST 5-6)',
    3: 'Light (MST 3-4)',
    4: 'Very Light (MST 1-2)'
}
mst_df['MST-5 Name'] = mst_df['mst5_class'].map(mst5_mapping)

plt.figure(figsize=(10, 6))
sns.histplot(
    data=mst_df, 
    x='ita', 
    hue='MST-5 Name', 
    element='step', 
    stat='density', 
    common_norm=False,
    palette=['#3d2314', '#8d5537', '#c08960', '#e2b49a', '#f5d0cd'], 
    hue_order=[mst5_mapping[i] for i in range(5)],
    alpha=0.6,
    linewidth=1.5
)

plt.title('Distribution of ITA Scores Across MST-5 Classes')
plt.xlabel('Individual Typology Angle (ITA)')
plt.ylabel('Density')
plt.axvline(x=41, color='gray', linestyle='--', alpha=0.5, label='Light / Medium boundary')
plt.axvline(x=28, color='gray', linestyle='--', alpha=0.5, label='Medium / Dark boundary')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()''')

# ----------------- SECTION 3 -----------------
md3 = nbf.v4.new_markdown_cell('''## Section 3: Dataset Comparisons (FairFace vs. UTKFace)

With the MST-5 scale established, we can compare the demographic makeup of our key training datasets. We'll look at the MST-5 distributions for FairFace, UTKFace, and the combined dataset.''')

code3 = nbf.v4.new_code_cell('''# Load individual datasets to be perfectly certain of their contents
fairface_df = pd.read_csv('datasets/fairface_mst_labels.csv')
fairface_df['source'] = 'FairFace'

utk_df = pd.read_csv('datasets/utkface_mst_labels.csv')
utk_df['source'] = 'UTKFace'

# Combine them
combined_df = pd.concat([fairface_df, utk_df], ignore_index=True)
combined_df['MST-5 Name'] = combined_df['mst5_class'].map(mst5_mapping)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Define hue order and palette for consistency
order = [mst5_mapping[i] for i in range(5)]
palette = ['#3d2314', '#8d5537', '#c08960', '#e2b49a', '#f5d0cd']

# FairFace
sns.countplot(data=fairface_df, x='mst5_class', ax=axes[0], palette=palette)
axes[0].set_title('FairFace MST-5 Distribution')
axes[0].set_xlabel('MST-5 Class')
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(order, rotation=45, ha='right')

# UTKFace
sns.countplot(data=utk_df, x='mst5_class', ax=axes[1], palette=palette)
axes[1].set_title('UTKFace MST-5 Distribution')
axes[1].set_xlabel('MST-5 Class')
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(order, rotation=45, ha='right')

# Combined
sns.countplot(data=combined_df, x='mst5_class', ax=axes[2], palette=palette)
axes[2].set_title('Combined (FairFace + UTKFace) MST-5 Distribution')
axes[2].set_xlabel('MST-5 Class')
axes[2].set_xticks(range(5))
axes[2].set_xticklabels(order, rotation=45, ha='right')

plt.tight_layout()
plt.show()''')

md4 = nbf.v4.new_markdown_cell('''## Conclusion

The continuous ITA scale and its mapping to robust MST-5 categories provide a much healthier distribution for deep learning than the disjointed FST scale. This enables models to train fairly across all skin tones without collapsing the middle classes.''')

nb['cells'] = [md1, code1, md2, code2, md3, code3, md4]

nbf.write(nb, 'FairFace_v3_Dataset_Visualization.ipynb')

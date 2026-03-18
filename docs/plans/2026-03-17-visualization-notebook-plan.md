# Visualization Notebook Updates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `FairFace_v3_Dataset_Visualization.ipynb` to accurately reflect MSKCC MST vs FST distributions, and add a rigorous comparison of MILK10k vs Fitzpatrick17k with strict color mapping.

**Architecture:** We will modify the existing notebook using standard JSON manipulation or direct file edits to update the matplotlib/seaborn plotting code in Section 1 and add a new Section 2. We will apply a specific hex color mapping to ensure consistency across the FST-like scales.

**Tech Stack:** Python, Pandas, Matplotlib, Seaborn, Jupyter Notebook

---

### Task 1: Update Section 1 (MSKCC Focus)

**Files:**
- Modify: `d:\skin cancer project\FairFace_v3_Dataset_Visualization.ipynb` (Cell containing Section 1 plots)

**Step 1: Write the updated plotting code for Section 1**

```python
# Updated Section 1 plotting code
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Define the exact hex colors requested by the user
fst_colors = {
    'I': '#FFE6CC',
    'II': '#FFD1A1',
    'III': '#CD906A',
    'IV': '#FFB374',
    'V': '#A36035',
    'VI': '#693F1B',
    'N/A': '#808080'
}

# MSKCC FST Plot
# Replace any missing/NaN with 'N/A'
mskcc_fst_data = mskcc_df['fitzpatrick_skin_type'].fillna('N/A')
sns.countplot(
    x=mskcc_fst_data, 
    order=['I', 'II', 'III', 'IV', 'V', 'VI', 'N/A'],
    ax=axes[0], 
    palette=fst_colors
)
axes[0].set_title('MSKCC FST Distribution')
axes[0].set_xlabel('Fitzpatrick Skin Type')

# MSKCC MST Plot (Rater 1)
sns.countplot(
    data=mskcc_mst_df.dropna(subset=['mst_r1']), 
    x='mst_r1', 
    ax=axes[1], palette='copper_r'
)
axes[1].set_title('MSKCC MST Distribution (Rater 1)')
axes[1].set_xlabel('Monk Skin Tone (MST-10)')

# MSKCC MST Plot (Rater 2) - Replacing MILK10k
sns.countplot(
    data=mskcc_mst_df.dropna(subset=['mst_r2']), 
    x='mst_r2', 
    ax=axes[2], palette='copper_r'
)
axes[2].set_title('MSKCC MST Distribution (Rater 2)')
axes[2].set_xlabel('Monk Skin Tone (MST-10)')

plt.tight_layout()
plt.show()
```

**Step 2: Apply the code to the notebook**

Modify the corresponding code cell in `FairFace_v3_Dataset_Visualization.ipynb`.

**Step 3: Run the notebook to verify it passes**

Run: `jupyter nbconvert --to notebook --execute "FairFace_v3_Dataset_Visualization.ipynb"` (or run the specific cell using a python script)
Expected: Execution succeeds without errors, and the plots are generated correctly.

**Step 4: Commit**

```bash
git add "FairFace_v3_Dataset_Visualization.ipynb"
git commit -m "feat: update MSKCC section with Rater 2 and custom FST colors"
```

---

### Task 2: Add Section 2 (MILK10k vs Fitzpatrick17k)

**Files:**
- Modify: `d:\skin cancer project\FairFace_v3_Dataset_Visualization.ipynb` (Insert new markdown and code cells)

**Step 1: Write the new Markdown cell content**

```markdown
## Section 2: Legacy Datasets (MILK10k vs Fitzpatrick17k)

While MSKCC allows us to compare FST and MST on the same images, we also rely on legacy datasets. 
Below is a comparison of Fitzpatrick17k (standard FST) and MILK10k. 

**Note:** MILK10k uses a "Skin Tone Class" from 1 to 6 that roughly approximates FST, but the scale is inverted (6 = Lightest, 1 = Darkest) and is not strictly Fitzpatrick.
```

**Step 2: Write the new Code cell content**

```python
# Load Fitzpatrick17k
fitz17k_df = pd.read_csv('datasets/fitzpatrick17k.csv')

# Prepare MILK10k data
# Replace NaN with 'N/A'
milk_st = milk_df['skin_tone_class'].fillna('N/A').astype(str)
# Format valid floats like '1.0' back to '1' or '6'
milk_st = milk_st.apply(lambda x: str(int(float(x))) if x != 'N/A' else x)

# Prepare Fitzpatrick17k data
# -1 represents missing/unassigned in this dataset
fitz_st = fitz17k_df['fitzpatrick_scale'].replace(-1, 'N/A').fillna('N/A').astype(str)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MILK10k colors mapping (inverted FST)
milk_colors = {
    '6': '#FFE6CC', # I
    '5': '#FFD1A1', # II
    '4': '#CD906A', # III
    '3': '#FFB374', # IV
    '2': '#A36035', # V
    '1': '#693F1B', # VI
    'N/A': '#808080'
}

# Subplot 1: Fitzpatrick17k
sns.countplot(
    x=fitz_st,
    order=['1', '2', '3', '4', '5', '6', 'N/A'],
    ax=axes[0],
    palette={'1': '#FFE6CC', '2': '#FFD1A1', '3': '#CD906A', '4': '#FFB374', '5': '#A36035', '6': '#693F1B', 'N/A': '#808080'}
)
axes[0].set_title('Fitzpatrick17k FST Distribution')
axes[0].set_xlabel('Fitzpatrick Skin Type')

# Subplot 2: MILK10k
sns.countplot(
    x=milk_st,
    order=['6', '5', '4', '3', '2', '1', 'N/A'],
    ax=axes[1],
    palette=milk_colors
)
axes[1].set_title('MILK10k Skin Tone Class Distribution')
axes[1].set_xlabel('Skin Tone Class (6=Lightest, 1=Darkest)')

plt.tight_layout()
plt.show()
```

**Step 3: Apply the cells to the notebook**

Insert the new Markdown and Code cells after the first section in `FairFace_v3_Dataset_Visualization.ipynb`.

**Step 4: Run the notebook to verify it passes**

Run: `jupyter nbconvert --to notebook --execute "FairFace_v3_Dataset_Visualization.ipynb"`
Expected: Execution succeeds, generating the new subplots.

**Step 5: Commit**

```bash
git add "FairFace_v3_Dataset_Visualization.ipynb"
git commit -m "feat: add section comparing MILK10k and Fitzpatrick17k"
```

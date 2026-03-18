# Skin Tone Dataset Visualization Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a story-driven Jupyter Notebook ("The Paradigm Shift") explaining the transition from Fitzpatrick (FST) to Monk Skin Tone (MST-5) via ITA, visualizing data from MSKCC, MILK10k, FairFace, and UTKFace.

**Architecture:** A single Jupyter notebook with markdown narratives interweaved with seaborn/matplotlib visualizations. It will load data from `datasets/`, compute necessary aggregations, and plot distributions.

**Tech Stack:** Python, Jupyter, Pandas, Matplotlib, Seaborn

---

### Task 1: Initialize Notebook and Section 1 (The FST Baseline)

**Files:**
- Create: `FairFace_v3_Dataset_Visualization.ipynb`

**Step 1: Write the notebook skeleton and Section 1 code**
- Import `pandas`, `matplotlib.pyplot`, `seaborn`.
- Set visualization style (`sns.set_theme(style="whitegrid")`).
- Add Markdown explaining the FST baseline and the problem of middle-class collapse.
- Load `datasets/mskcc-skin-tone-labeling-dataset_metadata_2025-11-24.csv` and `datasets/milk10k_harvard.csv`.
- Create a visualization showing the distribution of FST labels in MSKCC and MILK10k.

**Step 2: Execute the notebook cells manually (test)**
Run the cells to ensure data loads and plots render correctly without errors.

**Step 3: Commit**
```bash
git add FairFace_v3_Dataset_Visualization.ipynb
git commit -m "feat: initialize notebook and add FST baseline visualization"
```

---

### Task 2: Implement Section 2 (The ITA & MST Solution)

**Files:**
- Modify: `FairFace_v3_Dataset_Visualization.ipynb`

**Step 1: Add Section 2 markdown and code**
- Add Markdown explaining ITA (Individual Typology Angle) and how it maps to MST-10 and MST-5.
- Load `datasets/master_mst_labels.csv`.
- Create a visualization: A histogram or KDE plot showing the continuous distribution of ITA scores, colored by their resulting MST-5 class, demonstrating clean boundaries.
- Add a table or text cell summarizing the mapping from ITA to MST-10 to MST-5.

**Step 2: Execute the notebook cells manually (test)**
Run the new cells to ensure the ITA vs MST plot renders correctly.

**Step 3: Commit**
```bash
git add FairFace_v3_Dataset_Visualization.ipynb
git commit -m "feat: add ITA and MST solution visualization to notebook"
```

---

### Task 3: Implement Section 3 (Dataset Comparisons)

**Files:**
- Modify: `FairFace_v3_Dataset_Visualization.ipynb`

**Step 1: Add Section 3 markdown and code**
- Add Markdown explaining the application of MST to FairFace and UTKFace datasets.
- Load `datasets/fairface_mst_labels.csv` and `datasets/utkface_mst_labels.csv` (or use `master_mst_labels.csv` if it contains both, filtering by `source`).
- Create a visualization: Side-by-side or overlaid bar charts/histograms comparing the MST-5 distribution between FairFace and UTKFace.
- Include demographic breakdowns if available (e.g., age/gender in FairFace vs UTKFace mapping to MST) to show representation.

**Step 2: Execute the notebook cells manually (test)**
Run the new cells to ensure the dataset comparison plots render correctly.

**Step 3: Commit**
```bash
git add FairFace_v3_Dataset_Visualization.ipynb
git commit -m "feat: add dataset comparison visualization to notebook"
```

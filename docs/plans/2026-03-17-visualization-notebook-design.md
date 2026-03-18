# Brainstorming Design: FairFace_v3_Dataset_Visualization.ipynb Updates

## Overview
Update the dataset visualization notebook to better illustrate the paradigm shift from Fitzpatrick to MST, and better analyze legacy datasets (MILK10k and Fitzpatrick17k).

## Requirements Addressed
1. Replace MILK10k in the first subplot figure with "MSKCC MST Distribution (Rater 2)".
2. Create a new section dedicated to comparing MILK10k and Fitzpatrick17k side-by-side.
3. Explicitly note that MILK10k's scale is not exactly FST and its ordering is inverted (6 = lightest, 1 = darkest).
4. Implement a unified color scheme mapping FST categories to MILK10k values, including an explicitly tracked N/A category.

## Notebook Structure Changes

### 1. Section 1 (MSKCC Focus)
- Update the code to plot three subplots exclusively from MSKCC data:
  - MSKCC FST Distribution
  - MSKCC MST Distribution (Rater 1)
  - MSKCC MST Distribution (Rater 2)
- Re-balance the text to highlight how the ambiguity between raters or labeling scales in MSKCC reinforces the shift to MST.

### 2. NEW Section 2: Legacy Datasets (MILK10k vs Fitzpatrick17k)
- Text block explaining MILK10k versus Fitzpatrick17k, noting MILK10k's inverted 6-to-1 scale.
- We will replace any unassigned or invalid labels (e.g., `-1` or `NaN`) with `N/A`.
- **Unified Color Mapping:**
  - I (MILK 6): `#FFE6CC`
  - II (MILK 5): `#FFD1A1`
  - III (MILK 4): `#CD906A`
  - IV (MILK 3): `#FFB374`
  - V (MILK 2): `#A36035`
  - VI (MILK 1): `#693F1B`
  - N/A: `#808080`
- Two side-by-side subplots using the exact same colors:
  - Subplot 1: Fitzpatrick17k `fitzpatrick_scale` (I-VI & N/A)
  - Subplot 2: MILK10k `skin_tone_class` (6-1 & N/A)

### 3. Section 3: The ITA and MST Solution
- (Unchanged from current code: MST-5 Master Label distribution).

### 4. Section 4: Dataset Comparisons
- (Unchanged from current code: FairFace vs UTKFace MST-5).

# Centroid Analysis: MSKCC vs. MILK10k (Closeup Only)

We analyzed the L* (Lightness) distributions of "clinical: close-up" images in both the **MSKCC** and **MILK10k** datasets to determine the appropriate centroids for generating synthetic labels.

## 📊 Summary of Findings

1.  **Hardcoded Centroids are Too Light:** The default centroids provided in the script (purportedly from MILK10k) are significantly **lighter** than the actual median values found in MILK10k closeups.
2.  **MSKCC vs. MILK10k Divergence:**
    *   **Light Types (I-II):** MSKCC images are *darker* than MILK10k.
    *   **Dark Types (V-VI):** MSKCC images are *lighter* than MILK10k (though MILK10k has very little data for Type VI).
3.  **Data Quality:** MSKCC has much better representation of Type VI (n=177) compared to MILK10k (n=6).

### Comparative Table (Median L*)

| Feature | Type I (Light) | Type II | Type III | Type IV | Type V | Type VI (Dark) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Legacy Hardcoded** | **83.0** | **75.7** | **66.7** | **61.9** | **53.7** | **52.1** |
| **MILK10k (Actual)** | 72.5 | 64.7 | 55.3 | 53.3 | 45.1 | 28.8* |
| **MSKCC (Actual)** | **65.1** | **63.9** | **65.1** | **57.8** | **47.1** | **42.7** |

*\*Note: MILK10k Type VI has only 6 samples, making this value unreliable.*

## 🚨 Critical Insight

The "Legacy Hardcoded" centroids are **completely misaligned** with closeup photography in both datasets.
- They expect Type I to be 83.0 (very bright), whereas real closeups are 65-72.
- They expect Type VI to be 52.1, whereas real closeups are 28-43.

Using the legacy centroids would result in **systematic mislabeling**:
- Real **Type I/II** images would be classified as **Type III/IV** by the legacy centroids.
- Real **Type VI** images would be correctly classified or classified as V.

## 💡 Recommendation: Use MSKCC-Aligned Smoothed Centroids

Since our target domain is MSKCC, we must align with the MSKCC distribution. However, because MSKCC lacks separability in Types I-III (all clustered around 64-65), we will enforce a **smoothed monotonic progression** that stays true to the MSKCC domain while resolving the overlaps.

**Proposed "MSKCC Smoothed" Centroids:**

```python
CENTROIDS = {
    5: 68.0,   # Type I   (Target: MSKCC ~65, MILK ~72)
    4: 66.0,   # Type II  (Target: MSKCC ~64, MILK ~64)
    3: 64.0,   # Type III (Pivotal Cluster Point)
    2: 58.0,   # Type IV  (Target: MSKCC ~58)
    1: 47.0,   # Type V   (Target: MSKCC ~47)
    0: 43.0,   # Type VI  (Target: MSKCC ~43)
}
```

This configuration:
1.  **Respects the Domain:** Aligns with the *actual* lightness of MSKCC closeups.
2.  **Fixes the Overlap:** Enforces `Type I > Type II > Type III` to generate distinct labels for FairFace.
3.  **Ignores Biased Legacy:** Discards the overly bright legacy values.

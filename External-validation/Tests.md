Test 1 — Tone Audit of ISIC 2019 Test Set
Purpose: Verify that the test-set tone distribution matches the training-set 130:1 audit (or quantify the difference).

Procedure:

text

Apply 3-class MSKCC-fine-tuned EfficientNet-B4 to all
ISIC 2019 test images (N=8,238 incl. UNK).
Apply 0.6 confidence filter.
Report: Light / Medium / Dark counts + proportions.
Report: how many were filtered out (low-confidence) and
the per-tone breakdown of the filtered set.
Expected output for the dissertation/slide:
A small table: Light X (Y%), Medium X (Y%), Dark X (Y%), Filtered X. Compare to training-set audit. If the ratio holds, you have replicated the finding on held-out data — strong defence against "is the audit just an artefact of the training partition?"
Test 2 — Tone-Stratified Diagnostic Evaluation (Core Test)
Purpose: The headline new result. Per-model performance on each tone subset.

Procedure:

text
For each of the 5 models (baseline, λ=0.0, 0.3, 0.7, 1.0):
For each tone subset (Light, Medium, Dark):
Compute on that subset:
- Macro F1
- Per-class F1 (8 classes)
- Per-class recall (especially MEL, BCC, SCC)
- Macro AUC
Store predictions for paired comparisons (Test 4).
Output: A 5 × 3 grid of macro F1 values, with the Dark column being the new evidence.

Test 3 — Bootstrap Confidence Intervals
Purpose: Quantify uncertainty so the small Dark subset doesn't undermine the findings.

Procedure:

text
For each (model, tone subset) combination:
Resample the subset with replacement, 1000 iterations.
Recompute macro F1 each iteration.
Report the 2.5th and 97.5th percentiles as 95% CI.
Why this matters: With ~30–50 Dark images, point estimates are noisy. CIs let you say "λ=1.0 macro F1 on Dark = 0.42 [0.34, 0.51]" and have an examiner take it seriously rather than say "that's just one number."

Slide presentation: Bar chart with error bars. If the CI for λ=1.0 doesn't overlap the CI for the baseline on the Dark subset, you have "statistically distinguishable improvement" as a defensible claim. If they overlap, you say "the augmentation's effect on Dark macro F1 is within the noise floor of this small subset — which is itself the project's central argument: the field cannot adequately evaluate fairness with the test data that exists."

Either outcome is defensible. That's what makes this test worth running.

Test 4 — McNemar's Test for Paired Model Comparison
Purpose: Bootstrap CIs are unpaired. McNemar's is paired — it asks "on the same images, did model A and model B disagree systematically?" This is a stronger test for model-vs-model comparison.

Procedure:

text
For each pair (baseline, λ=k) and each tone subset:
Build a 2×2 contingency table:
- Both correct
- Baseline correct, λ=k wrong
- Baseline wrong, λ=k correct
- Both wrong
Apply McNemar's chi-squared test (with continuity correction).
Report p-value.
Why this matters: This directly answers "is the augmentation's effect a real shift in which images get classified correctly, or random reshuffling?"

Test 5 — Confounder Audits
This is what makes the defence impenetrable. Every alternative explanation for any observed Dark-subset effect, pre-empted.

Test 5a — Class composition by tone:

text
For each tone subset, report the diagnostic-class distribution.
If the Dark subset is, say, 80% Naevus, then a model that's
better at Naevus will look "fairer" without any actual fairness
gain. Report this transparently.
Test 5b — Source/site confounder:

text
ISIC images have 'attribution' or 'source' metadata.
For each tone subset, report the distribution across sources.
If Dark images cluster in one source, lighting/protocol confounds
tone with site.
Test 5c — Confidence-stratified results:

text
Re-run Test 2 using only the highest-confidence tone predictions
(softmax >= 0.8 instead of 0.6).
If results are similar: tone-label noise isn't driving the effect.
If results differ: report both and discuss.
Test 5d — UNK-inclusion sensitivity:

text
Re-run Test 2 with UNK images included.
ISIC challenge convention excludes them; report whether including
them shifts any conclusion.
Slide handling: You don't need a slide for these. They go in the dissertation appendix and you mention them in Q&A: "We checked source attribution, class composition, confidence stratification, and UNK inclusion — none of them flip the directional conclusion. Full audit in Appendix [X]."

Test 6 — Tone Classifier Calibration on ISIC
Purpose: Address the "your tone classifier is only 67% accurate, why should we trust the audit?" question.
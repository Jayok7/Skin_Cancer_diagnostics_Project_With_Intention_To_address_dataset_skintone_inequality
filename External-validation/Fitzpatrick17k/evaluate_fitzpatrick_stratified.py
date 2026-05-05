#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_fitzpatrick_stratified.py
===================================
Stage 2 (CPU): Tests 2-5 of the empirical-rigour battery, ported to
Fitzpatrick17k. Consumes the manifest + prediction CSVs produced by
generate_fitzpatrick_predictions.py.

Tests:
  Test 2  -- Per-model x per-FST-tertile metrics (macro F1, per-class F1,
             per-class recall, macro AUC).
  Test 3  -- Bootstrap 95% CIs on macro F1 (1000 iters by default).
  Test 4  -- Paired McNemar's test: baseline vs each other model on each
             tertile.
  Test 5a -- Diagnostic-class composition by FST tertile.
  Test 5b -- Image source (dermaamin / atlas-dermatologico / other) by
             tertile.  Confounds tone with imaging protocol.
  Test 5c -- Re-run Tests 2/3 only on rows where the two FST raters agree
             (fitzpatrick_scale == fitzpatrick_centaur).
  Test 5d -- Re-run with STRICT mapping (invoke with --mapping STRICT and
             a separate output dir).

Headline output: macroF1 grid (5 models x 3 tertiles), bootstrap CI plot
on the Dark subset, McNemar's table.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, recall_score, roc_auc_score

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False
    print("WARN: statsmodels not installed; McNemar p-values computed by hand.")


# Class order MUST match generate_fitzpatrick_predictions.py
CLASSES_MODEL = sorted(['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC', 'UNK'])
CLASSES_EVAL  = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
TONE_ORDER    = ['Light', 'Medium', 'Dark']


# =========================================================================
# Loading
# =========================================================================
def load_manifest(path):
    df = pd.read_csv(path)
    needed = {"image_id", "true_idx", "true_class",
              "fitzpatrick_scale", "fst_tertile"}
    miss = needed - set(df.columns)
    if miss:
        sys.exit(f"manifest missing columns: {miss}")
    return df


def load_predictions(pred_dir, tag, model_classes):
    p = Path(pred_dir) / f"{tag}_predictions.csv"
    if not p.is_file():
        sys.exit(f"Missing predictions: {p}")
    df = pd.read_csv(p)
    prob_cols = [f"prob_{c}" for c in model_classes]
    miss = [c for c in prob_cols if c not in df.columns]
    if miss:
        sys.exit(f"{p} missing prob columns: {miss}")
    return df


def build_merged(manifest, pred_dfs, model_classes):
    """Inner-join manifest x predictions per model."""
    base_cols = ["image_id", "true_idx", "true_class",
                 "fitzpatrick_scale", "fitzpatrick_centaur",
                 "fst_tertile", "source", "qc_flag"]
    base_cols = [c for c in base_cols if c in manifest.columns]
    base = manifest[base_cols].copy()
    out = {}
    prob_cols = [f"prob_{c}" for c in model_classes]
    for tag, p in pred_dfs.items():
        out[tag] = base.merge(p[["image_id", "pred_idx"] + prob_cols],
                              on="image_id", how="inner")
    return out


# =========================================================================
# Metric helpers
# =========================================================================
def compute_metrics(y_true, y_pred, probs, model_classes, eval_classes):
    out = {}
    if len(y_true) == 0:
        return {"n": 0, "macro_f1": np.nan, "macro_auc": np.nan}
    out["n"] = int(len(y_true))
    eval_idx = [model_classes.index(c) for c in eval_classes]

    out["macro_f1"]    = f1_score(y_true, y_pred, labels=eval_idx,
                                  average="macro", zero_division=0)
    out["weighted_f1"] = f1_score(y_true, y_pred, labels=eval_idx,
                                  average="weighted", zero_division=0)

    per_f1  = f1_score(y_true, y_pred, labels=eval_idx,
                       average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=eval_idx,
                           average=None, zero_division=0)
    for i, c in enumerate(eval_classes):
        out[f"f1_{c}"]     = float(per_f1[i])
        out[f"recall_{c}"] = float(per_rec[i])

    try:
        yt_oh = np.zeros((len(y_true), len(model_classes)))
        valid = np.array(y_true) >= 0
        yt_oh[np.arange(len(y_true))[valid], np.array(y_true)[valid]] = 1
        aucs = []
        for i in eval_idx:
            if yt_oh[:, i].sum() == 0:
                continue
            aucs.append(roc_auc_score(yt_oh[:, i], probs[:, i]))
        out["macro_auc"] = float(np.mean(aucs)) if aucs else np.nan
    except Exception:
        out["macro_auc"] = np.nan
    return out


def bootstrap_macro_f1(y_true, y_pred, model_classes, eval_classes,
                       n_iters=1000, seed=42):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    eval_idx = [model_classes.index(c) for c in eval_classes]
    rng = np.random.default_rng(seed)
    N = len(y_true)
    point = f1_score(y_true, y_pred, labels=eval_idx,
                     average="macro", zero_division=0)
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    samples = np.empty(n_iters, dtype=np.float32)
    for i in range(n_iters):
        idx = rng.integers(0, N, size=N)
        samples[i] = f1_score(yt[idx], yp[idx], labels=eval_idx,
                              average="macro", zero_division=0)
    return (float(point),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    yt = np.asarray(y_true); ya = np.asarray(y_pred_a); yb = np.asarray(y_pred_b)
    a_correct = (ya == yt)
    b_correct = (yb == yt)
    n_both    = int(( a_correct &  b_correct).sum())
    n_only_a  = int(( a_correct & ~b_correct).sum())
    n_only_b  = int((~a_correct &  b_correct).sum())
    n_neither = int((~a_correct & ~b_correct).sum())
    table = [[n_both, n_only_a], [n_only_b, n_neither]]

    if HAVE_STATSMODELS:
        result = mcnemar(table, exact=(n_only_a + n_only_b) < 25, correction=True)
        stat, p = float(result.statistic), float(result.pvalue)
    else:
        b, c = n_only_a, n_only_b
        if b + c == 0:
            stat, p = 0.0, 1.0
        else:
            stat = (abs(b - c) - 1) ** 2 / (b + c)
            from math import erfc, sqrt
            p = float(erfc(sqrt(stat) / sqrt(2)))
    return dict(both_correct=n_both, only_baseline_correct=n_only_a,
                only_other_correct=n_only_b, neither_correct=n_neither,
                statistic=stat, p_value=p)


# =========================================================================
# Plot helpers
# =========================================================================
def plot_macroF1_with_ci(ci_df, tone, tags, out_path, title):
    sub = ci_df[ci_df["tone"] == tone].set_index("model").reindex(tags)
    if sub.empty or sub["macro_f1"].isna().all():
        return
    x = np.arange(len(tags))
    f1_vals = sub["macro_f1"].values.astype(float)
    lo_vals = sub["ci_low"].values.astype(float)
    hi_vals = sub["ci_high"].values.astype(float)
    n_vals  = sub["n"].values

    yerr = np.vstack([f1_vals - lo_vals, hi_vals - f1_vals])
    yerr = np.clip(yerr, 0, None)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x, f1_vals, yerr=yerr, capsize=6, color="steelblue",
           edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("Macro F1")
    upper = float(np.nanmax(hi_vals)) if not np.all(np.isnan(hi_vals)) else 0.5
    ax.set_ylim(0, max(0.1, upper * 1.20))
    n_str = f"N={int(n_vals[0])}" if len(n_vals) and not np.isnan(n_vals[0]) else "N=?"
    ax.set_title(f"{title}\n({n_str})")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (m, lo, hi) in enumerate(zip(f1_vals, lo_vals, hi_vals)):
        if not np.isnan(m):
            ax.text(i, hi + upper * 0.02,
                    f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_macroF1_grouped(ci_df, tags, out_path):
    """All three tertiles side-by-side per model -- the headline figure."""
    fig, ax = plt.subplots(figsize=(max(9, len(tags) * 2.0), 6))
    x = np.arange(len(tags))
    w = 0.25
    colors = {"Light": "#F4D03F", "Medium": "#E67E22", "Dark": "#5D4037"}

    for j, t in enumerate(TONE_ORDER):
        sub = ci_df[ci_df["tone"] == t].set_index("model").reindex(tags)
        f1 = sub["macro_f1"].values.astype(float)
        lo = sub["ci_low"].values.astype(float)
        hi = sub["ci_high"].values.astype(float)
        n  = sub["n"].values
        yerr = np.vstack([f1 - lo, hi - f1]); yerr = np.clip(yerr, 0, None)
        bars = ax.bar(x + (j - 1) * w, f1, w, yerr=yerr, capsize=4,
                      color=colors[t], edgecolor="black",
                      linewidth=0.6, label=f"{t}")
        for bar, v, ni in zip(bars, f1, n):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}\n(n={int(ni)})",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Fitzpatrick17k -- Macro F1 by FST Tertile (95% bootstrap CI)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="FST Tertile", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================================
# Test runners
# =========================================================================
def run_test2(merged_dfs, tags, model_classes, eval_classes, out_dir):
    rows = []
    prob_cols = [f"prob_{c}" for c in model_classes]
    for tag, dfm in merged_dfs.items():
        for tone in TONE_ORDER + ["Overall"]:
            sub = dfm if tone == "Overall" else dfm[dfm["fst_tertile"] == tone]
            if len(sub) == 0:
                continue
            yt = sub["true_idx"].values
            valid = yt >= 0
            yt = yt[valid]
            yp = sub.loc[valid, "pred_idx"].values
            probs = sub.loc[valid, prob_cols].values
            m = compute_metrics(yt, yp, probs, model_classes, eval_classes)
            m.update({"model": tag, "tone": tone})
            rows.append(m)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_dir / "test2_metrics.csv", index=False)
    grid = long_df.pivot(index="model", columns="tone", values="macro_f1")
    grid = grid.reindex(index=tags, columns=TONE_ORDER + ["Overall"])
    grid.to_csv(out_dir / "test2_macroF1_grid.csv")
    return long_df, grid


def run_test3(merged_dfs, tags, model_classes, eval_classes, out_dir,
              n_iters=1000):
    rows = []
    for tag, dfm in merged_dfs.items():
        for tone in TONE_ORDER + ["Overall"]:
            sub = dfm if tone == "Overall" else dfm[dfm["fst_tertile"] == tone]
            yt = sub["true_idx"].values
            valid = yt >= 0
            yt = yt[valid]
            yp = sub.loc[valid, "pred_idx"].values
            point, lo, hi = bootstrap_macro_f1(yt, yp,
                                               model_classes, eval_classes,
                                               n_iters=n_iters)
            rows.append({"model": tag, "tone": tone, "n": int(valid.sum()),
                         "macro_f1": point, "ci_low": lo, "ci_high": hi})
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(out_dir / "test3_bootstrap_ci.csv", index=False)
    for tone in TONE_ORDER + ["Overall"]:
        plot_macroF1_with_ci(
            ci_df, tone=tone, tags=tags,
            out_path=out_dir / f"test3_macroF1_{tone.lower()}.png",
            title=f"Macro F1 on {tone} subset (95% bootstrap CI)")
    plot_macroF1_grouped(ci_df, tags=tags,
                         out_path=out_dir / "test3_macroF1_grouped.png")
    return ci_df


def run_test4(merged_dfs, tags, baseline_tag, out_dir):
    rows = []
    base = merged_dfs[baseline_tag]
    for tag in tags:
        if tag == baseline_tag:
            continue
        other = merged_dfs[tag]
        merged = base.merge(other[["image_id", "pred_idx"]],
                            on="image_id", suffixes=("_base", "_other"))
        for tone in TONE_ORDER + ["Overall"]:
            sub = merged if tone == "Overall" else merged[merged["fst_tertile"] == tone]
            yt = sub["true_idx"].values
            valid = yt >= 0
            yt = yt[valid]
            ya = sub.loc[valid, "pred_idx_base"].values
            yb = sub.loc[valid, "pred_idx_other"].values
            if len(yt) == 0:
                continue
            r = mcnemar_test(yt, ya, yb)
            r.update({"baseline": baseline_tag, "other": tag,
                      "tone": tone, "n": int(len(yt))})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test4_mcnemar.csv", index=False)
    return df


def run_test5a(merged_dfs, tags, eval_classes, out_dir):
    """Diagnostic-class composition by tertile."""
    dfm = merged_dfs[tags[0]]
    rows = []
    for tone in TONE_ORDER + ["Overall"]:
        sub = dfm if tone == "Overall" else dfm[dfm["fst_tertile"] == tone]
        sub = sub[sub["true_idx"] >= 0]
        n = len(sub)
        rec = {"tone": tone, "n": n}
        for c in eval_classes:
            cnt = int((sub["true_class"] == c).sum())
            rec[f"{c}_count"] = cnt
            rec[f"{c}_pct"]   = (cnt / n * 100) if n > 0 else 0.0
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test5a_class_by_tone.csv", index=False)
    return df


def run_test5b(merged_dfs, tags, out_dir):
    """Image-source composition by tertile (Fitzpatrick `url` host)."""
    dfm = merged_dfs[tags[0]]
    if "source" not in dfm.columns:
        print("  [Test 5b] No 'source' column in manifest -- skipping")
        return None
    rows = []
    for tone in TONE_ORDER + ["Overall"]:
        sub = dfm if tone == "Overall" else dfm[dfm["fst_tertile"] == tone]
        sub = sub[sub["true_idx"] >= 0]
        counts = sub["source"].fillna("UNKNOWN").value_counts()
        for src, c in counts.items():
            rows.append({"tone": tone, "source": src, "count": int(c),
                         "pct": float(c / max(len(sub), 1) * 100)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test5b_source_by_tone.csv", index=False)
    return df


def run_test5c(manifest, pred_dfs, tags, model_classes, eval_classes,
               out_dir, n_iters=1000):
    """Re-run Test 2/3 only on rows where both FST raters agree."""
    if "fitzpatrick_centaur" not in manifest.columns:
        print("  [Test 5c] No fitzpatrick_centaur column -- skipping")
        return None

    agree_mask = (
        manifest["fitzpatrick_scale"].notna() &
        manifest["fitzpatrick_centaur"].notna() &
        (manifest["fitzpatrick_scale"] == manifest["fitzpatrick_centaur"])
    )
    kept = manifest[agree_mask].copy()
    if len(kept) == 0:
        print("  [Test 5c] No rows where raters agree -- skipping")
        return None
    print(f"  [Test 5c] Rater agreement subset: N={len(kept)} "
          f"({len(kept)/len(manifest):.1%})")

    merged_dfs = build_merged(kept, pred_dfs, model_classes)
    rows_metrics, rows_ci = [], []
    prob_cols = [f"prob_{c}" for c in model_classes]
    for tag, dfm in merged_dfs.items():
        for tone in TONE_ORDER + ["Overall"]:
            sub = dfm if tone == "Overall" else dfm[dfm["fst_tertile"] == tone]
            yt = sub["true_idx"].values
            valid = yt >= 0
            yt = yt[valid]
            yp = sub.loc[valid, "pred_idx"].values
            probs = sub.loc[valid, prob_cols].values
            m = compute_metrics(yt, yp, probs, model_classes, eval_classes)
            m.update({"model": tag, "tone": tone})
            rows_metrics.append(m)
            point, lo, hi = bootstrap_macro_f1(yt, yp,
                                               model_classes, eval_classes,
                                               n_iters=n_iters)
            rows_ci.append({"model": tag, "tone": tone, "n": int(valid.sum()),
                            "macro_f1": point, "ci_low": lo, "ci_high": hi})

    pd.DataFrame(rows_metrics).to_csv(
        out_dir / "test5c_metrics_rater_agree.csv", index=False)
    ci_df = pd.DataFrame(rows_ci)
    ci_df.to_csv(out_dir / "test5c_bootstrap_ci_rater_agree.csv", index=False)
    plot_macroF1_grouped(ci_df, tags=tags,
                         out_path=out_dir / "test5c_macroF1_grouped_rater_agree.png")
    return ci_df


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="Directory with manifest.csv and <tag>_predictions.csv")
    ap.add_argument("--tags", nargs="+",
                    default=["orig", "lambda00", "lambda03", "lambda07", "lambda10"])
    ap.add_argument("--baseline-tag", default="orig")
    ap.add_argument("--output-dir", default="outputs/fitzpatrick_stratified_eval")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--model-classes", type=str, default=",".join(CLASSES_MODEL))
    ap.add_argument("--eval-classes",  type=str, default=",".join(CLASSES_EVAL))
    return ap.parse_args()


def main():
    args = parse_args()
    model_classes = args.model_classes.split(",")
    eval_classes  = args.eval_classes.split(",")

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {pred_dir / 'manifest.csv'}...")
    manifest = load_manifest(pred_dir / "manifest.csv")
    print(f"  Manifest rows: {len(manifest)}")
    print(f"  Mapping: {manifest['mapping'].iloc[0] if 'mapping' in manifest.columns else 'unknown'}")

    print(f"\nLoading {len(args.tags)} prediction CSVs...")
    pred_dfs = {tag: load_predictions(pred_dir, tag, model_classes)
                for tag in args.tags}

    merged_dfs = build_merged(manifest, pred_dfs, model_classes)
    sample_tag = args.tags[0]
    print(f"\nMerged rows for '{sample_tag}': {len(merged_dfs[sample_tag])}")
    if len(merged_dfs[sample_tag]) == 0:
        sys.exit("Empty merge -- check image_id alignment.")

    print("  Per-tertile N:")
    for t in TONE_ORDER:
        n = (merged_dfs[sample_tag]["fst_tertile"] == t).sum()
        print(f"    {t:8s}: {n:5d}")

    print("\n[Test 2] Tone-stratified diagnostic evaluation...")
    long_df, grid = run_test2(merged_dfs, args.tags,
                              model_classes, eval_classes, out_dir)
    print("\n  Macro F1 grid:")
    print(grid.round(4).to_string())

    print(f"\n[Test 3] Bootstrap CIs ({args.bootstrap_iters} iters)...")
    ci_df = run_test3(merged_dfs, args.tags, model_classes, eval_classes,
                      out_dir, n_iters=args.bootstrap_iters)
    print("\n  Bootstrap macro F1:")
    for _, r in ci_df.iterrows():
        print(f"    {r['model']:10s} | {r['tone']:7s} | N={int(r['n']):4d} | "
              f"F1={r['macro_f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    print(f"\n[Test 4] McNemar's paired test (baseline = {args.baseline_tag})...")
    mc_df = run_test4(merged_dfs, args.tags, args.baseline_tag, out_dir)
    if not mc_df.empty:
        for _, r in mc_df.iterrows():
            sig = ('***' if r['p_value'] < 0.001 else
                   '**'  if r['p_value'] < 0.01  else
                   '*'   if r['p_value'] < 0.05  else 'ns')
            print(f"    {r['baseline']:>8s} vs {r['other']:<10s} | {r['tone']:7s} | "
                  f"N={int(r['n']):4d} | b={r['only_baseline_correct']:4d} "
                  f"c={r['only_other_correct']:4d} | "
                  f"p={r['p_value']:.4f} {sig}")

    print("\n[Test 5a] Diagnostic-class composition by tertile...")
    comp_df = run_test5a(merged_dfs, args.tags, eval_classes, out_dir)
    print(comp_df[["tone", "n"] + [f"{c}_pct" for c in eval_classes]]
          .round(1).to_string(index=False))

    print("\n[Test 5b] Image-source composition by tertile...")
    src_df = run_test5b(merged_dfs, args.tags, out_dir)
    if src_df is not None:
        pivot = src_df.pivot(index="tone", columns="source",
                             values="pct").fillna(0).round(1)
        print(pivot.to_string())

    print("\n[Test 5c] Re-running Tests 2/3 on rater-agreement subset...")
    run_test5c(manifest, pred_dfs, args.tags,
               model_classes, eval_classes, out_dir,
               n_iters=args.bootstrap_iters)

    print(f"\n[Test 5d] Re-invoke generate_fitzpatrick_predictions.py with "
          f"--mapping STRICT and a fresh --output-dir to run the strict-mapping arm.")
    print(f"\n[OK] All tests written to {out_dir}/")


if __name__ == "__main__":
    main()
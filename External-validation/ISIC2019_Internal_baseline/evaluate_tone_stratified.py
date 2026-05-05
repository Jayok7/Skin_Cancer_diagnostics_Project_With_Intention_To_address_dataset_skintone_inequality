#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_tone_stratified.py
============================
Tests 2-5 of the empirical-rigour battery. Pure analysis: no GPU.
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


# Class order MUST match the trainer (alphabetical sort).
# The trained model is 9-class (UNK was a real training column);
# we evaluate over the 8 real diagnostic classes.
CLASSES_MODEL = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'UNK', 'VASC']
CLASSES_EVAL  = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
TONE_ORDER    = ['Light', 'Medium', 'Dark']


# =========================================================================
# Loading
# =========================================================================

def load_ground_truth(gt_csv, eval_classes, include_unk=False):
    df = pd.read_csv(gt_csv)
    id_col = 'image' if 'image' in df.columns else (
             'image_id' if 'image_id' in df.columns else df.columns[0])
    df = df.rename(columns={id_col: 'image_id'})

    has_unk = 'UNK' in df.columns
    if has_unk:
        unk_mask = df['UNK'].astype(float) > 0.5
        n_unk = int(unk_mask.sum())
        print(f"  Detected UNK column: {n_unk} UNK rows")
        if not include_unk:
            df = df[~unk_mask].copy()
            print(f"  Dropped UNK rows (use --include-unk to keep)")

    cls_present = [c for c in eval_classes if c in df.columns]
    if len(cls_present) != len(eval_classes):
        missing = set(eval_classes) - set(cls_present)
        sys.exit(f"GT csv missing class columns: {missing}")

    arr = df[eval_classes].values.astype(float)
    df['true_idx'] = arr.argmax(axis=1)
    df['true_class'] = [eval_classes[i] for i in df['true_idx']]
    if has_unk and include_unk:
        df.loc[df['UNK'].astype(float) > 0.5, 'true_idx'] = -1
        df.loc[df['true_idx'] == -1, 'true_class'] = 'UNK'

    return df[['image_id', 'true_idx', 'true_class']].reset_index(drop=True)


def load_tone(tone_csv, conf_threshold=0.6):
    df = pd.read_csv(tone_csv)
    if 'image_id' not in df.columns and 'file' in df.columns:
        df['image_id'] = df['file'].apply(lambda x: Path(x).stem)
    keep = df[df['confidence_val'] >= conf_threshold].copy()
    return df, keep


def load_predictions(pred_dir, tag, model_classes):
    p = Path(pred_dir) / f"{tag}_predictions.csv"
    if not p.is_file():
        sys.exit(f"Missing predictions: {p}")
    df = pd.read_csv(p)
    prob_cols = [f'prob_{c}' for c in model_classes]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        sys.exit(f"{p} missing prob columns: {missing}")
    return df


def build_merged(tone_kept_df, gt_df, pred_dfs_full, model_classes):
    """Inner-join tone Ã- GT Ã- predictions per model."""
    base = tone_kept_df[['image_id', 'mst_name', 'confidence_val']].merge(
        gt_df[['image_id', 'true_idx', 'true_class']], on='image_id', how='inner')
    out = {}
    prob_cols = [f'prob_{c}' for c in model_classes]
    for tag, p in pred_dfs_full.items():
        out[tag] = base.merge(p[['image_id', 'pred_idx'] + prob_cols],
                              on='image_id', how='inner')
    return out


# =========================================================================
# Core metric helpers
# =========================================================================

def compute_metrics(y_true, y_pred, probs, model_classes, eval_classes):
    out = {}
    if len(y_true) == 0:
        return {'n': 0, 'macro_f1': np.nan, 'macro_auc': np.nan}
    out['n'] = len(y_true)
    eval_idx = [model_classes.index(c) for c in eval_classes]

    out['macro_f1'] = f1_score(y_true, y_pred, labels=eval_idx,
                                average='macro', zero_division=0)
    out['weighted_f1'] = f1_score(y_true, y_pred, labels=eval_idx,
                                   average='weighted', zero_division=0)

    per_f1 = f1_score(y_true, y_pred, labels=eval_idx,
                       average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, labels=eval_idx,
                            average=None, zero_division=0)
    for i, c in enumerate(eval_classes):
        out[f'f1_{c}'] = per_f1[i]
        out[f'recall_{c}'] = per_rec[i]

    try:
        present = sorted(set(y_true) - {-1})
        if len(present) >= 2:
            yt_oh = np.zeros((len(y_true), len(model_classes)))
            valid = np.array(y_true) >= 0
            yt_oh[np.arange(len(y_true))[valid], np.array(y_true)[valid]] = 1
            aucs = []
            for i in eval_idx:
                if yt_oh[:, i].sum() == 0:
                    continue
                aucs.append(roc_auc_score(yt_oh[:, i], probs[:, i]))
            out['macro_auc'] = float(np.mean(aucs)) if aucs else np.nan
        else:
            out['macro_auc'] = np.nan
    except Exception:
        out['macro_auc'] = np.nan
    return out


def bootstrap_macro_f1(y_true, y_pred, model_classes, eval_classes,
                       n_iters=1000, seed=42):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan
    eval_idx = [model_classes.index(c) for c in eval_classes]
    rng = np.random.default_rng(seed)
    N = len(y_true)
    point = f1_score(y_true, y_pred, labels=eval_idx,
                     average='macro', zero_division=0)
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    samples = np.empty(n_iters, dtype=np.float32)
    for i in range(n_iters):
        idx = rng.integers(0, N, size=N)
        samples[i] = f1_score(yt[idx], yp[idx], labels=eval_idx,
                               average='macro', zero_division=0)
    return float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    yt = np.asarray(y_true); ya = np.asarray(y_pred_a); yb = np.asarray(y_pred_b)
    a_correct = (ya == yt)
    b_correct = (yb == yt)
    n_both    = int((a_correct & b_correct).sum())
    n_only_a  = int((a_correct & ~b_correct).sum())
    n_only_b  = int((~a_correct & b_correct).sum())
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


def plot_macroF1_with_ci(ci_df, tone, tags, out_path, title):
    sub = ci_df[ci_df['tone'] == tone].set_index('model').reindex(tags)
    if sub.empty or sub['macro_f1'].isna().all():
        return
    x = np.arange(len(tags))
    f1_vals = sub['macro_f1'].values.astype(float)
    lo_vals = sub['ci_low'].values.astype(float)
    hi_vals = sub['ci_high'].values.astype(float)
    n_vals  = sub['n'].values

    yerr = np.vstack([f1_vals - lo_vals, hi_vals - f1_vals])
    yerr = np.clip(yerr, 0, None)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x, f1_vals, yerr=yerr, capsize=6, color='steelblue',
           edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel('Macro F1')
    upper = float(np.nanmax(hi_vals)) if not np.all(np.isnan(hi_vals)) else 0.5
    ax.set_ylim(0, max(0.1, upper * 1.20))
    n_str = f"N={int(n_vals[0])}" if len(n_vals) > 0 and not np.isnan(n_vals[0]) else "N=?"
    ax.set_title(f"{title}\n({n_str})")
    ax.grid(True, axis='y', alpha=0.3)
    for i, (m, lo, hi) in enumerate(zip(f1_vals, lo_vals, hi_vals)):
        if not np.isnan(m):
            ax.text(i, hi + upper * 0.02,
                    f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================================
# Test runners (all module-level)
# =========================================================================

def run_test2(merged_dfs, tags, tone_order, model_classes, eval_classes, out_dir):
    """Per-model x per-tone metrics."""
    rows = []
    prob_cols = [f'prob_{c}' for c in model_classes]
    for tag, dfm in merged_dfs.items():
        for tone in tone_order + ['Overall']:
            sub = dfm if tone == 'Overall' else dfm[dfm['mst_name'] == tone]
            if len(sub) == 0:
                continue
            yt = sub['true_idx'].values
            valid = yt >= 0
            yt = yt[valid]
            yp = sub.loc[valid, 'pred_idx'].values
            probs = sub.loc[valid, prob_cols].values
            m = compute_metrics(yt, yp, probs, model_classes, eval_classes)
            m.update({'model': tag, 'tone': tone})
            rows.append(m)
    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_dir / 'test2_metrics.csv', index=False)
    grid = long_df.pivot(index='model', columns='tone', values='macro_f1')
    grid = grid.reindex(index=tags, columns=tone_order + ['Overall'])
    grid.to_csv(out_dir / 'test2_macroF1_grid.csv')
    return long_df, grid


def run_test3(merged_dfs, tags, tone_order, model_classes, eval_classes,
              out_dir, n_iters=1000):
    """Bootstrap 95% CIs on macro F1."""
    rows = []
    for tag, dfm in merged_dfs.items():
        for tone in tone_order + ['Overall']:
            sub = dfm if tone == 'Overall' else dfm[dfm['mst_name'] == tone]
            yt = sub['true_idx'].values
            valid = yt >= 0
            yt = yt[valid]
            yp = sub.loc[valid, 'pred_idx'].values
            point, lo, hi = bootstrap_macro_f1(yt, yp, model_classes, eval_classes,
                                                n_iters=n_iters)
            rows.append({'model': tag, 'tone': tone, 'n': int(valid.sum()),
                         'macro_f1': point, 'ci_low': lo, 'ci_high': hi})
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(out_dir / 'test3_bootstrap_ci.csv', index=False)
    for tone in tone_order + ['Overall']:
        plot_macroF1_with_ci(ci_df, tone=tone, tags=tags,
                             out_path=out_dir / f'test3_macroF1_{tone.lower()}.png',
                             title=f'Macro F1 on {tone} subset (95% bootstrap CI)')
    return ci_df


def run_test4(merged_dfs, tags, baseline_tag, tone_order, out_dir):
    """Paired McNemar's test: baseline vs each other model, per tone subset."""
    rows = []
    base = merged_dfs[baseline_tag]
    for tag in tags:
        if tag == baseline_tag:
            continue
        other = merged_dfs[tag]
        merged = base.merge(other[['image_id', 'pred_idx']],
                            on='image_id', suffixes=('_base', '_other'))
        for tone in tone_order + ['Overall']:
            sub = merged if tone == 'Overall' else merged[merged['mst_name'] == tone]
            yt = sub['true_idx'].values
            valid = yt >= 0
            yt = yt[valid]
            ya = sub.loc[valid, 'pred_idx_base'].values
            yb = sub.loc[valid, 'pred_idx_other'].values
            if len(yt) == 0:
                continue
            r = mcnemar_test(yt, ya, yb)
            r.update({'baseline': baseline_tag, 'other': tag,
                      'tone': tone, 'n': len(yt)})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'test4_mcnemar.csv', index=False)
    return df


def run_test5a(merged_dfs, tags, eval_classes, tone_order, out_dir):
    """Diagnostic-class composition by tone (Test 5a)."""
    dfm = merged_dfs[tags[0]]
    rows = []
    for tone in tone_order + ['Overall']:
        sub = dfm if tone == 'Overall' else dfm[dfm['mst_name'] == tone]
        sub = sub[sub['true_idx'] >= 0]
        n = len(sub)
        rec = {'tone': tone, 'n': n}
        for c in eval_classes:
            cnt = int((sub['true_class'] == c).sum())
            rec[f'{c}_count'] = cnt
            rec[f'{c}_pct'] = (cnt / n * 100) if n > 0 else 0.0
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'test5a_class_by_tone.csv', index=False)
    return df


def run_test5b(merged_dfs, tags, tone_order, metadata_csv, out_dir):
    """Source / site composition by tone (Test 5b)."""
    if not metadata_csv or not Path(metadata_csv).is_file():
        print("  [Test 5b] No metadata CSV - skipping site/attribution audit")
        return None
    meta = pd.read_csv(metadata_csv)
    id_col = 'image' if 'image' in meta.columns else (
             'isic_id' if 'isic_id' in meta.columns else meta.columns[0])
    meta = meta.rename(columns={id_col: 'image_id'})

    candidates = ['attribution', 'lesion_id', 'anatom_site_general',
                  'anatom_site_general_challenge', 'dataset', 'source']
    site_col = next((c for c in candidates if c in meta.columns), None)
    if site_col is None:
        print(f"  [Test 5b] No site/attribution column. Available: {list(meta.columns)}")
        return None
    print(f"  [Test 5b] Using site column: {site_col}")

    dfm = merged_dfs[tags[0]].merge(meta[['image_id', site_col]],
                                     on='image_id', how='left')
    rows = []
    for tone in tone_order + ['Overall']:
        sub = dfm if tone == 'Overall' else dfm[dfm['mst_name'] == tone]
        sub = sub[sub['true_idx'] >= 0]
        counts = sub[site_col].fillna('UNKNOWN').value_counts()
        for site, c in counts.items():
            rows.append({'tone': tone, 'site': site, 'count': int(c),
                         'pct': float(c / max(len(sub), 1) * 100)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'test5b_site_by_tone.csv', index=False)
    return df


def run_test5c(tone_full_df, gt_df, pred_dfs_full, tags,
               model_classes, eval_classes, tone_order, out_dir,
               strict_threshold=0.8, n_iters=1000):
    """Re-run Test 2 + Test 3 at strict tone confidence (Test 5c)."""
    kept = tone_full_df[tone_full_df['confidence_val'] >= strict_threshold].copy()
    if len(kept) == 0:
        print(f"  [Test 5c] No tone predictions pass conf >= {strict_threshold}")
        return None
    print(f"  [Test 5c] Strict tone confidence >= {strict_threshold}: "
          f"N={len(kept)} ({len(kept)/len(tone_full_df):.1%})")

    merged_dfs = build_merged(kept, gt_df, pred_dfs_full, model_classes)
    rows_metrics, rows_ci = [], []
    prob_cols = [f'prob_{c}' for c in model_classes]
    for tag, dfm in merged_dfs.items():
        for tone in tone_order + ['Overall']:
            sub = dfm if tone == 'Overall' else dfm[dfm['mst_name'] == tone]
            yt = sub['true_idx'].values
            valid = yt >= 0
            yt = yt[valid]
            yp = sub.loc[valid, 'pred_idx'].values
            probs = sub.loc[valid, prob_cols].values
            m = compute_metrics(yt, yp, probs, model_classes, eval_classes)
            m.update({'model': tag, 'tone': tone}); rows_metrics.append(m)
            point, lo, hi = bootstrap_macro_f1(yt, yp, model_classes, eval_classes,
                                                n_iters=n_iters)
            rows_ci.append({'model': tag, 'tone': tone, 'n': int(valid.sum()),
                            'macro_f1': point, 'ci_low': lo, 'ci_high': hi})
    pd.DataFrame(rows_metrics).to_csv(out_dir / 'test5c_metrics_hiconf.csv', index=False)
    pd.DataFrame(rows_ci).to_csv(out_dir / 'test5c_bootstrap_ci_hiconf.csv', index=False)
    return rows_metrics, rows_ci


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tone-csv', required=True)
    ap.add_argument('--gt-csv', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--tags', nargs='+',
                    default=['orig', 'lambda00', 'lambda03', 'lambda07', 'lambda10'])
    ap.add_argument('--baseline-tag', default='orig')
    ap.add_argument('--metadata-csv', default=None)
    ap.add_argument('--output-dir', default='outputs/tone_stratified_eval')
    ap.add_argument('--tone-conf', type=float, default=0.6)
    ap.add_argument('--tone-conf-strict', type=float, default=0.8)
    ap.add_argument('--include-unk', action='store_true')
    ap.add_argument('--bootstrap-iters', type=int, default=1000)
    ap.add_argument('--model-classes', type=str, default=','.join(CLASSES_MODEL))
    ap.add_argument('--eval-classes',  type=str, default=','.join(CLASSES_EVAL))
    ap.add_argument('--require-no-vignette', action='store_true',
                    help='Restrict analysis to images flagged as has_vignette=False '
                         '(sensitivity arm to control for site/equipment confounding)')
    return ap.parse_args()


def main():
    args = parse_args()
    model_classes = args.model_classes.split(',')
    eval_classes  = args.eval_classes.split(',')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.include_unk:
        out_dir = out_dir / 'with_unk'
        out_dir.mkdir(parents=True, exist_ok=True)
        print("[Test 5d] Running WITH UNK rows included")

    print("Loading ground truth...")
    gt_df = load_ground_truth(args.gt_csv, eval_classes, include_unk=args.include_unk)
    print(f"  GT rows: {len(gt_df)}")

    print(f"\nLoading tone CSV (full, all confidences)...")
    tone_full, tone_kept = load_tone(args.tone_csv, conf_threshold=args.tone_conf)
    if args.require_no_vignette:
        if 'has_vignette' not in tone_full.columns:
            sys.exit("--require-no-vignette set but 'has_vignette' missing from tone CSV")
        n_before_full = len(tone_full)
        n_before_kept = len(tone_kept)
        tone_full = tone_full[~tone_full['has_vignette']].copy()
        tone_kept = tone_kept[~tone_kept['has_vignette']].copy()
        print(f"  [Vignette filter] tone_full: {n_before_full} -> {len(tone_full)} "
              f"(dropped {n_before_full - len(tone_full)} vignette-flagged)")
        print(f"  [Vignette filter] tone_kept: {n_before_kept} -> {len(tone_kept)} "
              f"(dropped {n_before_kept - len(tone_kept)} vignette-flagged)")
        out_dir = out_dir / 'no_vignette'
        out_dir.mkdir(parents=True, exist_ok=True)
        print("  [Vignette filter] Output redirected to:", out_dir)
    print(f"  Tone rows: {len(tone_full)} | passing conf>={args.tone_conf}: {len(tone_kept)}")

    print(f"\nLoading {len(args.tags)} prediction CSVs from {args.pred_dir}...")
    pred_dfs_full = {tag: load_predictions(args.pred_dir, tag, model_classes)
                     for tag in args.tags}

    merged_dfs = build_merged(tone_kept, gt_df, pred_dfs_full, model_classes)
    sample_tag = args.tags[0]
    print(f"\nMerged rows for {sample_tag}: {len(merged_dfs[sample_tag])}")
    if len(merged_dfs[sample_tag]) == 0:
        sys.exit("Empty merge - check image_id alignment.")

    print("  Per-tone N:")
    for tone in TONE_ORDER:
        n = (merged_dfs[sample_tag]['mst_name'] == tone).sum()
        print(f"    {tone:8s}: {n:5d}")

    print("\n[Test 2] Tone-stratified diagnostic evaluation...")
    long_df, grid = run_test2(merged_dfs, args.tags, TONE_ORDER,
                               model_classes, eval_classes, out_dir)
    print("\n  Macro F1 grid:")
    print(grid.round(4).to_string())

    print(f"\n[Test 3] Bootstrap CIs ({args.bootstrap_iters} iters)...")
    ci_df = run_test3(merged_dfs, args.tags, TONE_ORDER,
                      model_classes, eval_classes, out_dir,
                      n_iters=args.bootstrap_iters)
    print("\n  Bootstrap macro F1 (selected):")
    for _, r in ci_df.iterrows():
        print(f"    {r['model']:10s} | {r['tone']:7s} | N={int(r['n']):4d} | "
              f"F1={r['macro_f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    print(f"\n[Test 4] McNemar's paired test (baseline = {args.baseline_tag})...")
    mc_df = run_test4(merged_dfs, args.tags, args.baseline_tag, TONE_ORDER, out_dir)
    if not mc_df.empty:
        for _, r in mc_df.iterrows():
            sig = '***' if r['p_value'] < 0.001 else ('**' if r['p_value'] < 0.01
                  else ('*' if r['p_value'] < 0.05 else 'ns'))
            print(f"    {r['baseline']:>8s} vs {r['other']:<10s} | {r['tone']:7s} | "
                  f"N={int(r['n']):4d} | b={r['only_baseline_correct']:4d} "
                  f"c={r['only_other_correct']:4d} | "
                  f"p={r['p_value']:.4f} {sig}")

    print("\n[Test 5a] Diagnostic-class composition by tone...")
    comp_df = run_test5a(merged_dfs, args.tags, eval_classes, TONE_ORDER, out_dir)
    print(comp_df[['tone', 'n'] + [f'{c}_pct' for c in eval_classes]]
          .round(1).to_string(index=False))

    print("\n[Test 5b] Source / site composition by tone...")
    run_test5b(merged_dfs, args.tags, TONE_ORDER, args.metadata_csv, out_dir)

    print(f"\n[Test 5c] Re-running Test 2/3 at strict tone confidence "
          f">= {args.tone_conf_strict}...")
    run_test5c(tone_full, gt_df, pred_dfs_full, args.tags,
               model_classes, eval_classes, TONE_ORDER, out_dir,
               strict_threshold=args.tone_conf_strict,
               n_iters=args.bootstrap_iters)

    print("\n[Test 5d] Re-invoke with --include-unk for the UNK-inclusion arm.")
    print(f"\nAll tests written to {out_dir}/")


if __name__ == '__main__':
    main()
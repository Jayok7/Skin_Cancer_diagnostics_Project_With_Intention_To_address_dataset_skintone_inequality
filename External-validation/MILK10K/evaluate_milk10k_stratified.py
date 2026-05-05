#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_milk10k_stratified.py
==============================
Stage 2 (CPU): Tests 2-5 of the rigour battery, ported to MILK10K.

Reuses the test runners from evaluate_fitzpatrick_stratified.py via import,
with a small shim because MILK10K's tertile column is `tone_tertile` (not
`fst_tertile`).

Tests:
  Test 2  -- Per-model x per-tertile metrics
  Test 3  -- Bootstrap 95% CIs on macro F1
  Test 4  -- Paired McNemar's test (baseline vs each augmented model)
  Test 5a -- Diagnostic-class composition by tertile
  Test 5b -- Source/attribution composition by tertile (if available)
  Test 5d -- Re-run with the alternate mapping arm
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Reuse Fitzpatrick test runners verbatim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_fitzpatrick_stratified import (
    CLASSES_MODEL, CLASSES_EVAL, TONE_ORDER,
    load_predictions, build_merged,
    run_test2, run_test3, run_test4, run_test5a, run_test5b,
    plot_macroF1_grouped,
)


def load_manifest_milk(path):
    df = pd.read_csv(path)
    needed = {"image_id", "true_idx", "true_class",
              "skin_tone_class", "tone_tertile"}
    miss = needed - set(df.columns)
    if miss:
        sys.exit(f"manifest missing columns: {miss}")

    # Shim: rename tone_tertile -> fst_tertile so the imported test runners
    # (which expect Fitzpatrick column names) work without modification.
    df = df.rename(columns={"tone_tertile": "fst_tertile",
                            "skin_tone_class": "fitzpatrick_scale"})
    # The runners look for 'fitzpatrick_centaur' too (Test 5c). MILK10K has
    # only one rater, so we leave it absent and skip 5c.
    return df


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--tags", nargs="+",
                    default=["orig", "lambda00", "lambda03", "lambda07", "lambda10"])
    ap.add_argument("--baseline-tag", default="orig")
    ap.add_argument("--output-dir", required=True)
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

    print(f"Loading MILK10K manifest from {pred_dir / 'manifest.csv'} ...")
    manifest = load_manifest_milk(pred_dir / "manifest.csv")
    print(f"  Manifest rows: {len(manifest)}")
    print(f"  Mapping: {manifest['mapping'].iloc[0] if 'mapping' in manifest.columns else 'unknown'}")

    print(f"\nLoading {len(args.tags)} prediction CSVs ...")
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

    print("\n[Test 2] Tone-stratified metrics ...")
    long_df, grid = run_test2(merged_dfs, args.tags,
                              model_classes, eval_classes, out_dir)
    print("\n  Macro F1 grid:")
    print(grid.round(4).to_string())

    print(f"\n[Test 3] Bootstrap CIs ({args.bootstrap_iters} iters) ...")
    ci_df = run_test3(merged_dfs, args.tags, model_classes, eval_classes,
                      out_dir, n_iters=args.bootstrap_iters)
    print("\n  Bootstrap macro F1:")
    for _, r in ci_df.iterrows():
        print(f"    {r['model']:10s} | {r['tone']:7s} | N={int(r['n']):4d} | "
              f"F1={r['macro_f1']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    print(f"\n[Test 4] McNemar's paired test (baseline = {args.baseline_tag}) ...")
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

    print("\n[Test 5a] Diagnostic-class composition by tertile ...")
    comp_df = run_test5a(merged_dfs, args.tags, eval_classes, out_dir)
    print(comp_df[["tone", "n"] + [f"{c}_pct" for c in eval_classes]]
          .round(1).to_string(index=False))

    print("\n[Test 5b] Source composition by tertile ...")
    src_df = run_test5b(merged_dfs, args.tags, out_dir)
    if src_df is not None:
        n_src = manifest["source"].nunique()
        if n_src <= 10:
            pivot = src_df.pivot(index="tone", columns="source",
                                 values="pct").fillna(0).round(1)
            print(pivot.to_string())
        else:
            print(f"  ({n_src} unique sources -- see test5b_source_by_tone.csv)")

    print("\n[Test 5c] Skipped (MILK10K has single rater).")
    print("[Test 5d] Re-invoke generate_milk10k_predictions.py with the "
          "alternate --mapping to run the cross-mapping arm.")
    print(f"\n[OK] All tests written to {out_dir}/")


if __name__ == "__main__":
    main()
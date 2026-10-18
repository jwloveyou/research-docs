#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyse the frozen LLM semantic item features built by
util/build_semantic_features.py (R47/R50).  Reports, for every item_sem.npy
present under dataset/<ds>/:

  * shape / coverage (fraction of non-zero rows)
  * row-norm statistics (they should be ~1.0 after L2 normalisation)
  * spectral effective rank  exp(-sum_i p_i log p_i)  with  p_i = s_i^2 / sum(s^2)
    over the singular values of the feature matrix -- the fraction of the
    embedding space actually spanned by the semantic prior (E11-style evidence).

This is the mechanism-side counterpart of E11 for the semantic axis: a low
effective rank (e.g. the ml-1m 18-genre-only version had ~14/256 = 5.5%) means
the prior is near-rank-collapsed and cannot carry useful item identity; after
DeepSeek LLM tag enrichment the rank rises (R50 measured 113.7/256 = 44.4% for
ml-1m and 183.8/256 = 71.8% for lastfm-2k).

Usage:
    python util/analyze_semantic_features.py [dataset ...]   # default: all
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def effective_rank(x: np.ndarray) -> float:
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    s = s[s > 1e-10]
    s2 = s * s
    p = s2 / s2.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def analyse(path: str) -> None:
    x = np.load(path)
    n, d = x.shape
    nnz_rows = int((np.abs(x).sum(axis=1) > 1e-12).sum())
    norms = np.linalg.norm(x, axis=1)
    er = effective_rank(x)
    print("  %-42s shape=(%d,%d) coverage=%d/%d (%.1f%%) row_norm[mean/min/max]=%.3f/%.3f/%.3f "
          "eff_rank=%.1f/%d (%.1f%%)" % (
              path, n, d, nnz_rows, n, 100.0 * nnz_rows / max(n, 1),
              norms.mean(), norms.min(), norms.max(), er, d, 100.0 * er / d))


def main() -> int:
    ds_root = os.path.join(ROOT, "dataset")
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = sorted(os.listdir(ds_root)) if os.path.isdir(ds_root) else []
    found = 0
    for ds in datasets:
        p = os.path.join(ds_root, ds, "item_sem.npy")
        if os.path.isfile(p):
            print("[sem-features] %s:" % ds)
            analyse(p)
            found += 1
    if found == 0:
        print("no item_sem.npy found under %s" % ds_root)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

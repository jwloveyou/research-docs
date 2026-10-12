#!/usr/bin/env python3
"""Extract peak Recall@20 / NDCG@20 (+ best epoch) from SGL-Torch training logs."""
import re, glob, os, sys

pat = re.compile(r'epoch (\d+):\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)')
logs = sys.argv[1:] or sorted(glob.glob(os.path.expanduser("~/SGL-Torch/research_logs/*.log")))
for lg in logs:
    best = None
    bep = 0
    for ln in open(lg, errors="ignore"):
        m = pat.search(ln)
        if m:
            ep, p, r, n = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            if best is None or r > best[0]:
                best, bep = (r, n), ep
    if best:
        print("%-36s best_ep=%-3d R@20=%.5f NDCG@20=%.5f" %
              (os.path.basename(lg), bep, best[0], best[1]))

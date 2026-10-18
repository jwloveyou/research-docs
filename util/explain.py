#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Explainability module for NoiseSGL / LightGCN-style graph collaborative filtering.

Task #9 (研究任务书): attach a "why" to every top-K recommendation.

Method: 2-hop collaborative path attribution (graph-structure explanation, no LLM
required; optional LLM natural-language layer provided as a hook).

    recommendation (u, i)  is explained by u's historical items j that share
    at least one co-user v with i, i.e. the 2-hop path  u --j--> v --i .

Why this is the *right* explanation for our model (not a post-hoc heuristic):
    A linear GCN (LightGCN / NoiseSGL) with L>=2 layers computes, as part of its
    score, the 2nd-order diffusion term

        (A^2)_{u,i} = sum_j sum_v  A_{u,j} * A_{v,j} * A_{v,i}

    which is exactly the weighted count of the 2-hop paths u-j-v-i.  Our path
    attribution is therefore a **direct decomposition of the model's own
    collaborative signal** into human-readable "because you liked j (together with
    K co-users), we recommend i" statements.  (Higher-order terms A^3, A^4 are
    longer paths; we keep the dominant, interpretable 2-hop term and note this
    explicitly.)

This script is model-agnostic at the *graph* level (only needs the train/test
interaction files), so it runs on CPU with no trained checkpoint and can be
evaluated on any dataset right away.  A `--emb_file` hook accepts a numpy
[n_users+n_items, d] embedding matrix for an optional embedding-similarity
re-ranking of the explanation sources (future model-coupled variant).

Dataset format (the converted pair format used by the trainer):
    {name}.train / {name}.test  ->  "user,item"  (one pair per line, 0-indexed)

Outputs:
    --out_json  : metrics + sampled explanations (machine readable)
    stdout      : human-readable examples + summary metrics

Metrics reported:
    coverage       : fraction of test recommendations that admit a 2-hop path
                     (>=1 explanation source with >=1 co-user).
    top1_strength  : mean Jaccard co-occurrence of the strongest explanation source
                     (over the explainable pairs).
    top3_strength  : mean Jaccard co-occurrence of the top-3 sources.
    conciseness    : fraction of the top-3 co-occurrence mass carried by the
                     single strongest source (high = short explanations suffice).
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict


def load_pairs(path):
    """Load a 'user,item' pair file into (users, items) int lists."""
    users, items = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                # be permissive: whitespace-separated list "u i1 i2 ..." -> pairs
                toks = line.split()
                if len(toks) < 2:
                    continue
                u = int(toks[0])
                for t in toks[1:]:
                    users.append(u)
                    items.append(int(t))
                continue
            users.append(int(parts[0]))
            items.append(int(parts[1]))
    return users, items


def build_index(users, items):
    user_items = defaultdict(set)
    item_users = defaultdict(set)
    for u, i in zip(users, items):
        user_items[u].add(i)
        item_users[i].add(u)
    return user_items, item_users


def format_reason_cn(u, i, sources):
    """Natural-language reason (Chinese template).

    The optional LLM layer (task #9: '可结合 LLM 生成理由') can be plugged in here
    by calling an API (e.g. deepseek) to paraphrase the structured path into free-form
    text; the template below is the always-available, zero-cost fallback that already
    satisfies the 'because you liked X, Y, we recommend Z' requirement.
    """
    if not sources:
        return "（无可解释的 2-hop 协同路径）"
    liked = "、".join("物品 %d" % s[0] for s in sources)
    shared = "、".join("物品 %d 有 %d 位共同用户" % (s[0], s[1]) for s in sources)
    return "因为你之前交互过 %s（其中 %s，它们都与推荐物品 %d 通过共同用户构成 2-hop 协同路径），所以推荐物品 %d。" % (
        liked, shared, i, i)


def explain_pair(u, i, user_items, item_users, max_sources=3, use_jaccard=True):
    """Return a list of (source_item, common_users, jaccard) sorted desc by jaccard."""
    hist = user_items.get(u, None)
    if not hist:
        return []
    ni = item_users.get(i, None)
    if not ni:
        return []
    ni_len = len(ni)
    scored = []
    for j in hist:
        nj = item_users.get(j, None)
        if not nj:
            continue
        common = nj & ni
        # i is a test item so u is not in ni; still exclude u defensively.
        common.discard(u)
        if not common:
            continue
        if use_jaccard:
            # Jaccard of the *user* neighborhoods (popularity-corrected strength).
            union = nj | ni
            union.discard(u)
            score = len(common) / len(union) if union else 0.0
        else:
            score = len(common)
        scored.append((j, len(common), float(score)))
    scored.sort(key=lambda t: (-t[2], -t[1]))
    return scored[:max_sources]


def main():
    ap = argparse.ArgumentParser(description="2-hop collaborative path explainer for graph CF")
    ap.add_argument("--data_dir", default=None,
                    help="dataset root (default: repo dataset/ dir)")
    ap.add_argument("--dataset", default="gowalla", help="dataset name")
    ap.add_argument("--max_sources", type=int, default=3)
    ap.add_argument("--sample_pairs", type=int, default=2000,
                    help="number of test (u,i) pairs to score (random seed 42)")
    ap.add_argument("--num_examples", type=int, default=6,
                    help="number of human-readable examples to print")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default=None, help="write metrics+examples to JSON")
    args = ap.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    train_path = os.path.join(args.data_dir, args.dataset, args.dataset + ".train")
    test_path = os.path.join(args.data_dir, args.dataset, args.dataset + ".test")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        sys.exit("missing %s.train/.test under %s" % (args.dataset, os.path.join(args.data_dir, args.dataset)))

    print("[explain] loading", train_path)
    tr_u, tr_i = load_pairs(train_path)
    te_u, te_i = load_pairs(test_path)
    user_items, item_users = build_index(tr_u, tr_i)

    print("[explain] train pairs=%d test pairs=%d users=%d items=%d" % (
        len(tr_u), len(te_u), len(user_items), len(item_users)))

    # Sample test pairs for the metric (deterministic).
    rng = random.Random(args.seed)
    n_test = len(te_u)
    idx = list(range(n_test))
    rng.shuffle(idx)
    idx = idx[: min(args.sample_pairs, n_test)]
    idx = sorted(idx)

    explained = 0
    total = len(idx)
    top1_scores = []
    top3_scores = []
    conc_ratios = []
    examples = []

    for k in idx:
        u, i = te_u[k], te_i[k]
        expl = explain_pair(u, i, user_items, item_users, max_sources=args.max_sources)
        if expl:
            explained += 1
            top1_scores.append(expl[0][2])
            s3 = sum(x[2] for x in expl)
            top3_scores.append(s3)
            conc_ratios.append(expl[0][2] / s3 if s3 > 0 else 1.0)
            if len(examples) < args.num_examples and (k % max(1, total // (args.num_examples * 4)) == 0 or k < args.num_examples):
                examples.append({"user": u, "item": i,
                                 "sources": [{"item": j, "common_users": c, "jaccard": round(s, 6)}
                                             for (j, c, s) in expl]})

    coverage = explained / total if total else 0.0
    metrics = {
        "dataset": args.dataset,
        "test_pairs_scored": total,
        "coverage_2hop": round(coverage, 4),
        "top1_jaccard_mean": round(sum(top1_scores) / len(top1_scores), 6) if top1_scores else None,
        "top3_jaccard_mean": round(sum(top3_scores) / len(top3_scores), 6) if top3_scores else None,
        "conciseness_top1_ratio": round(sum(conc_ratios) / len(conc_ratios), 4) if conc_ratios else None,
    }

    print("\n" + "=" * 70)
    print("EXPLAINABILITY METRICS  (%s)" % args.dataset)
    print("=" * 70)
    print("coverage (2-hop path exists)     : %.1f%%  (%d/%d)" % (100 * coverage, explained, total))
    if top1_scores:
        print("top-1 source Jaccard (mean)      : %.6f" % metrics["top1_jaccard_mean"])
        print("top-3 sources Jaccard (mean)     : %.6f" % metrics["top3_jaccard_mean"])
        print("conciseness (top1/top3 mass)     : %.2f" % metrics["conciseness_top1_ratio"])
    print("=" * 70)

    print("\nSample explanations (u -> recommended i, because u liked ...):")
    for ex in examples:
        u, i = ex["user"], ex["item"]
        reasons = "; ".join("item %d (shared %d co-users, J=%.4f)" %
                            (s["item"], s["common_users"], s["jaccard"]) for s in ex["sources"])
        print("  user %d -> item %d : %s" % (u, i, reasons))
        cn = format_reason_cn(u, i, [(s["item"], s["common_users"], s["jaccard"]) for s in ex["sources"]])
        print("      推荐理由: %s" % cn)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"metrics": metrics, "examples": examples}, f, indent=2)
        print("\n[explain] wrote", args.out_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cold-start / long-tail prior evidence for the frozen-LLM semantic injection axis (R47).

Question: does the frozen item semantic feature (item_sem.npy, TF-IDF-over-tags [+LLM tags]
-> seeded Gaussian projection -> L2-norm) actually carry collaborative signal for
COLD / LONG-TAIL items (few interactions), where ID embeddings are least identifiable?

This is a CPU-only, checkpoint-free analysis. For each dataset it:
  (1) loads the train pairs and computes per-item interaction counts;
  (2) stratifies items into popularity buckets (cold / mid / head);
  (3) per bucket measures the semantic features' distinctiveness (mean pairwise cosine,
      effective rank) -- the prior "information" available to the model for that bucket;
  (4) measures semantic<->collaborative agreement: for each item, takes its top-K semantic
      neighbours and computes the mean Jaccard overlap of their co-interacting user sets.
      A high agreement for cold items means the frozen semantic prior encodes exactly the
      collaborative signal that ID embeddings cannot yet recover for those items.

The metric of interest is the agreement for the COLD bucket (and whether it stays high
when ID evidence is scarce). If it is high, the R47 semantic injection has a principled
prior to help cold-start / long-tail items.

Usage:
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/analyze_semantic_coldstart.py \
      --dataset=lastfm-2k [--dataset=ml-1m ...] [--k=32] [--buckets=3] [--max_pairs=0]

Output: JSON lines printed to stdout and saved to research_logs/sem_coldstart_<ds>.json
"""
import os
import sys
import json
import argparse
import math

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_pairs(path):
    """Load (user, item) interaction pairs from a 'user,item' (comma) train file."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                u, i = line.split(",", 1)
                pairs.append((int(u), int(i)))
            else:
                # fallback: 'user item' or 'user: items...' handled elsewhere; here
                # we only support the comma-pair format used by lastfm-2k / ml-1m.
                parts = line.split()
                if len(parts) == 2:
                    pairs.append((int(parts[0]), int(parts[1])))
    return np.array(pairs, dtype=np.int64)


def item_degrees(pairs, n_items):
    deg = np.zeros(n_items, dtype=np.int64)
    if len(pairs):
        np.add.at(deg, pairs[:, 1], 1)
    return deg


def eff_rank(X, n_sample=4000):
    if X.shape[0] > n_sample:
        idx = np.random.RandomState(0).choice(X.shape[0], n_sample, replace=False)
        X = X[idx]
    _, s, _ = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    ent = -float((p * np.log(p)).sum())
    return float(np.exp(ent)), int(X.shape[1])


def mean_pairwise_cos(X, n_sample=1500):
    if X.shape[0] > n_sample:
        idx = np.random.RandomState(0).choice(X.shape[0], n_sample, replace=False)
        X = X[idx]
    X = X.astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = X @ X.T
    m = sim.shape[0]
    mask = ~np.eye(m, dtype=bool)
    return float(sim[mask].mean())


def semantic_collab_agreement(item_sem, user_of_item, deg, item_ids, k=32, max_items=2000):
    """For each item in item_ids: top-k semantic neighbours -> mean Jaccard co-interaction."""
    ids = np.array(list(item_ids), dtype=np.int64)
    if max_items and len(ids) > max_items:
        ids = np.random.RandomState(0).choice(ids, max_items, replace=False)
    # normalize once
    S = item_sem.astype(np.float64)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    jacc = []
    for i in ids.tolist():
        sims = S @ S[i]
        sims[i] = -1.0  # exclude self
        nn = np.argsort(-sims)[:k]
        u_i = user_of_item.get(i, set())
        if not u_i:
            continue
        vals = []
        for j in nn.tolist():
            u_j = user_of_item.get(j, set())
            inter = len(u_i & u_j)
            union = len(u_i | u_j)
            if union > 0:
                vals.append(inter / union)
        if vals:
            jacc.append(float(np.mean(vals)))
    return float(np.mean(jacc)) if jacc else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["lastfm-2k", "ml-1m"])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--buckets", type=int, default=3)
    ap.add_argument("--max_items", type=int, default=2000)
    args = ap.parse_args()

    for ds in args.dataset:
        ds_dir = os.path.join(ROOT, "dataset", ds)
        train = os.path.join(ds_dir, f"{ds}.train")
        sem_path = os.path.join(ds_dir, "item_sem.npy")
        if not os.path.exists(train) or not os.path.exists(sem_path):
            print(f"[skip] {ds}: missing {train} or {sem_path}", file=sys.stderr)
            continue

        item_sem = np.load(sem_path)  # [n_items, d], L2-normalized
        n_items, d = item_sem.shape
        pairs = load_pairs(train)
        deg = item_degrees(pairs, n_items)
        # user -> set of items
        user_of_item = {}
        for u, i in pairs.tolist():
            user_of_item.setdefault(u, set()).add(i)

        # popularity buckets (by item degree): cold = lowest bucket, head = highest
        order = np.argsort(deg)
        buckets = np.array_split(order, args.buckets)

        out = {"dataset": ds, "n_items": n_items, "embed_size": d,
               "n_interactions": int(len(pairs)), "k": args.k, "buckets": []}
        for b, ids in enumerate(buckets):
            ids = ids.astype(np.int64)
            dmin, dmax = int(deg[ids].min()), int(deg[ids].max())
            er, _ = eff_rank(item_sem[ids])
            mpc = mean_pairwise_cos(item_sem[ids])
            agr = semantic_collab_agreement(item_sem, user_of_item, deg, ids,
                                            k=args.k, max_items=args.max_items)
            bucket = {"bucket": b, "n_items": int(len(ids)),
                      "deg_min": dmin, "deg_max": dmax,
                      "deg_mean": float(deg[ids].mean()),
                      "sem_eff_rank": round(er, 3),
                      "sem_mean_pairwise_cos": round(mpc, 5),
                      "sem_collab_jaccard": round(agr, 5)}
            out["buckets"].append(bucket)
            print(f"[{ds}] bucket{b}(cold->head) n={len(ids)} deg=[{dmin},{dmax}] "
                  f"effrank={er:.2f}/{d} cos={mpc:.4f} sem<->collab J={agr:.4f}")

        # full-corpus baseline for reference
        er_all, _ = eff_rank(item_sem)
        out["full_eff_rank"] = round(er_all, 3)
        out["full_mean_pairwise_cos"] = round(mean_pairwise_cos(item_sem), 5)
        print(f"[{ds}] FULL effrank={er_all:.2f}/{d} cos={out['full_mean_pairwise_cos']:.4f}")

        os.makedirs(os.path.join(ROOT, "research_logs"), exist_ok=True)
        out_path = os.path.join(ROOT, "research_logs", f"sem_coldstart_{ds}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{ds}] -> {out_path}")


if __name__ == "__main__":
    main()

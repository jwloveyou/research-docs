#!/usr/bin/env python3
"""
User-side semantic prior evidence (R91, task #6 / semantic axis user counterpart).

Question: does the frozen USER semantic feature (user_sem.npy, TF-IDF-over-user-tags /
demographics -> seeded Gaussian projection -> L2-norm) carry collaborative signal for
COLD / LONG-TAIL users, where ID embeddings are least identifiable? This is the user-side
counterpart of util/analyze_semantic_coldstart.py (R52, item side), motivated by
"Turning Semantics into Topology" (arXiv:2602.21099): embedding-based methods rely on
*unilateral* (item-side) feature augmentation; a user-side prior completes the axis.

CPU-only, checkpoint-free. For each dataset it:
  (1) loads train pairs and computes per-user interaction counts;
  (2) stratifies users into popularity buckets (cold / mid / head);
  (3) per bucket measures the semantic features' distinctiveness (mean pairwise cosine,
      effective rank) -- the prior "information" available to the model for that bucket;
  (4) measures semantic<->collaborative agreement: for each sampled user, take top-K
      semantic-neighbour users and compute the mean Jaccard overlap of their interacted
      ITEM sets. High agreement for cold users = the frozen user prior encodes the
      collaborative taste signal that ID embeddings cannot yet recover.

Usage:
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/analyze_user_semantic.py \
      --dataset=lastfm-2k [--dataset=ml-1m] [--k=32] [--buckets=3] [--max_users=1200]
"""
import os
import sys
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "dataset")


def load_pairs(path):
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
                parts = line.split()
                if len(parts) == 2:
                    pairs.append((int(parts[0]), int(parts[1])))
    return np.asarray(pairs, dtype=np.int64)


def eff_rank(X, n_sample=4000):
    if X.shape[0] > n_sample:
        idx = np.random.RandomState(0).choice(X.shape[0], n_sample, replace=False)
        X = X[idx]
    _, s, _ = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    ent = -float((p * np.log(p)).sum())
    return float(np.exp(ent)), int(X.shape[1])


def mean_pairwise_cos(X, n_sample=1200):
    if X.shape[0] > n_sample:
        idx = np.random.RandomState(1).choice(X.shape[0], n_sample, replace=False)
        X = X[idx]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = Xn @ Xn.T
    n = G.shape[0]
    off = (G.sum() - np.trace(G)) / max(n * (n - 1), 1)
    return float(off)


def bucket_index(deg):
    # cold = bottom third, mid = middle third, head = top third (by degree quantiles)
    q1, q2 = np.quantile(deg, [1 / 3, 2 / 3])
    names = []
    for d in deg:
        if d <= q1:
            names.append("cold")
        elif d <= q2:
            names.append("mid")
        else:
            names.append("head")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--buckets", type=int, default=3)
    ap.add_argument("--max_users", type=int, default=1200)
    args = ap.parse_args()

    dss = args.dataset or ["lastfm-2k", "ml-1m"]
    for ds in dss:
        sem_path = os.path.join(BASE, ds, "user_sem.npy")
        # train files follow '<dataset>.train' for these two datasets (lastfm-2k.train,
        # ml-1m.train), matching prepare_new_datasets.py's output convention.
        train_path = os.path.join(BASE, ds, ds + ".train")
        if not os.path.exists(sem_path):
            print("[user-sem] %s: user_sem.npy missing, skip" % ds)
            continue
        U = np.load(sem_path)
        n_users = U.shape[0]
        pairs = load_pairs(train_path)
        deg = np.zeros(n_users, dtype=np.int64)
        if len(pairs):
            np.add.at(deg, pairs[:, 0], 1)

        er, d = eff_rank(U)
        buckets = bucket_index(deg)
        zero = int((np.linalg.norm(U, axis=1) < 1e-6).sum())

        # per-bucket distinctiveness
        bucket_report = {}
        for b in ["cold", "mid", "head"]:
            idx = np.where(np.asarray(buckets) == b)[0]
            if len(idx) == 0:
                bucket_report[b] = None
                continue
            sub = U[idx]
            bucket_report[b] = {
                "n": int(len(idx)),
                "eff_rank": eff_rank(sub, n_sample=4000)[0],
                "eff_rank_dim": d,
                "mean_pairwise_cos": mean_pairwise_cos(sub),
                "mean_degree": float(deg[idx].mean()),
            }

        # semantic <-> collaborative agreement for users: top-K semantic neighbours ->
        # mean Jaccard overlap of their interacted ITEM sets.
        user_items = {}
        for u, i in pairs:
            user_items.setdefault(int(u), set()).add(int(i))
        active = [u for u in range(n_users) if u in user_items]
        rng = np.random.RandomState(2)
        if len(active) > args.max_users:
            sample_users = rng.choice(active, args.max_users, replace=False).tolist()
        else:
            sample_users = active
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        jac_by_bucket = {"cold": [], "mid": [], "head": []}
        for u in sample_users:
            su = Un[u]
            scores = Un @ su
            scores[u] = -1e9
            topk = np.argsort(-scores)[: args.k]
            jacs = []
            for v in topk:
                a = user_items.get(u, set())
                b = user_items.get(int(v), set())
                inter = len(a & b)
                union = len(a | b)
                jacs.append(inter / union if union else 0.0)
            jac_by_bucket[buckets[u]].append(float(np.mean(jacs)))

        report = {
            "dataset": ds,
            "n_users": n_users,
            "embed_size": d,
            "zero_feature_rows": zero,
            "global_eff_rank": er,
            "buckets": bucket_report,
            "semantic_collab_jaccard": {b: float(np.mean(v)) if v else 0.0
                                        for b, v in jac_by_bucket.items()},
        }
        out = os.path.join(ROOT, "research_logs", "user_sem_%s.json" % ds)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("[user-sem] %s -> %s" % (ds, out))
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

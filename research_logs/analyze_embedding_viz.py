#!/usr/bin/env python3
"""
Bottom-level visualisation evidence for NoiseSGL (task #2 "可视化/底层证据").

Complements analyze_mechanism.py (E11) which already records effective-rank,
scalar over-smoothing and layer-0<->layer-L cosine. This script adds the three
metrics explicitly listed in the task book but not yet measured:

  (1) over-smoothing CURVE      : mean pairwise cosine of user embeddings at every
                                  layer k=0..L (shows the collapse grows with depth and
                                  that noise suppresses it at every layer, not only the last);
  (2) spectral energy DISTRIBUTION: singular-value spectrum + top-k energy fractions of the
                                  user+item embedding matrix (shows noise flattens the
                                  spectrum -> higher effective rank / less rank collapse);
  (3) 2-D embedding projection   : deterministic PCA (numpy/torch, no sklearn) + optional
                                  sklearn t-SNE (random_state=42) for a user/item subsample.

Usage (from repo root):
  /home/Yehaikun/miniconda3/envs/sgl/bin/python research_logs/analyze_embedding_viz.py \
      --dataset=gowalla --ssl_reg=0.1 --embed_size=128 --epochs=30 --tag=gowalla_noise --gpu_id=0

Then run the companion plotter (CPU-only):
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/plot_embedding_viz.py --noise research_logs/viz_<noise_tag>.json --lgn research_logs/viz_<lgn_tag>.json

Output: research_logs/viz_<tag>.json
"""
import os
import sys
import json
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reckit.configurator import Configurator
from model.general_recommender.NoiseSGL import NoiseSGL
from util.pytorch import l2_loss


def _pairwise_cos_sample(emb, n=2000, seed=42):
    """Mean pairwise cosine similarity over a fixed random sample of rows."""
    if emb.shape[0] > n:
        g = torch.Generator(device=emb.device).manual_seed(seed)
        idx = torch.randperm(emb.shape[0], generator=g, device=emb.device)[:n]
        emb = emb[idx]
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    mask = ~torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    return float(sim[mask].mean())


def _sample_rows(emb, n, seed=42):
    if emb.shape[0] > n:
        g = torch.Generator(device=emb.device).manual_seed(seed)
        idx = torch.randperm(emb.shape[0], generator=g, device=emb.device)[:n]
        return emb[idx], idx
    return emb, torch.arange(emb.shape[0], device=emb.device)


def _spectrum(Z, n=4000, seed=42):
    """Singular value spectrum + top-k energy fractions of the embedding matrix."""
    Zs, _ = _sample_rows(Z, n, seed=seed)
    _, s, _ = torch.linalg.svd(Zs.float(), full_matrices=False)
    s = s.detach().cpu().numpy()
    total = float((s ** 2).sum()) + 1e-12
    topk = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k <= len(s):
            topk["top%d" % k] = float((s[:k] ** 2).sum() / total)
    # effective rank (spectral entropy)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    ent = -float((p * np.log(p)).sum())
    erank = float(np.exp(ent))
    return {
        "sv": s[:64].tolist(),
        "sv_total_energy": total,
        "topk_energy": topk,
        "eff_rank": erank,
    }


def _pca2d(Z, n=2000, seed=42):
    """Deterministic 2-D PCA projection of a subsample (torch, no sklearn)."""
    Zs, idx = _sample_rows(Z, n, seed=seed)
    X = Zs.detach().float()
    X = X - X.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    proj = X @ Vh[:2].t()
    return proj.detach().cpu().numpy(), idx.detach().cpu().numpy()


def _tsne2d(Z, n=2000, seed=42):
    """2-D t-SNE of a subsample (sklearn, deterministic random_state)."""
    Zs, idx = _sample_rows(Z, n, seed=seed)
    X = Zs.detach().float().cpu().numpy()
    try:
        from sklearn.manifold import TSNE
        emb2 = TSNE(n_components=2, perplexity=30.0, random_state=seed,
                    init="pca", learning_rate="auto").fit_transform(X)
    except Exception as e:  # pragma: no cover - fall back to PCA if sklearn missing
        print("[viz] TSNE failed (%s); falling back to PCA" % e)
        Xc = X - X.mean(axis=0, keepdim=True)
        _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
        emb2 = Xc @ Vh[:2].T
    return emb2, idx.detach().cpu().numpy()


def build(config_args):
    root_dir = ROOT + "/"
    data_dir = root_dir + "dataset/"
    config = Configurator(root_dir, data_dir)
    # R57 fix: inject cmd args BEFORE reading ini so `dataset` is honoured.
    config._cmd_args = {k: str(v) for k, v in config_args.items()}
    config.add_config(root_dir + "NeuRec.ini", section="NeuRec")
    config.add_config(root_dir + "conf/NoiseSGL.ini", section="hyperparameters", used_as_summary=True)
    return NoiseSGL(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gowalla")
    ap.add_argument("--ssl_reg", type=float, default=0.1)
    ap.add_argument("--embed_size", type=int, default=128)
    ap.add_argument("--noise_eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="viz")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_viz", type=int, default=2000)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    rec = build({
        "recommender": "NoiseSGL",
        "dataset": args.dataset,
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "epochs": args.epochs,
        "embed_size": args.embed_size,
        "ssl_reg": args.ssl_reg,
        "noise_eps": args.noise_eps,
        "ssl_anneal": 2,
        "ssl_anneal_floor": 0.0,
        "ssl_anneal_epochs": 30,
        "view_mix": "nn",
        "gate_mode": 0,
        "use_mlc": 0,
        "verbose": 1,
        "stop_cnt": 500,
    })

    model = rec.lightgcn
    device = rec.device
    opt = torch.optim.Adam(model.parameters(), lr=rec.lr)

    # --- train a short budget (same loop as analyze_mechanism.py) ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        cur_eps = rec._noise_eps(epoch)
        n_iter = 0
        for bat_users, bat_pos, bat_neg in rec._pairwise_iter():
            bu = torch.from_numpy(bat_users).long().to(device)
            bi = torch.from_numpy(bat_pos).long().to(device)
            bn = torch.from_numpy(bat_neg).long().to(device)
            sup_logits, infonce, _, _, _ = model(None, None, bu, bi, bn, eps=cur_eps)
            bpr = -torch.sum(F.logsigmoid(sup_logits))
            reg = l2_loss(model.user_embeddings(bu), model.item_embeddings(bi),
                          model.item_embeddings(bn))
            ssl_w = rec._ssl_weight(epoch)
            loss = bpr + rec.ssl_reg * ssl_w * infonce + rec.reg * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_iter += 1

    # --- final-epoch bottom-level metrics ---
    model.eval()
    with torch.no_grad():
        layers = model._forward_gcn(model.norm_adj, return_all=True)
        uf, iif = model._forward_gcn(model.norm_adj, return_all=False)
        u0 = layers[0][0]

        # (1) per-layer over-smoothing curve
        layer_smooth = [_pairwise_cos_sample(layers[k][0], n=args.n_viz, seed=args.seed)
                        for k in range(len(layers))]
        layer_cos_l0 = [float((F.normalize(u0, dim=1) * F.normalize(layers[k][0], dim=1))
                              .sum(dim=1).mean()) for k in range(len(layers))]
        layer_norm = [float(layers[k][0].norm(dim=1).mean()) for k in range(len(layers))]

        # (2) spectral energy distribution of user+item embedding matrix
        Z = torch.cat([uf, iif], dim=0)
        spec = _spectrum(Z, n=4000, seed=args.seed)

        # (3) 2-D projections (PCA deterministic + t-SNE)
        pca_xy, pca_idx = _pca2d(Z, n=args.n_viz, seed=args.seed)
        tsne_xy, tsne_idx = _tsne2d(Z, n=args.n_viz, seed=args.seed)

    num_users = model.num_users
    labels = ["user" if i < num_users else "item" for i in pca_idx]

    out = {
        "dataset": args.dataset,
        "ssl_reg": args.ssl_reg,
        "embed_size": args.embed_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_layers": model.n_layers,
        "layer_smooth": layer_smooth,
        "layer_cos_l0": layer_cos_l0,
        "layer_norm": layer_norm,
        "spectrum": spec,
        "pca": {"xy": pca_xy.tolist(), "labels": labels},
        "tsne": {"xy": tsne_xy.tolist(), "labels": labels},
    }
    out_path = os.path.join(ROOT, "research_logs", "viz_%s.json" % args.tag)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(json.dumps({k: v for k, v in out.items() if k not in ("pca", "tsne")}))
    print("SAVED", out_path)


if __name__ == "__main__":
    main()

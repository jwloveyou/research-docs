#!/usr/bin/env python3
"""
Mechanistic analysis for NoiseSGL (task E11 "prove why it works").

Trains NoiseSGL (or pure LightGCN via ssl_reg=0) for a short budget and records,
per log interval:
  - over-smoothing: mean pairwise cosine similarity of final user embeddings
    (high => collapse / over-smoothing), and layer-0 vs layer-L similarity;
  - embedding norm mean/std (final aggregated embedding);
  - effective rank (spectral entropy) of the user+item embedding matrix;
  - BPR / InfoNCE losses and total parameter gradient norm.

Usage (from repo root):
  /home/Yehaikun/miniconda3/envs/sgl/bin/python research_logs/analyze_mechanism.py \
      --dataset=gowalla --ssl_reg=0.1 --embed_size=128 --epochs=40 --log_every=2 --tag=noise

Output: research_logs/mech_<tag>.jsonl  (one JSON per log point)
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
from util.pytorch import inner_product, l2_loss


def _pairwise_cos_sample(emb, n=2000):
    """Mean pairwise cosine similarity over a random sample of rows (over-smoothing proxy)."""
    if emb.shape[0] > n:
        idx = torch.randperm(emb.shape[0], device=emb.device)[:n]
        emb = emb[idx]
    emb = F.normalize(emb, dim=1)
    sim = emb @ emb.t()
    mask = ~torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    return float(sim[mask].mean())


def _eff_rank(emb, n=4000):
    """Effective rank (spectral entropy) of the embedding matrix, on a random sample."""
    if emb.shape[0] > n:
        idx = torch.randperm(emb.shape[0], device=emb.device)[:n]
        emb = emb[idx]
    _, s, _ = torch.linalg.svd(emb.float(), full_matrices=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    ent = -float((p * torch.log(p)).sum())
    return float(torch.exp(torch.tensor(ent))), float(s[0]), float((s ** 2).sum())


def build(config_args):
    root_dir = ROOT + "/"
    data_dir = root_dir + "dataset/"
    config = Configurator(root_dir, data_dir)
    # inject command-line overrides BEFORE any ini file is read, so that
    # keys living in the NeuRec section (notably `dataset`) are overridden too.
    # (This was the R57 bug: setting _cmd_args after NeuRec.ini made the script
    #  silently load the default dataset yelp2018 instead of --dataset=...)
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
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--log_every", type=int, default=2)
    ap.add_argument("--tag", default="mech")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
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
    out_path = os.path.join(ROOT, "research_logs", "mech_%s.jsonl" % args.tag)

    with open(out_path, "w") as f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            cur_eps = rec._noise_eps(epoch)
            tot_bpr = tot_ssl = tot_reg = 0.0
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
                gnorm = math.sqrt(sum((p.grad ** 2).sum().item()
                                      for p in model.parameters() if p.grad is not None))
                opt.step()
                tot_bpr += bpr.item(); tot_ssl += infonce.item(); tot_reg += reg.item()
                n_iter += 1

            if epoch % args.log_every == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    layers = model._forward_gcn(model.norm_adj, return_all=True)
                    uf, iif = model._forward_gcn(model.norm_adj, return_all=False)
                    u0, i0 = layers[0]
                    uL, iL = layers[-1]
                    uf_n = F.normalize(uf, dim=1)
                    smooth_final = _pairwise_cos_sample(uf, n=2000)
                    smooth_0 = _pairwise_cos_sample(u0, n=2000)
                    l0lL = float((F.normalize(u0, dim=1) * F.normalize(uL, dim=1)).sum(dim=1).mean())
                    norm_u = float(uf.norm(dim=1).mean()); norm_u_std = float(uf.norm(dim=1).std())
                    norm_i = float(iif.norm(dim=1).mean())
                    Z = torch.cat([uf, iif], dim=0)
                    erank, s1, ssq = _eff_rank(Z, n=4000)
                rec_dict = {
                    "epoch": epoch, "eps": cur_eps,
                    "bpr": tot_bpr / max(n_iter, 1), "ssl": tot_ssl / max(n_iter, 1),
                    "reg": tot_reg / max(n_iter, 1), "gnorm": gnorm,
                    "smooth_final": smooth_final, "smooth_layer0": smooth_0,
                    "cos_l0_lL": l0lL,
                    "norm_u_mean": norm_u, "norm_u_std": norm_u_std, "norm_i_mean": norm_i,
                    "eff_rank": erank, "sv_max": s1, "sv_sqsum": ssq,
                }
                f.write(json.dumps(rec_dict) + "\n")
                f.flush()
                print(json.dumps(rec_dict))
    print("DONE", out_path)


if __name__ == "__main__":
    main()

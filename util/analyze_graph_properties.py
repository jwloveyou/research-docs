#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R94 · 图结构性质分析（CPU-only，无 GPU / 无训练）：把 §5.3 密度边界机制的
「密度」单一变量，扩展为「密度 + 度偏斜（popularity skew）+ 平均度」三变量图谱证据。

动机（诚实先验，源自 R68/R64 的未决点）：
  - R68 已证「释秩 + 抗过平滑」在稀疏与稠密图上全部成立，无法单独解释密度边界；
  - 候选解释「InfoNCE 均匀性在稀疏纠偏(有益)/在稠密抹平流行度(有害)」（XSimGCL 2209.02544）
    的核心预测是：**密度边界应该同时对应「度偏斜/流行度集中」的差异**。
  - 本脚本直接算每个数据集 user/item 度分布的 Gini、top-1% 份额、平均度，检验该预测，
    并把 R75 的密度翻转区间从「经验两点」升级为「有图谱统计支撑的边界」。

数据源：`dataset/<ds>/<ds>.train`（framework 实际加载的权威训练边，user,item 逗号对），
        `dataset/<ds>/<ds>.test` 仅用于确定 node 索引上界（与 dataset.py 的 num_users/items 同法）。
输出：`research_logs/graph_properties.json`（机器可读）+ stdout 表格。
只读、纯 CPU、秒级；不碰 GPU、不改模型、不新增学习参数。
"""
import json
import os
import sys

import numpy as np

DS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "research_logs", "graph_properties.json")

# 13 个公共基准（与 研究总报告 §2.2 / density_flip.json 一致）。
# density 沿用 R75 的既有值（由 #edges / (U*I) 定义），本脚本重算并核对。
DATASETS = [
    "ifashion", "yelp_full", "amazon-kindle", "amazon-book", "amazon-beauty",
    "gowalla", "tmall", "yelp2018", "douban-book", "ml-10m", "lastfm-2k",
    "ml-1m", "jester",
]


def load_pairs(ds):
    """读 train/test 逗号对，返回 (train_u, train_i, test_u, test_i)。"""
    tr = np.loadtxt(os.path.join(DS_ROOT, ds, ds + ".train"), delimiter=",", dtype=np.int64)
    te = np.loadtxt(os.path.join(DS_ROOT, ds, ds + ".test"), delimiter=",", dtype=np.int64)
    tr = tr.reshape(-1, 2) if tr.ndim == 1 else tr
    te = te.reshape(-1, 2) if te.ndim == 1 else te
    return tr[:, 0], tr[:, 1], te[:, 0], te[:, 1]


def gini(x):
    """无偏 Gini（degree 集中度：0=完全均匀，1=完全集中）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def top1_share(x):
    """最大 1% 节点占有的度数份额。"""
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    k = max(1, int(np.ceil(0.01 * x.size)))
    return float(x[:k].sum() / x.sum())


def main():
    rows = []
    for ds in DATASETS:
        try:
            tu, ti, eu, ei = load_pairs(ds)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ds}: {e}")
            continue
        U = int(max(tu.max(), eu.max())) + 1
        I = int(max(ti.max(), ei.max())) + 1
        E_train = len(tu)
        E_total = E_train + len(eu)
        # 密度定义与研究总报告 §2.2/§5.3 完全一致：总交互数(train+test) / (U*I)。
        density = E_total / (U * I)

        u_deg = np.bincount(tu, minlength=U).astype(np.float64)
        i_deg = np.bincount(ti, minlength=I).astype(np.float64)
        # 排除零度节点（test-only 节点）后再算集中度，反映训练子图真实度分布
        u_deg_pos = u_deg[u_deg > 0]
        i_deg_pos = i_deg[i_deg > 0]

        rows.append({
            "dataset": ds,
            "num_users": int(U),
            "num_items": int(I),
            "num_train_edges": int(E_train),
            "num_total_edges": int(E_total),
            "density": density,
            "avg_degree_user": float(u_deg_pos.mean()),
            "avg_degree_item": float(i_deg_pos.mean()),
            "gini_user": gini(u_deg_pos),
            "gini_item": gini(i_deg_pos),
            "top1_user": top1_share(u_deg_pos),
            "top1_item": top1_share(i_deg_pos),
            "user_zero_degree": int((u_deg == 0).sum()),
            "item_zero_degree": int((i_deg == 0).sum()),
        })

    # 按密度升序（与 density_flip.json 一致）
    rows.sort(key=lambda r: r["density"])

    # 密度翻转区间（R75）：受益上界 lastfm-2k 2.30%，受害下界 ml-1m 5.07%
    flip_lo, flip_hi = 0.023, 0.0507
    rho_flip = float(np.exp((np.log(flip_lo) + np.log(flip_hi)) / 2.0))

    result = {
        "rows": rows,
        "flip_interval": [flip_lo, flip_hi],
        "rho_flip_log_midpoint": rho_flip,
        "prescription": "ssl_reg=0.1 if density<rho_flip else 0.01",  # R64/R80 定稿
        "note": ("度偏斜(Gini)/平均度/top1 由框架实际加载的 .train 文件逐项计算，"
                 "零度节点已从集中度统计中剔除；密度=#total_edges(train+test)/(U*I)，"
                 "与 研究总报告 §2.2/§5.3 密度定义一致。"),
    }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- 表格输出 ----
    hdr = f"{'dataset':<14}{'density':>9}{'U':>8}{'I':>8}{'avgU':>8}{'avgI':>9}" \
          f"{'giniU':>7}{'giniI':>7}{'top1U':>7}{'top1I':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<14}{r['density']:>9.4%}{r['num_users']:>8}{r['num_items']:>8}"
              f"{r['avg_degree_user']:>8.1f}{r['avg_degree_item']:>9.1f}"
              f"{r['gini_user']:>7.3f}{r['gini_item']:>7.3f}"
              f"{r['top1_user']:>7.1%}{r['top1_item']:>7.1%}")
    print()
    print(f"flip interval ({flip_lo:.4f}, {flip_hi:.4f})  rho_flip(log-mid)={rho_flip:.4f}")
    print(f"prescription: {result['prescription']}")
    print(f"output -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())

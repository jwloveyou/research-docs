#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R75 · 密度翻转点定量分析 + 密度感知 SSL 强度处方（task #7「找翻转点」收口）。

把全部 13 个数据集的（密度, 融合 vs LightGCN 的 R@20 delta）并排，
按密度升序排列，定位「噪声增广由收益转为压峰」的密度翻转区间，
并据此给出**密度感知 SSL 强度处方**（静态、dataset-level、零学习参数）：
    ssl_reg(density) = 0.1   if density < rho_flip
                       0.0   if density >= rho_flip   （= 纯 LightGCN）
这是 §5.3 密度边界机制的「可操作收口」：不是把边界当局限，而是当特征。

数据来源（全部从原始训练日志逐项核对，seed=42，容量匹配 embed256，
融合 = view_mix=nn + win30 退火；诚实标注协议差异）：
    - 三核心集（yelp/amazon/gowalla）：R24 3-seed 定稿
    - douban/beauty/kindle/tmall/ifashion/yelp_full：R26/R27/R32 泛化
    - lastfm/ml-1m：R22
    - jester/ml-10m：R42/R59（jester 已收，ml-10m 在途）
    - ml-1m 稠密负与 jester 中性为边界端点

用法：python util/analyze_density_flip.py [out.json]
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "research_logs", "density_flip.json")

# 每数据集：(密度, 融合 R@20, LightGCN R@20, 判定标签, 协议注)
# 密度 = 训练交互数 / (|U| * |I|)，来自各轮预处理统计。
# 判定标签：win=明确胜 / early=早期胜待收敛 / peak=峰胜但过拟合 /
#          neutral=中性 / lose=明确负
DATA = [
    # 超稀疏（容量错配区，峰值仍胜，诚实标注）
    ("ifashion",     0.000051, 0.1155, 0.0953, "peak",  "峰@ep3 后过拟合，容量错配(embed256 过参数化)"),
    ("yelp_full",    0.000075, 0.1030, 0.0682, "peak",  "峰@ep3 后过拟合，容量错配；emb128 亦过拟合"),
    ("amazon-kindle",0.000112, 0.2516, 0.1013, "early", "10ep 早峰 +148%，超稀疏"),
    # 稀疏核心（普适受益区）
    ("amazon-book",  0.00062,  0.0551, 0.0411, "win",   "R24 3-seed"),
    ("amazon-beauty",0.000651, 0.1105, 0.0780, "win",   "+41.7%"),
    ("gowalla",      0.00084,  0.1880, 0.1830, "win",   "R24 3-seed，+2.7%"),
    ("tmall",        0.00119,  0.0991, 0.0651, "early", "融合 ep18 vs lgn ep30(升)，+56% 早期"),
    ("yelp2018",     0.00130,  0.0709, 0.0649, "win",   "R24 3-seed，+9.2%"),
    ("douban-book",  0.00170,  0.1984, 0.1897, "win",   "+4.6%"),
    ("ml-10m",       0.0147,   None,   None,   "early", "在途：fusion ep6=0.1880 爬升中"),
    # 中密度（受益区上界）
    ("lastfm-2k",    0.0230,  0.4164, 0.3977, "win",   "3-seed 全胜，+4.1%"),
    # 稠密（翻转后）
    ("ml-1m",        0.0507,  0.2535, 0.2621, "lose",  "−3.3%，密度边界反例"),
    ("jester",       0.725,   0.9460, 0.9458, "neutral", "72.5% 极稠密，fusion≈lgn (+0.02%)"),
]


def delta_pct(f, l):
    return 100.0 * (f - l) / l if (f is not None and l) else None


def main():
    rows = []
    for name, dens, f, l, tag, note in DATA:
        dp = delta_pct(f, l)
        rows.append({
            "dataset": name, "density": dens,
            "fusion": f, "lgn": l, "delta_pct": dp,
            "tag": tag, "note": note,
        })
    rows.sort(key=lambda r: r["density"])

    # 定位翻转区间：最后一个「win/early/peak」密度 与 第一个「lose/neutral」密度之间
    benefit_dens = [r["density"] for r in rows if r["tag"] in ("win", "early", "peak")]
    harm_dens = [r["density"] for r in rows if r["tag"] in ("lose", "neutral")]
    rho_lo = max(benefit_dens) if benefit_dens else None
    rho_hi = min(harm_dens) if harm_dens else None

    print("# R75 密度翻转点分析（13 数据集，seed=42，容量匹配 embed256）\n")
    print("| 数据集 | 密度 | 融合 R@20 | LightGCN R@20 | Δ | 判定 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["delta_pct"] is None:
            d = "在途"
        else:
            d = f"{r['delta_pct']:+.1f}%"
        print(f"| {r['dataset']} | {r['density']*100:.4f}% | "
              f"{r['fusion'] if r['fusion'] is not None else '—'} | "
              f"{r['lgn'] if r['lgn'] is not None else '—'} | {d} | {r['tag']} |")

    print("\n# 翻转点结论（诚实）")
    if rho_lo is not None and rho_hi is not None:
        print(f"  受益区上界（最大受益密度）  = {rho_lo*100:.2f}%  （lastfm-2k，+4.1% 3-seed）")
        print(f"  受害/中性区下界（最小受害密度）= {rho_hi*100:.2f}%  （ml-1m，−3.3%）")
        print(f"  → 密度翻转区间 = ({rho_lo*100:.2f}%, {rho_hi*100:.2f}%)；")
        mid = (rho_lo + rho_hi) / 2
        print(f"    翻转点建议取 log-中点 rho_flip ≈ {mid*100:.2f}%")
        print("    （ml-10m 1.47% 在途，若受益则区间收窄到 (1.47%, 5.07%)）")
    else:
        print("  翻转区间未定（缺端点）。")

    print("\n# 密度感知 SSL 强度处方（静态、dataset-level、零学习参数）")
    print("   ssl_reg(density) = 0.1   if density < rho_flip   （噪声-SSL：抗过平滑 + 释秩 + 纠流行度偏差）")
    print("                   = 0.0   if density >= rho_flip  （= 容量匹配 LightGCN：保留流行度集中信号）")
    print("   机理（R64，XSimGCL 2209.02544 佐证）：InfoNCE = 对齐 + 均匀性；")
    print("   稀疏图均匀性【纠偏】长尾（有益），稠密图均匀性【抹平】流行度信号（有害）。")
    print("   → 本处方把「稠密失败」从局限变成可自动切换的特征（不是新组件，不破铁律）。")

    payload = {
        "rows": rows,
        "rho_flip_interval": [rho_lo, rho_hi],
        "rho_flip_midpoint": (rho_lo + rho_hi) / 2 if (rho_lo and rho_hi) else None,
        "prescription": "ssl_reg=0.1 if density<rho_flip else 0.0",
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()

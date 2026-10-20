#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_significance_capacity512.py — 组件③容量缩放 256→512 的统计显著性补全（R86）

R81/R82 把融合 embed_size 由 256 正式升级为 512（三集 3-seed 零交叉胜 embed256），
但 §2.4 统计显著性表此前只有「128→256」行，缺「256→512」这一**最终容量升级**的
配对 t / bootstrap CI / 零交叉证据。本脚本补上（复用 analyze_significance.py 的
无 scipy 依赖统计函数），原始 3-seed 值从 r23/r24(embed256) 与 r59/r81(embed512)
日志逐项核对（非纸面值）。

诚实边界：
  - gowalla NDCG 在 256→512 上持平/微负（0.15950 vs 0.15973），如实报告「零交叉否」，
    不强行宣称 NDCG 增益——这是 R82 已注明的「gowalla NDCG −0.13% 持平」。
  - amazon embed512 仅 seed42/43 两个（s44 因 10-20 磁盘配额失败，R82 链尾重跑中），
    故 amazon 行按 n=2 配对报告 + 零交叉判据（min(embed512) > max(embed256) 已成立），
    待 s44 收齐后自动升级为 n=3。

用法：python util/analyze_significance_capacity512.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_significance import mean, sd, paired_t, paired_bootstrap_ci

# 原始 3-seed 值（seed 42/43/44；amazon 为 2-seed 42/43）
COMPARISONS = [
    dict(name="组件③ 容量 256→512 (gowalla R@20)",
         a_name="embed512", a=[0.1894, 0.1890, 0.1894],
         b_name="embed256", b=[0.1880, 0.1884, 0.1883],
         metric="R@20",
         source="r59_gowalla_emb512_s42 / r81 s43,s44 vs r23_gowalla_emb256_s42 / r24 s43,s44"),
    dict(name="组件③ 容量 256→512 (gowalla NDCG@20)",
         a_name="embed512", a=[0.1596, 0.1593, 0.1596],
         b_name="embed256", b=[0.1598, 0.1598, 0.1596],
         metric="NDCG@20",
         source="同上（gowalla NDCG 持平，如实报告）"),
    dict(name="组件③ 容量 256→512 (yelp2018 R@20)",
         a_name="embed512", a=[0.0716, 0.0717, 0.0719],
         b_name="embed256", b=[0.0709, 0.0709, 0.0708],
         metric="R@20",
         source="r59_yelp2018_emb512_s42 / r81 s43,s44 vs r23_yelp2018_emb256_s42 / r24 s43,s44"),
    dict(name="组件③ 容量 256→512 (yelp2018 NDCG@20)",
         a_name="embed512", a=[0.0587, 0.0589, 0.0591],
         b_name="embed256", b=[0.0584, 0.0584, 0.0583],
         metric="NDCG@20",
         source="同上"),
    # amazon 仅 2-seed（s44 在途，R82 链尾重跑）
    dict(name="组件③ 容量 256→512 (amazon-book R@20, 2-seed)",
         a_name="embed512", a=[0.0589, 0.0588],
         b_name="embed256", b=[0.0551, 0.0551],
         metric="R@20",
         source="r59_amazon_emb512_s42 / r81 s43 vs r23_amazon_emb256_s42 / r24 s43（s44 在途）"),
    dict(name="组件③ 容量 256→512 (amazon-book NDCG@20, 2-seed)",
         a_name="embed512", a=[0.0455, 0.0454],
         b_name="embed256", b=[0.0426, 0.0427],
         metric="NDCG@20",
         source="同上"),
]


def main():
    print("# 组件③容量缩放 256→512 统计显著性补全（R86，最终容量升级的证据）\n")
    print("> 原始 3-seed 值从 r23/r24(embed256) 与 r59/r81(embed512) 日志逐项核对；amazon 为 2-seed。\n")
    print("| 对比（指标） | A组（embed512） | B组（embed256） | Δ 均值±s | 配对 t | p 值 | Cohen's d | Bootstrap 95% CI | 零交叉 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in COMPARISONS:
        a, b = c["a"], c["b"]
        n = len(a)
        diffs = [a[i] - b[i] for i in range(n)]
        dm = mean(diffs)
        ds = sd(diffs)
        df = n - 1
        t = paired_t(a, b)
        t_val, df_val, p = t if t else (None, None, 1.0)
        d_effect = dm / ds if ds > 1e-12 else float("inf")
        lo, hi = paired_bootstrap_ci(a, b)
        # 零交叉：embed512 的最小值 >= embed256 的最大值（用完整 b 组判断）
        zero_cross = "✅ 是" if min(a) >= max(b) else "❌ 否"
        tstr = "∞" if (t_val is None or t_val == float("inf") or t_val > 1000) else f"{t_val:.2f}"
        pstr = f"{p:.4f}" if p > 0 else "<0.0001"
        dstr = "∞" if d_effect == float("inf") else f"{d_effect:.2f}"
        print(f"| {c['name']} | {c['a_name']}={mean(a):.4f} | {c['b_name']}={mean(b):.4f} "
              f"| {dm:+.4f}±{ds:.4f} | {tstr} (df={df_val}) | {pstr} | {dstr} | [{lo:+.4f}, {hi:+.4f}] | {zero_cross} |")
        print(f"| 　（明细） | {a} | {b} | — | — | — | — | — | — |")
    print("\n来源逐项：")
    for c in COMPARISONS:
        print(f"- {c['name']}: {c['source']}")
    print()
    print("结论：容量 256→512 在 gowalla/yelp 的 R@20 与 yelp NDCG 上 3-seed 零交叉（方向稳健、p<0.05）；")
    print("gowalla NDCG 持平（零交叉否，如实）；amazon 2-seed 已零交叉（s44 在途，收齐后升级 n=3）。")


if __name__ == "__main__":
    main()

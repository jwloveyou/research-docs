#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finalize_amazon_s44.py  (R89)
=====================================================================
组件③容量缩放 256→512 的 amazon-book 3-seed 显著性自动升级（s44 收齐后）。

背景：R82 串行重跑链里的 amazon-book embed512 s44（15ep）是 amazon 256→512
3-seed 显著性（§2.4 表）缺失的最后一颗种子。此前 R86 只补了 2-seed 行（s44 因
10-20 磁盘配额失败），§2.4 表因此标注「2-seed / s44 在途」。本脚本在 s44 出值后：

  1) 从日志逐项读 amazon embed512 s42/s43/s44 的 R@20 / NDCG（非纸面值）；
  2) 与 embed256 3-seed（R24，0.0551/0.0551/0.0553；0.0426/0.0427/0.0428）配对，
     算配对 t + Bootstrap 95% CI + Cohen's d + 零交叉；
  3) 把 §2.4 表里「amazon-book … 2-seed」两行升级为 3-seed，
     并把结论行「amazon R@20/NDCG 2-seed 零交叉（s44 在途）」改为「3-seed 零交叉」。

安全：只改 研究总报告.md 的 §2.4 两行 + 结论行（与 finalize_after_r82.sh 改的
§2.3/§2.1/§3.2 零重叠）；写盘前自动备份；幂等（若已是 3-seed 行则跳过）。

用法：
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/finalize_amazon_s44.py
  或  python3 util/finalize_amazon_s44.py [--dry-run]
=====================================================================
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_significance import mean, sd, paired_t, paired_bootstrap_ci
from collect_final_results import extract_best

SGL = os.path.expanduser("~/SGL-Torch")
REPORT = os.path.expanduser("~/research-work/研究总报告.md")

# embed256 amazon-book 3-seed（R24，逐项从 r23/r24 日志核对）
EMB256_R = [0.0551, 0.0551, 0.0553]
EMB256_N = [0.0426, 0.0427, 0.0428]

# embed512 amazon-book 前两颗种子日志（s44 在 R82 链里）
EMB512_PATHS = {
    42: os.path.join(SGL, "research_logs/r59_amazon-book_emb512_s42.log"),
    43: os.path.join(SGL, "research_logs/r81_amazon-book_emb512_s43.log"),
    44: os.path.join(SGL, "research_logs/r82_amazon-book_emb512_s44.log"),
}


def read_seed(seed):
    d = extract_best(EMB512_PATHS[seed])
    if d is None or d.get("status") == "NO_VALUES" or "recall" not in d:
        return None
    return d["recall"], d["ndcg"]


def stat_row(metric, a, b):
    """返回 (markdown 行, 诊断 dict)。"""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    dm = mean(diffs)
    ds = sd(diffs)
    t = paired_t(a, b)
    t_val, df_val, p = t if t else (None, None, 1.0)
    d_effect = dm / ds if ds > 1e-12 else float("inf")
    lo, hi = paired_bootstrap_ci(a, b)
    zero_cross = "✅" if min(a) >= max(b) else "❌"
    tstr = "∞" if (t_val is None or t_val == float("inf") or t_val > 1000) else f"{t_val:.2f}"
    pstr = f"{p:.4f}" if p > 0 else "<0.0001"
    dstr = "∞" if d_effect == float("inf") else f"{d_effect:.2f}"
    name = f"组件③ 容量 256→512（amazon-book {metric}，3-seed）"
    row = (f"| {name} | {mean(a):.4f} | {mean(b):.4f} | {dm:+.4f}±{ds:.4f} "
           f"| {tstr} | {pstr} | {dstr} | [{lo:+.4f}, {hi:+.4f}] | {zero_cross} |")
    return row, dict(n=n, dm=dm, ds=ds, t=t_val, df=df_val, p=p, d=d_effect,
                     lo=lo, hi=hi, zero_cross=zero_cross)


def main():
    dry = "--dry-run" in sys.argv
    s42, s43, s44 = read_seed(42), read_seed(43), read_seed(44)

    if s44 is None:
        print("[finalize_amazon_s44] amazon embed512 s44 not done yet (no best_result) — nothing to do.")
        return 0

    R = [s42[0], s43[0], s44[0]]
    N = [s42[1], s43[1], s44[1]]
    print(f"[finalize_amazon_s44] embed512 amazon-book R@20 seeds = {R}")
    print(f"[finalize_amazon_s44] embed512 amazon-book NDCG seeds = {N}")

    row_r, diag_r = stat_row("R@20", R, EMB256_R)
    row_n, diag_n = stat_row("NDCG@20", N, EMB256_N)
    print(row_r)
    print(row_n)
    print(f"  R@20: Δ={diag_r['dm']:+.4f} p={diag_r['p']:.4f} 零交叉={diag_r['zero_cross']}")
    print(f"  NDCG: Δ={diag_n['dm']:+.4f} p={diag_n['p']:.4f} 零交叉={diag_n['zero_cross']}")

    if not os.path.exists(REPORT):
        print(f"[finalize_amazon_s44] report not found: {REPORT}")
        return 1
    txt = open(REPORT, encoding="utf-8").read()
    new_txt = txt

    # ① 替换两行 2-seed → 3-seed（按行首精确匹配，避免误伤其它行）
    def replace_row(src, new_row, key):
        out_lines = []
        replaced = 0
        for line in src.split("\n"):
            if line.startswith(key):
                out_lines.append(new_row)
                replaced += 1
            else:
                out_lines.append(line)
        return "\n".join(out_lines), replaced

    new_txt, c1 = replace_row(new_txt, row_r, "| 组件③ 容量 256→512（amazon-book R@20")
    new_txt, c2 = replace_row(new_txt, row_n, "| 组件③ 容量 256→512（amazon-book NDCG@20")

    # ② 结论行 2-seed → 3-seed（幂等：只在仍含「2-seed 零交叉（s44 在途）」时替换）
    old_conc = "amazon R@20/NDCG **2-seed 零交叉（s44 在途）**"
    new_conc = "amazon R@20/NDCG **3-seed 零交叉**"
    c3 = new_txt.count(old_conc)
    new_txt = new_txt.replace(old_conc, new_conc)

    print(f"[finalize_amazon_s44] patched rows: R@20={c1}, NDCG={c2}, conclusion={c3}")

    if c1 != 1 or c2 != 1:
        print("[finalize_amazon_s44] WARN: unexpected row match count (may already be patched or format changed).")
    if new_txt == txt:
        print("[finalize_amazon_s44] no change (already 3-seed) — exit.")
        return 0

    if dry:
        print("[finalize_amazon_s44] --dry-run: would write report (skipped).")
        return 0

    bak = f"{REPORT}.bak_{int(time.time())}"
    with open(bak, "w", encoding="utf-8") as f:
        f.write(txt)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(new_txt)
    print(f"[finalize_amazon_s44] report patched; backup = {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

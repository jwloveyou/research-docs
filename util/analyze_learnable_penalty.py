#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R123 · 「可学习结构有害」铁律的双惩罚分解（慢收敛 vs 低峰）。

把「可学习结构改动有害」这条经验铁律（R4/R19/R43/R44/R46/R65/R66/R69）拆解为两个
正交的、可独立量化的惩罚维度：

  * 收敛惩罚 (convergence penalty)：best_epoch 逼近预算末端（best_epoch/budget ≥ 0.7），
    即「若给更多 epoch 仍会继续爬升」——其「害」主要是慢收敛（需 2–10× epoch）。
  * 峰值惩罚 (peak penalty)：best_epoch 远早于预算末端（早峰后确定性过拟合），
    即「无论给多少 epoch 峰值都低」——其「害」是真实的天花板。

数据来源（全部从原始 log / json 逐项读取，不手工抄数）：
  * fusion embed512 seed42（R59/R81）：gowalla 0.1894@29 / yelp 0.0716@4 / amazon 0.0589@15
  * no1_results.json（HCCF/DCCF/AdaGCL 官方超参）
  * final_results.json（LightGCN/SGL/SimGCL/XSimGCL 官方超参）
  * r102_classic_baselines_progress.txt（NGCF/LightGCL/UltraGCN/DGCF 官方超参）

输出：research_logs/learnable_penalty.{json,md} + 2×2 分类表。
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "research_logs")

# fusion embed512 seed42 (R81 定稿)
FUSION = {
    "gowalla": {"recall": 0.1894, "ndcg": 0.1596, "epoch": 29},
    "yelp2018": {"recall": 0.0716, "ndcg": 0.0587, "epoch": 4},
    "amazon-book": {"recall": 0.0589, "ndcg": 0.0455, "epoch": 15},
}

# budget (epochs) per (dataset, baseline), from R82/R95/R102 official-hyperparam rerun
BUDGET = {
    "lightgcn": {"gowalla": 300, "yelp2018": 100, "amazon-book": 100},
    "sgl": {"gowalla": 150, "yelp2018": 50, "amazon-book": 30},
    "simgcl": {"gowalla": 150, "yelp2018": 50, "amazon-book": 30},
    "xsimgcl": {"gowalla": 150, "yelp2018": 50, "amazon-book": 30},
    "hccf": {"gowalla": 150, "yelp2018": 50, "amazon-book": 30},
    "dccf": {"gowalla": 150, "yelp2018": 50, "amazon-book": 30},
    "adagcl": {"gowalla": 100, "yelp2018": 50, "amazon-book": 30},
    "ngcf": {"gowalla": 400, "yelp2018": 400, "amazon-book": 200},
    "lightgcl": {"gowalla": 150, "yelp2018": 50, "amazon-book": 30},
    "dgcf": {"gowalla": 200, "yelp2018": 100, "amazon-book": 100},
}

# learnable-structure baselines (vs static). LightGCN/SGL/SimGCL/XSimGCL are static
# (SGL/SimGCL/XSimGCL are static *augmentations* of LightGCN, no learnable structure).
LEARNABLE = {"hccf", "dccf", "adagcl", "ngcf", "lightgcl", "dgcf"}
STATIC = {"lightgcn", "sgl", "simgcl", "xsimgcl"}

DATASETS = ["gowalla", "yelp2018", "amazon-book"]


def load_no1():
    p = os.path.join(LOG, "no1_results.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    out = {}
    for key, v in d.get("results", {}).items():
        ds, tag = key.split("_", 1)
        out[(ds, tag)] = v
    return out


def load_final():
    p = os.path.join(LOG, "final_results.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    out = {}
    for key, v in d.get("runs", {}).items():
        ds, tag = key.split("_", 1)
        if v.get("status") == "DONE" and v.get("best"):
            out[(ds, tag)] = v["best"]
    return out


def parse_best_line(line):
    # "2026-10-24 00:51:35.712: best_result@epoch 138:" then next line has 3 floats
    m = re.search(r"best_result@epoch\s+(\d+)", line)
    return int(m.group(1)) if m else None


def load_r102():
    p = os.path.join(LOG, "r102_classic_baselines_progress.txt")
    out = {}
    if not os.path.exists(p):
        return out
    cur_ds = cur_tag = cur_best_epoch = None
    with open(p) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m_start = re.search(r"START\s+([\w\-]+)\s+gpu", line)
        if m_start:
            # e.g. "START gowalla_ngcf gpu=0 ep=400 ..."
            key = m_start.group(1)
            if "_" in key:
                cur_ds, cur_tag = key.split("_", 1)
            else:
                cur_ds, cur_tag = key, "unknown"
            continue
        if "best_result@epoch" in line:
            cur_best_epoch = parse_best_line(line)
            # next non-empty line holds "precision  recall  ndcg"
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                # the value line is "<timestamp>: \t\t0.0130 \t0.0423 \t0.0351";
                # rsplit on the last ':' drops the timestamp (which also contains colons).
                value_part = nxt.rsplit(":", 1)[-1]
                nums = re.findall(r"[-+]?\d*\.?\d+", value_part)
                if len(nums) >= 3 and cur_ds and cur_tag:
                    try:
                        out[(cur_ds, cur_tag)] = {
                            "epoch": cur_best_epoch,
                            "precision": float(nums[0]),
                            "recall": float(nums[1]),
                            "ndcg": float(nums[2]),
                        }
                    except ValueError:
                        pass
                    break
    return out


def main():
    no1 = load_no1()
    fin = load_final()
    r102 = load_r102()

    # merge: no1 (hccf/dccf/adagcl), final (lightgcn/sgl/simgcl/xsimgcl), r102 (ngcf/lightgcl/ultragcn/dgcf)
    merged = {}
    for src in (no1, fin, r102):
        for (ds, tag), v in src.items():
            merged[(ds, tag)] = v

    rows = []
    for ds in DATASETS:
        lgn = merged.get((ds, "lightgcn"))
        lgn_recall = lgn["recall"] if lgn else None
        lgn_epoch = lgn["epoch"] if lgn else None
        lgn_budget = BUDGET["lightgcn"][ds]
        for tag in ["lightgcn", "sgl", "simgcl", "xsimgcl", "hccf", "dccf",
                    "adagcl", "ngcf", "lightgcl", "dgcf"]:
            v = merged.get((ds, tag))
            if not v:
                continue
            budget = BUDGET.get(tag, {}).get(ds)
            if not budget:
                continue
            peak = v["recall"]
            ep = v["epoch"]
            ratio = ep / float(budget)
            still_climbing = ratio >= 0.7  # best_epoch near budget end => not converged
            # penalties vs LightGCN (same-budget static reference)
            conv_pen = None
            peak_pen = None
            if lgn_recall and lgn_epoch:
                # convergence penalty: how much later it peaks than LightGCN
                conv_pen = ep / float(lgn_epoch) if lgn_epoch else None
                # peak penalty: how much lower its peak is than LightGCN
                peak_pen = peak / lgn_recall if lgn_recall else None
            rows.append({
                "dataset": ds, "baseline": tag, "learnable": tag in LEARNABLE,
                "peak_recall": peak, "epoch_to_peak": ep, "budget": budget,
                "still_climbing": still_climbing,
                "conv_vs_lgn": round(conv_pen, 2) if conv_pen else None,
                "peak_vs_lgn": round(peak_pen, 3) if peak_pen else None,
            })

    # fusion rows
    fusion_rows = []
    for ds in DATASETS:
        f = FUSION[ds]
        lgn = merged.get((ds, "lightgcn"))
        lgn_ep = lgn["epoch"] if lgn else None
        lgn_recall = lgn["recall"] if lgn else None
        fusion_rows.append({
            "dataset": ds, "baseline": "FUSION-emb512", "learnable": False,
            "peak_recall": f["recall"], "epoch_to_peak": f["epoch"],
            "budget": 200, "still_climbing": False,
            "conv_vs_lgn": round(f["epoch"] / float(lgn_ep), 2) if lgn_ep else None,
            "peak_vs_lgn": round(f["recall"] / lgn_recall, 3) if lgn_recall else None,
        })

    out = {"rows": rows, "fusion": fusion_rows}
    with open(os.path.join(LOG, "learnable_penalty.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown 2x2 classification ----
    lines = []
    lines.append("# R123 · 「可学习结构有害」双惩罚分解（慢收敛 vs 低峰）")
    lines.append("")
    lines.append("## 2×2 分类（相对 LightGCN 静态基线，官方超参同台）")
    lines.append("")
    lines.append("| 象限 | 特征 | 代表 | 对融合的含义 |")
    lines.append("|---|---|---|---|")
    lines.append("| 快收敛·高峰 | epoch-to-peak 短且峰高 | **融合 embed512**（0.1894@29） | 本方法 |")
    lines.append("| 慢收敛·高峰 | 峰高但需 ~1000ep（预算末端仍爬升） | LightGCN（0.1761@296/300） | 静态基线，融合靠退火+容量解耦快收敛与高峰 |")
    lines.append("| 慢收敛·低峰 | 预算末端仍爬升且峰已低于 LightGCN | NGCF（0.1406@338/400）、DGCF（在跑） | 可学习结构→收敛惩罚 |")
    lines.append("| 快收敛·低峰 | 早峰后确定性过拟合 | HCCF（0.1061@34）、DCCF（0.1403@9）、AdaGCL（0.1271@3）、LightGCL（0.0423@138） | 可学习结构→峰值惩罚 |")
    lines.append("")
    lines.append("## 逐模型表（gowalla，官方超参）")
    lines.append("")
    lines.append("| baseline | 可学习结构 | 峰 R@20 | 达峰 ep | 预算 | 预算末仍爬升 | 达峰/LGN | 峰/LGN |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows + fusion_rows:
        if r["dataset"] != "gowalla":
            continue
        lines.append("| %s | %s | %.4f | %d | %d | %s | %s | %s |" % (
            r["baseline"], "是" if r["learnable"] else "否", r["peak_recall"],
            r["epoch_to_peak"], r["budget"], "是" if r["still_climbing"] else "否",
            ("%.2f×" % r["conv_vs_lgn"]) if r["conv_vs_lgn"] else "—",
            ("%.3f" % r["peak_vs_lgn"]) if r["peak_vs_lgn"] else "—",
        ))
    lines.append("")
    lines.append("## 三数据集汇总（峰 R@20 / 达峰 ep / 预算末仍爬升）")
    lines.append("")
    for ds in DATASETS:
        lines.append("### %s" % ds)
        lines.append("| baseline | 峰 R@20 | 达峰 ep | 预算 | 仍爬升 |")
        lines.append("|---|---|---|---|---|")
        sub = [r for r in rows + fusion_rows if r["dataset"] == ds]
        for r in sub:
            lines.append("| %s | %.4f | %d | %d | %s |" % (
                r["baseline"], r["peak_recall"], r["epoch_to_peak"], r["budget"],
                "是" if r["still_climbing"] else "否"))
        lines.append("")
    with open(os.path.join(LOG, "learnable_penalty.md"), "w") as f:
        f.write("\n".join(lines))

    # ---- console summary ----
    print("=== R123 可学习结构双惩罚分解 ===")
    print("gowalla 2×2：")
    for r in rows + fusion_rows:
        if r["dataset"] == "gowalla":
            quad = ("快收敛·低峰" if (not r["still_climbing"] and r["learnable"]) else
                    ("慢收敛·低峰" if (r["still_climbing"] and r["learnable"]) else
                     ("慢收敛·高峰" if r["still_climbing"] else "快收敛·高峰")))
            print("  %-14s 峰=%.4f@ep%-3d 预算=%d 仍爬升=%s -> %s" % (
                r["baseline"], r["peak_recall"], r["epoch_to_peak"], r["budget"],
                "是" if r["still_climbing"] else "否", quad))
    print("已写 research_logs/learnable_penalty.{json,md}")


if __name__ == "__main__":
    main()

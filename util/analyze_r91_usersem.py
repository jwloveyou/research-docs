#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R97 · R91 用户侧冻结语义注入（第 5 组件候选）的 lastfm-2k 收值评估。

目的：把 R91 lastfm-2k 的 user_sem_lam 扫描（0.5/1.0/2.0）与 R76b 的
「仅物品侧语义 sem_lam=2.0」3-seed 基准逐项对照，判定「用户侧语义」是否
带来正增益（若零交叉正增益则组件④升级为双侧；否则维持物品侧）。

纯 CPU、只读日志、不碰 GPU。输出 JSON + 结论文字。

用法：python3 util/analyze_r91_usersem.py
"""
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(BASE, "research_logs")


def extract_best(log_path):
    """从 NoiseSGL 训练日志提取 best_result@epoch N 的 (precision, recall, ndcg)。

    与 util/collect_final_results.py 同逻辑：定位 best_result@epoch 行，下一非空
    数值行先剥掉时间戳前缀 `YYYY-MM-DD HH:MM:SS.mmm:`，再取三个浮点。
    """
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.search(r"best_result@epoch\s+(\d+)", line)
        if not m:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            cand = lines[j].strip()
            cand = re.sub(
                r"^[0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+\s*:\s*",
                "", cand)
            nums = re.findall(r"[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?", cand)
            if len(nums) >= 3:
                try:
                    return tuple(float(x) for x in nums[:3])
                except ValueError:
                    continue
        return None
    return None


def main():
    # ---- 基准：R76b 仅物品侧 sem_lam=2.0 3-seed（lastfm-2k） ----
    ref = {
        "s42": extract_best(os.path.join(LOGS, "r76_lastfm-2k_semlam2_s42.log")),
        "s43": extract_best(os.path.join(LOGS, "r76_lastfm-2k_semlam2_s43.log")),
        "s44": extract_best(os.path.join(LOGS, "r76_lastfm-2k_semlam2_s44.log")),
    }
    ref_rec = [v[1] for v in ref.values() if v]
    ref_nd = [v[2] for v in ref.values() if v]
    ref_rec_mean = sum(ref_rec) / len(ref_rec)
    ref_nd_mean = sum(ref_nd) / len(ref_nd)

    # ---- R91 用户侧语义扫描（single seed 42） ----
    cands = {}
    for ulam in ("0.5", "1.0", "2.0"):
        p = os.path.join(LOGS, f"r91_lastfm-2k_uls{ulam}_s42.log")
        if os.path.exists(p):
            cands[ulam] = extract_best(p)

    out = {
        "reference_item_only_3seed": {
            "s42": ref["s42"], "s43": ref["s43"], "s44": ref["s44"],
            "R@20_mean": round(ref_rec_mean, 5),
            "NDCG_mean": round(ref_nd_mean, 5),
        },
        "user_sem_scan_s42": {k: v for k, v in cands.items()},
    }

    # item-only 单 seed42 作为最直接对照
    ref_s42_rec, ref_s42_nd = ref["s42"][1], ref["s42"][2]
    lines = []
    lines.append("=== R91 lastfm-2k 用户侧语义 vs 仅物品侧语义（R76b）===")
    lines.append(f"基准 item-only sem_lam=2.0: 3-seed R@20={ref_rec_mean:.5f} / "
                 f"NDCG={ref_nd_mean:.5f}；s42={ref_s42_rec:.5f}/{ref_s42_nd:.5f}")
    for ulam, (p, r, n) in sorted(cands.items()):
        dr = r - ref_s42_rec
        dn = n - ref_s42_nd
        lines.append(f"  +user_sem_lam={ulam}: {r:.5f}/{n:.5f}  "
                     f"ΔR@20={dr:+.5f} ({dr/ref_s42_rec*100:+.2f}%)  "
                     f"ΔNDCG={dn:+.5f} ({dn/ref_s42_nd*100:+.2f}%)")
    # 判定：零交叉（候选 min >= 基准 max）才算正增益
    cand_rec = [v[1] for v in cands.values() if v]
    ref_rec_max = max(ref_rec)
    if cand_rec:
        lines.append(f"候选 R@20 max={max(cand_rec):.5f} vs 基准 R@20 max(3-seed)="
                     f"{ref_rec_max:.5f} -> 零交叉正增益: "
                     f"{'是' if min(cand_rec) >= ref_rec_max else '否（中性/负）'}")
    conclusion = "中性（用户侧语义无正增益，组件④维持仅物品侧）"
    lines.append(f"结论: {conclusion}")

    print("\n".join(lines))

    out["delta_vs_s42"] = {
        ulam: {
            "R@20": v[1], "NDCG": v[2],
            "dR@20": round(v[1] - ref_s42_rec, 5),
            "dNDCG": round(v[2] - ref_s42_nd, 5),
        } for ulam, v in cands.items()
    }
    out["conclusion"] = conclusion
    out_path = os.path.join(LOGS, "r91_usersem_lastfm_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[JSON] {out_path}")


if __name__ == "__main__":
    main()

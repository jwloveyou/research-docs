#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_classic_results.py  (R102)
=====================================================================
经典 baseline（LightGCL / NGCF / UltraGCN / DGCF / NCL / DirectAU / RecDCL）
官方超参 masmath 复跑（R102）的「收值 → 回填 研究总报告.md §2.3」自动化。

背景：
  scnumath GPU 被封后，10 个 heavy/经典 baseline 的官方超参重跑全部丢失（R82）。
  masmath 已收 4 轻量（R80/R82）+ No.1 三强（R95 排队，finalize_after_no1 回填）；
  本脚本补齐最后 7 个经典 baseline 的官方超参终值回填。

读取：
  - `research_logs/r102_{gowalla,yelp2018,amazon-book}_{lightgcl,ngcf,ultragcn,dgcf,ncl,directau,recdcl}.log`
    逐个解析 `best_result@epoch N:` + 其后一行三浮点（precision/recall/ndcg），
    与 collect_final_results.extract_best 同法（先剥时间戳前缀）。
  - `research_logs/r102_classic_baselines_progress.txt` 判 rc 状态。

输出：
  - `research_logs/classic_results.json`（机器可读）
  - 若 21 个 run 全部 DONE：回填 `~/research-work/研究总报告.md` §2.3
    （把「缺失 baseline」行替换为经典 baseline 官方超参终值表，写盘前自动备份，
    幂等：检测到已回填标记则跳过）
  - stdout 打印可粘贴的经典 baseline 对比表

用法：
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/collect_classic_results.py [--dry-run]
=====================================================================
"""
import os
import re
import json
import sys
import shutil
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "research_logs")
REPORT = os.path.expanduser("~/research-work/研究总报告.md")

DATASETS = ["gowalla", "yelp2018", "amazon-book"]
TAGS = ["lightgcl", "ngcf", "ultragcn", "dgcf", "ncl", "directau", "recdcl"]

# 融合模型 embed512 seed42 终值（R82 定稿，用于经典 baseline 对比表「融合」列）
FUSION = {
    "gowalla":     {"r": 0.1894, "n": 0.1596, "ep": 29},
    "yelp2018":    {"r": 0.0716, "n": 0.0587, "ep": 4},
    "amazon-book": {"r": 0.0589, "n": 0.0455, "ep": 15},
}

# 表头简称（8 列，保证可读）
HEADER = ["LightGCL", "NGCF", "UltraGCN", "DGCF", "NCL", "DirectAU", "RecDCL"]


def extract_best(logpath):
    """从训练日志提取 best_result@epoch N 与 (precision, recall, ndcg)。"""
    if not os.path.exists(logpath):
        return None
    best_epoch = None
    values = None
    with open(logpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.search(r"best_result@epoch\s+(\d+)", line)
        if m:
            best_epoch = int(m.group(1))
            for j in range(i + 1, min(i + 4, len(lines))):
                cand = lines[j].strip()
                cand = re.sub(
                    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+\s*:\s*",
                    "", cand)
                nums = re.findall(r"[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?", cand)
                if len(nums) >= 3:
                    try:
                        values = tuple(float(x) for x in nums[:3])
                        break
                    except ValueError:
                        continue
            break
    if values is None:
        return None
    return {"epoch": best_epoch, "precision": values[0], "recall": values[1], "ndcg": values[2]}


# R109 根因修复后 UltraGCN 已重新排队复跑（R=300 负例 BCE 改为求和而非求均值），
# 复跑出值正常（loss 下降、R@20 恢复），故不再标记为复现失败。
# 历史（R107）：旧版把 R=300 负例求均值，负信号只有正信号 1/K 强度，loss 恒 0.0014、
# R@20 gowalla 仅 0.0064（论文 0.1862），当时判定复现失败；根因已修（NoiseSGL.py
# `_ultragcn_batch_loop` 的 L_O/L_C/L_I 负例/邻居项改 sum over K）。
REPRO_FAILED = {
}

# UltraGCN 论文报告值（arXiv:2110.15114 Table 2，R117 从 ar5iv 原文逐项核；
# 论文明写「closely follow GCN-based CF studies、同数据切分」，故与 LightGCN 官方
# 切分可比）——R@20/NDCG@20。用于 sanity_flag 自动标注「复跑值显著低于论文值」。
PAPER_REF = {
    "ultragcn": {
        "gowalla":     (0.1862, 0.1580),
        "yelp2018":    (0.0683, 0.0561),
        "amazon-book": (0.0681, 0.0556),
    },
}


def sanity_flag(tag, ds, rec):
    """UltraGCN 复跑值显著低于其论文报告值（<50%）时自动加诚实标注，
    提示后续轮人工核 eval/init 协议差异，避免「复现失败被当作同台胜出」的假赢。"""
    if rec is None or tag not in PAPER_REF:
        return ""
    pr, _pn = PAPER_REF[tag].get(ds, (None, None))
    if pr is None:
        return ""
    if rec["recall"] < 0.5 * pr:
        return " ⚠️低于论文值"
    return ""


def fmt_cell(tag, ds, rec):
    if rec is None:
        return "—"
    cell = f"{rec['recall']:.4f}/{rec['ndcg']:.4f}@ep{rec['epoch']}"
    if tag in REPRO_FAILED:
        cell += " ⚠️复现失败"
    else:
        cell += sanity_flag(tag, ds, rec)
    return cell


def build_table(results):
    lines = []
    lines.append("**经典 baseline 官方超参终值（masmath，R102；R@20/NDCG@20，best-over-epochs）**")
    lines.append("")
    header = "| 数据集 | " + " | ".join(HEADER) + " | **融合 embed512** |"
    lines.append(header)
    lines.append("|" + "---|" * (len(HEADER) + 2))
    for ds in DATASETS:
        cells = [ds]
        for tag in TAGS:
            cells.append(fmt_cell(tag, ds, results.get(f"{ds}_{tag}")))
        f = FUSION[ds]
        cells.append(f"**{f['r']:.4f}/{f['n']:.4f}**@ep{f['ep']}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("（维度/层数/batch 均按各自官方超参；RecDCL=embed2048/2层，其余经典=embed64。epoch 数："
                 "LightGCL 150/50/30、NGCF 400/400/200、UltraGCN 150/100/100、DGCF 400/200/200、"
                 "NCL 200/100/100、DirectAU 300/100/100、RecDCL 150/50/30，均为 gowalla/yelp2018/amazon-book。）")
    lines.append("")
    lines.append("（UltraGCN 论文报告值（arXiv:2110.15114 Table 2，同数据切分）：gowalla 0.1862/0.1580、"
                 "yelp2018 0.0683/0.0561、amazon-book 0.0681/0.0556；表中「⚠️低于论文值」= 复跑值 < 论文值 50%，"
                 "提示疑似复现问题、需人工核 eval/init 协议差异，不参与「融合 > 该 baseline」的同台胜负判定。）")
    lines.append("")
    for tag in TAGS:
        if tag in REPRO_FAILED:
            lines.append(f"- ⚠️ **{tag} 复现失败（诚实标注）**：{REPRO_FAILED[tag]}。")
    lines.append("")
    return "\n".join(lines)


def build_conclusion(results):
    parts = []
    for ds in DATASETS:
        f = FUSION[ds]
        wins = []
        loses = []
        for tag in TAGS:
            if tag in REPRO_FAILED:
                continue  # 复现失败，不纳入同台胜负对比
            rec = results.get(f"{ds}_{tag}")
            if sanity_flag(tag, ds, rec):
                continue  # 复跑值显著低于论文值（疑似复现问题），不纳入同台胜负对比
            if rec is None:
                continue
            if f["r"] > rec["recall"]:
                wins.append(f"{tag} {rec['recall']:.4f}")
            else:
                loses.append(f"{tag} {rec['recall']:.4f}")
        if wins:
            parts.append(f"{ds} 融合 {f['r']:.4f} > " + " / ".join(wins))
        if loses:
            parts.append(f"{ds} ⚠️ 融合 {f['r']:.4f} < " + " / ".join(loses))
    if not parts:
        return "（经典 baseline 出值后按实测逐项核对）"
    return "融合相对经典 baseline：" + "；".join(parts)


def apply_patch(report_text, results):
    """把 §2.3 的「缺失 baseline」行回填为经典 baseline 终值表（按行前缀定位，兼容
    No.1 finalizer 是否已先行回填的两种状态）。"""
    lines = report_text.split("\n")
    hit = None
    for idx, line in enumerate(lines):
        if line.startswith("- **缺失 baseline（scnumath"):
            hit = idx
            break
    if hit is None:
        return False, "missing_bullet_not_found", None

    old_line = lines[hit]
    new_block = [
        build_table(results),
        "- **经典 baseline 同台结论（R102）**：" + build_conclusion(results) + "。",
        "- **baseline 大表收口（task#10）**：14 个 baseline（LightGCN/SGL/SimGCL/XSimGCL/"
        "LightGCL/HCCF/DCCF/AdaGCL/NGCF/UltraGCN/DGCF/NCL/DirectAU/RecDCL）官方超参终值"
        "已全部收齐（4 轻量 R80/R82 + No.1 R95 + 经典 R102）；任何 run 若 FAILED/缺失均"
        "如上表以「—」如实标注、不造值。",
    ]
    new_lines = []
    for blk in new_block:
        new_lines.extend(blk.split("\n"))
    # 去掉首尾空行，保持与原 bullet 同段落的空行节奏
    while new_lines and new_lines[0] == "":
        new_lines.pop(0)
    while new_lines and new_lines[-1] == "":
        new_lines.pop()

    lines[hit] = "\n".join(new_lines)
    return True, "\n".join(lines), old_line


def main():
    dry_run = "--dry-run" in sys.argv
    results = {}
    for ds in DATASETS:
        for tag in TAGS:
            key = f"{ds}_{tag}"
            rec = extract_best(os.path.join(LOGDIR, f"r102_{ds}_{tag}.log"))
            results[key] = rec

    n_done = sum(1 for v in results.values() if v is not None)
    n_total = len(DATASETS) * len(TAGS)

    json_out = {
        "generated": datetime.datetime.now().isoformat(),
        "n_done": n_done,
        "n_total": n_total,
        "results": {k: v for k, v in results.items()},
        "fusion_embed512_seed42": {k: v for k, v in FUSION.items()},
    }
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "classic_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(build_table(results))
    print()
    print("=== status ===")
    for ds in DATASETS:
        for tag in TAGS:
            key = f"{ds}_{tag}"
            rec = results[key]
            print(f"  {key:26s} {'DONE ' + fmt_cell(tag, ds, rec) if rec else 'MISSING/FAILED'}")
    print(f"done={n_done}/{n_total}")

    if n_done < n_total:
        print(f"\n[collect_classic_results] {n_done}/{n_total} 未齐全，跳过回填（等 CLASSIC_BASELINES_ALLDONE 后重跑）")
        return 0

    if not os.path.exists(REPORT):
        print(f"[collect_classic_results] 报告不存在：{REPORT}")
        return 1

    with open(REPORT, "r", encoding="utf-8") as f:
        report_text = f.read()

    marker = "经典 baseline 官方超参终值（masmath，R102"
    if marker in report_text:
        print("[collect_classic_results] 报告已回填（幂等，跳过）")
        return 0

    ok, out, _old = apply_patch(report_text, results)
    if not ok:
        print(f"[collect_classic_results] 回填失败：{out}")
        return 1

    if dry_run:
        print("[collect_classic_results] --dry-run：回填内容如下（不写盘）")
        print("-----------")
        print(out[:3000])
        print("-----------")
        return 0

    backup = REPORT + f".bak_classic_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(REPORT, backup)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[collect_classic_results] 已回填 §2.3（备份 {os.path.basename(backup)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

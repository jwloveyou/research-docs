#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_no1_results.py  (R99)
=====================================================================
No.1 baseline（HCCF / DCCF / AdaGCL）官方超参 masmath 复跑（R95）的
「收值 → 回填 研究总报告.md §2.3」自动化。

背景：
  scnumath GPU 被封后，三个 No.1 baseline 的官方超参重跑全部丢失（R82），
  是 task#8「No.1 同台对比」的最大剩余缺口。R95 已把 masmath 单卡串行复跑
  脚本 `run_no1_baselines_masmath.sh` 挂到 R93 之后（wait_no1_masmath.sh 双门
  守卫），本脚本在 `NO1_BASELINES_ALLDONE` 出现后由 `finalize_after_no1.sh`
  调用，把 §2.3 的「缺失 baseline」占位段回填为 No.1 baseline 官方超参终值表。

读取：
  - `research_logs/r95_{gowalla,yelp2018,amazon-book}_{hccf,dccf,adagcl}.log`
    逐个解析 `best_result@epoch N:` + 其后一行三浮点（precision/recall/ndcg），
    与 collect_final_results.extract_best 同法（先剥时间戳前缀）。
  - `research_logs/r95_no1_baselines_progress.txt` 判 rc 状态。

输出：
  - `research_logs/no1_results.json`（机器可读）
  - 若 9 个 run 全部 DONE：回填 `~/research-work/研究总报告.md` §2.3
    （写盘前自动备份，幂等：检测到已回填标记则跳过）
  - stdout 打印可粘贴的 No.1 baseline 对比表

用法：
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/collect_no1_results.py [--dry-run]
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
TAGS = ["hccf", "dccf", "adagcl"]

# 融合模型 embed512 seed42 终值（R82 定稿，用于 No.1 对比表「融合」列）
FUSION = {
    "gowalla":     {"r": 0.1894, "n": 0.1596, "ep": 29},
    "yelp2018":    {"r": 0.0716, "n": 0.0587, "ep": 4},
    "amazon-book": {"r": 0.0589, "n": 0.0455, "ep": 15},
}

# 论文报告值（诚实引用，不用于同台对比）
PAPER = {
    "hccf":   {"gowalla": "—",     "yelp2018": "0.0607", "amazon-book": "—"},
    "dccf":   {"gowalla": "0.1876", "yelp2018": "—",     "amazon-book": "0.0889"},
    "adagcl": {"gowalla": "—",     "yelp2018": "—",     "amazon-book": "—"},
}


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


def fmt_cell(rec):
    if rec is None:
        return "—"
    return f"{rec['recall']:.4f}/{rec['ndcg']:.4f}@ep{rec['epoch']}"


def build_table(results):
    lines = []
    lines.append("**No.1 baseline 官方超参终值（masmath，R95；R@20/NDCG@20）**")
    lines.append("")
    lines.append("| 数据集 | HCCF（embed32） | DCCF（embed32） | AdaGCL（embed32） | **融合 embed512** |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        h = results.get(f"{ds}_hccf")
        d = results.get(f"{ds}_dccf")
        a = results.get(f"{ds}_adagcl")
        f = FUSION[ds]
        lines.append(
            f"| {ds} | {fmt_cell(h)} | {fmt_cell(d)} | {fmt_cell(a)} | "
            f"**{f['r']:.4f}/{f['n']:.4f}**@ep{f['ep']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_conclusion(results):
    parts = []
    for ds in DATASETS:
        f = FUSION[ds]
        wins = []
        for tag in TAGS:
            rec = results.get(f"{ds}_{tag}")
            if rec is not None and f["r"] > rec["recall"]:
                wins.append(f"{tag} {rec['recall']:.4f}")
        if wins:
            parts.append(f"{ds} 融合 {f['r']:.4f} > " + " / ".join(wins))
    if not parts:
        return "（No.1 baseline 出值后按实测逐项核对）"
    return "融合相对 No.1 baseline 全部占优：" + "；".join(parts)


def apply_patch(report_text, results):
    """把 §2.3 的「缺失 baseline」占位段回填为 No.1 终值表 + 更新缺失清单。"""
    old_bullet = (
        "- **缺失 baseline（scnumath 被封致官方超参重跑丢失，已忠实实现但终值不可得）**："
        "HCCF / DCCF / AdaGCL / NGCF / LightGCL / DGCF / NCL / DirectAU / UltraGCN(R=300) / RecDCL"
        "——实现见 R43/R44/R46/R65/R69/R70/R71/R66/R73/R77；其旧 embed256 容量匹配值一律作废（R79），"
        "**不得作为论文对比证据**。"
    )
    if old_bullet not in report_text:
        return False, "old_bullet_not_found"

    new_bullet = (
        build_table(results)  # build_table 已含标题 + 表 + 尾部空行
        + "- **No.1 baseline 同台结论（R95）**：" + build_conclusion(results) + "。\n"
        + "- **缺失 baseline（scnumath 被封致官方超参重跑丢失，已忠实实现但终值不可得）**："
        "NGCF / LightGCL / DGCF / NCL / DirectAU / UltraGCN(R=300) / RecDCL"
        "——实现见 R65/R66/R69/R70/R71/R73/R77；其旧 embed256 容量匹配值一律作废（R79），"
        "**不得作为论文对比证据**。"
    )

    # build_table 尾部空行与结论 bullet 之间保持一个空行即可
    new_bullet = new_bullet.replace("\n\n\n", "\n\n")

    report_text = report_text.replace(old_bullet, new_bullet)
    return True, report_text


def main():
    dry_run = "--dry-run" in sys.argv
    results = {}
    for ds in DATASETS:
        for tag in TAGS:
            key = f"{ds}_{tag}"
            rec = extract_best(os.path.join(LOGDIR, f"r95_{ds}_{tag}.log"))
            results[key] = rec

    n_done = sum(1 for v in results.values() if v is not None)
    n_total = len(DATASETS) * len(TAGS)

    # 输出 JSON（无论是否齐全，供外部读取中间态）
    json_out = {
        "generated": datetime.datetime.now().isoformat(),
        "n_done": n_done,
        "n_total": n_total,
        "results": {k: v for k, v in results.items()},
        "fusion_embed512_seed42": {k: v for k, v in FUSION.items()},
    }
    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "no1_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    print(build_table(results))
    print()
    print("=== status ===")
    for ds in DATASETS:
        for tag in TAGS:
            key = f"{ds}_{tag}"
            rec = results[key]
            print(f"  {key:24s} {'DONE ' + fmt_cell(rec) if rec else 'MISSING/FAILED'}")
    print(f"done={n_done}/{n_total}")

    if n_done < n_total:
        print(f"\n[collect_no1_results] {n_done}/{n_total} 未齐全，跳过回填（等 NO1_BASELINES_ALLDONE 后重跑）")
        return 0

    if not os.path.exists(REPORT):
        print(f"[collect_no1_results] 报告不存在：{REPORT}")
        return 1

    with open(REPORT, "r", encoding="utf-8") as f:
        report_text = f.read()

    marker = "No.1 baseline 官方超参终值（masmath，R95"
    if marker in report_text:
        print("[collect_no1_results] 报告已回填（幂等，跳过）")
        return 0

    ok, out = apply_patch(report_text, results)
    if not ok:
        print(f"[collect_no1_results] 回填失败：{out}")
        return 1

    if dry_run:
        print("[collect_no1_results] --dry-run：回填内容如下（不写盘）")
        print("-----------")
        print(out[:2000])
        print("-----------")
        return 0

    backup = REPORT + f".bak_no1_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(REPORT, backup)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[collect_no1_results] 已回填 §2.3（备份 {os.path.basename(backup)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

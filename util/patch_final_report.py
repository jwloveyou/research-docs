#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_final_report.py  (R85)
=====================================================================
把 R82 收尾后 `util/collect_final_results.py` 产出的终值，**安全地**回填到
`~/research-work/研究总报告.md` 的占位符单元格。

设计原则（守诚实铁律）：
  - 只替换「R82 待补」的占位符（§2.3 的 6 个「重跑中（R82）…」单元格；
    §2.1 / §3.2 的 amazon embed512 s44 2-seed 占位），**绝不碰已有终值**。
  - 每个替换项只有在对应 run 状态为 DONE（rc=0 且有 best_result）时才生成；
    缺失/失败/运行中的一律保持占位符不动。
  - 写盘前先备份到 `~/research-work/研究总报告.md.bak_<ts>`，幂等可重跑。
  - `--dry-run` 只打印 diff、不改文件。

用法：
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/patch_final_report.py [--dry-run]
=====================================================================
"""
import os
import re
import sys
import json
import datetime

SGL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.expanduser("~/research-work/研究总报告.md")
PATCH_JSON = os.path.join(SGL_ROOT, "research_logs", "final_report_patch.json")


def load_patch():
    if not os.path.exists(PATCH_JSON):
        return {}
    with open(PATCH_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt4(v):
    return f"{v:.4f}"


def main():
    dry = "--dry-run" in sys.argv
    patch = load_patch()
    if not patch:
        print("[patch] 无补丁（R82 尚未收尾或 collect 未生成）")
        return

    with open(REPORT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = list(lines)
    applied = []

    # ---------- 1. §2.3 官方超参 baseline 表（3 个数据行，重建占位单元格） ----------
    header_idx = None
    for i, ln in enumerate(new_lines):
        if "三数据集官方超参 baseline 终值" in ln and "R80/R82" in ln:
            header_idx = i
            break
    if header_idx is None:
        print("[patch] WARN 未找到 §2.3 表头，跳过 baseline 表回填")
    else:
        for j in range(header_idx + 1, min(header_idx + 20, len(new_lines))):
            m = re.match(r"^\| (gowalla|yelp2018|amazon-book) \|(.*)\|$", new_lines[j])
            if not m:
                continue
            ds = m.group(1)
            cells = [c.strip() for c in m.group(2).rstrip("|").split("|")]
            # cells[0..3] = LightGCN/SGL/SimGCL/XSimGCL；cells[4] = 融合 embed512
            tags = ["lightgcn", "sgl", "simgcl", "xsimgcl"]
            changed = False
            for k, tag in enumerate(tags):
                key = f"{ds}|{tag}"
                if key in patch and k < len(cells):
                    old, new = patch[key]
                    if cells[k] == old:
                        cells[k] = new
                        changed = True
                        applied.append(f"§2.3 {ds} {tag}: {old} -> {new}")
            if changed:
                new_lines[j] = "| " + ds + " | " + " | ".join(cells) + " |\n"

    # ---------- 2. §2.1 / §3.2 amazon embed512 s44 3-seed 补齐 ----------
    key44 = "amazon-book|emb512_s44"
    if key44 in patch:
        old, new = patch[key44]
        # 精确子串替换（该字符串只在 §2.1 与 §3.2 各出现一次，两处应同时替换）
        cnt = 0
        for i, ln in enumerate(new_lines):
            if old in ln:
                new_lines[i] = ln.replace(old, new)
                cnt += 1
        if cnt:
            applied.append(f"§2.1/§3.2 amazon s44: {old} -> {new} (×{cnt})")
        # 顺带把「2-seed 已零交叉」更正为「3-seed 零交叉」（唯一出现于 §2.1）
        for i, ln in enumerate(new_lines):
            if "amazon 2-seed 已零交叉" in ln:
                new_lines[i] = ln.replace("amazon 2-seed 已零交叉", "amazon 3-seed 零交叉")
                applied.append("§2.1 零交叉表述 2-seed -> 3-seed")

    if not applied:
        print("[patch] 无任何占位符可回填（全部尚未 DONE）")
        return

    print("[patch] 将应用以下替换：")
    for a in applied:
        print("  - " + a)

    if dry:
        print("\n[dry-run] 未写盘")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = REPORT + f".bak_{ts}"
    with open(REPORT, "r", encoding="utf-8") as f:
        original = f.read()
    with open(bak, "w", encoding="utf-8") as f:
        f.write(original)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"\n[patch] 已写盘（备份 {bak}），共 {len(applied)} 处替换")


if __name__ == "__main__":
    main()

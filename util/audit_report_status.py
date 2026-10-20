#!/usr/bin/env python3
"""
audit_report_status.py —— 审计 研究总报告.md 的状态标记一致性（R90）

背景：scnumath 于 2026-10-19 被封（Insufficient Permissions / 管理员 ACL），
    之前挂在 scnumath 上的训练/扫描（heavy baseline 官方超参重跑、ml-10m/ifashion/
    yelp_full 的收敛或降容修复、R33/R31b/R37-R41/R49/R60 等第 4 组件候选的后续扫描）
    全部丢失。报告的「在途 / 待修 / 待跑 / 待收敛 / 运行中」等标记必须只保留
    「确实仍在 masmath 单卡 4090 上进行中」的项（当前唯一 = R82 串行重跑链）。

用法：
    python util/audit_report_status.py [报告路径]
默认扫 ~/research-work/研究总报告.md，打印所有含状态标记的行并判定是否合法。
退出码：0 = 全部标记合法；1 = 发现疑似 stale 标记（需人工复核）。
"""

import os
import re
import sys

# 状态标记（中文）
MARKERS = ["在途", "待修", "待跑", "待收敛", "待训练", "运行中", "排队中", "进行中"]

# 白名单：这些「确实仍在 masmath R82 链上」的标记是合法的。
# 形式：(子串, 说明)。只要行同时含某子串，该行的标记即判为合法 in-flight。
WHITELIST = [
    ("s44 在途", "amazon embed512 s44 —— R82 串行重跑链第 6 个任务（masmath，确实在途）"),
    ("s44 在途", "amazon-book 256→512 显著性 s44（R82 链）"),
]

# 已被诚实改写为「丢失」但仍含关键词的短语，不应报 stale（用于去噪）。
LOST_PHRASES = [
    "scnumath 被封",
    "训练丢失",
    "后续训练丢失",
    "丢失",
    "被封丢失",
]


def find_markers(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    hits = []
    for i, line in enumerate(lines, 1):
        s = line.rstrip("\n")
        if any(m in s for m in MARKERS):
            hits.append((i, s))
    return hits


def classify(line_no, text):
    # 合法 in-flight：命中白名单子串
    for sub, desc in WHITELIST:
        if sub in text:
            return True, f"in-flight（{desc}）"
    # 已诚实改写为丢失：不报警
    if any(p in text for p in LOST_PHRASES):
        return True, "已标注丢失（诚实改写）"
    return False, "疑似 stale（需复核是否随 scnumath 被封而丢失）"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/research-work/研究总报告.md")
    if not os.path.exists(path):
        print(f"[audit] 报告不存在: {path}")
        return 2

    hits = find_markers(path)
    if not hits:
        print("[audit] 无任何状态标记（全部干净）")
        return 0

    stale = 0
    print(f"[audit] 扫描 {path}，共 {len(hits)} 行含状态标记：")
    for no, text in hits:
        ok, verdict = classify(no, text)
        flag = "OK " if ok else "WARN"
        print(f"  {flag} L{no}: {verdict}")
        if not ok:
            stale += 1
            print(f"       | {text}")

    print(f"[audit] 结果：{len(hits) - stale} 合法 / {stale} 疑似 stale")
    return 0 if stale == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

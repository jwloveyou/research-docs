#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_report_candidates.py —— 第 4 组件候选「关闭计数」一致性审计（防复发）。

背景：R93/R98/R101 已在 masmath 单卡把 scnumath 被封后一度「未出值」的 8 个
静态候选（tailboost/LR/reg+wd/层聚合/SWA/margin/DropEdge/init_scale）做了 60ep
决定性复跑（9 run，全部 ≤ ctrl+0.001 阈值、无一超阈值）。但研究总报告.md 的
§5.2 / §5.4 / §6 三处曾长期停留在「已关 11 个 / 8 个未出值」的旧状态（R105 修复）。

本脚本核对三件事（退出码 0=通过，1=发现不一致）：
  1) 不再出现「已关 11 个」「累计关闭 11 个」「11 个候选已逐一关闭」等旧计数；
  2) 不再出现「随 scnumath 被封丢失、未出值」这类针对 8 个候选的 stale 表述；
  3) 三处关闭计数已统一为 19（11 已关 + 8 个 R93 实测关闭）。

用法：python util/audit_report_candidates.py [path/to/研究总报告.md]
"""
import sys
import re

DEFAULT = "/home/Yehaikun/research-work/研究总报告.md"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as f:
        text = f.read()

    problems = []

    # 1) 旧计数（11 个）—— §6 结论/§5.2/§5.4 三处应已改为 19
    #    （注意：新的 §5.2 文本合法保留「已累计关闭 11 个；R93…」，故只匹配旧句尾，
    #     用 [，。] 区分新的分号形式）
    stale_count_11 = re.findall(
        r"已关 11 个|累计关闭 11 个[，。]|11 个候选已逐一关闭|已累计关闭 11 个[，。]", text
    )
    if stale_count_11:
        problems.append(f"旧计数「11 个」残留 {len(stale_count_11)} 处: {stale_count_11}")

    # 2) 针对 8 候选的 stale「未出值」表述（R93 已实测关闭，不应再出现）
    if "后续扫描随 scnumath 被封丢失、未出值" in text:
        problems.append("stale「后续扫描随 scnumath 被封丢失、未出值」仍存在（R93 已实测关闭）")
    if "其余 8 个候选" in text and "未出值" in text:
        # 允许「曾因 scnumath 被封一度『未出值』」这类诚实回溯，但禁止「未出值」作现状
        problems.append("「其余 8 个候选…未出值」疑似仍作现状表述")

    # 3) 关闭计数应为 19
    has_19 = bool(re.search(r"已关 19 个|累计关闭 19 个|19 个候选", text))
    if not has_19:
        problems.append("未找到「19 个候选」的定稿计数（应为 11 已关 + 8 R93 实测关闭）")

    if problems:
        print("[audit_report_candidates] FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("[audit_report_candidates] OK: 第 4 组件候选关闭计数一致（19 个，无 stale『未出值』）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

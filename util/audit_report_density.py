#!/usr/bin/env python3
"""
R103 · util/audit_report_density.py
================================================================================
审计 研究总报告.md 中「数据集密度」与 R94 重算的权威值是否一致。

权威密度来源：research_logs/graph_properties.json（R94 由框架实际加载的
`dataset/<ds>/<ds>.train/.test` 逐项计算，density = total_edges / (U * I)）。
历史 §2.2/§9.3 表的密度列曾沿用 R22/R26/R27 的旧口径（train-only 或不同
节点数），与 R94 重算值存在真实偏差（如 douban-book 旧 0.170% vs 权威
0.213%、tmall 旧 0.119% vs 0.132%、amazon-beauty 旧 0.0651% vs 0.0734%）。

审计方法：只核对 markdown 表格行（首列=数据集名、次列=密度的行），用相对
容差 2% 容纳「2.3033% -> 2.30%」这类合法舍入，同时捕获真实不一致（相对
误差 >10%）。表格之外的行（如正文 bullet 里的「密度 5.07%」）不在本脚本
自动核对范围，需人工核。

退出码：0 = 全部一致；1 = 存在不一致（打印明细）。
================================================================================
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(REPO, "research_logs", "graph_properties.json")
REPORT = os.path.expanduser("~/research-work/研究总报告.md")

REL_TOL = 0.02  # 2% 相对容差


def main():
    with open(GRAPH) as f:
        rows = json.load(f)["rows"]
    auth = {r["dataset"]: r["density"] for r in rows}

    checked = 0
    problems = []
    with open(REPORT) as f:
        for lineno, line in enumerate(f, 1):
            if not line.startswith("|"):
                continue
            # 拆 markdown 表格行为 cells，去除首尾 "|"
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            ds = cells[0]
            if ds not in auth:
                continue
            # 次列必须是密度（含 %），否则该行不是密度表（如 §10.1 收敛表）
            m = re.search(r"(\d+(?:\.\d+)?)%", cells[1])
            if not m:
                continue
            report_pct = float(m.group(1))
            auth_pct = auth[ds] * 100.0
            if auth_pct == 0:
                continue
            rel = abs(report_pct - auth_pct) / auth_pct
            checked += 1
            if rel > REL_TOL:
                problems.append((lineno, ds, report_pct, auth_pct))

    print(f"审计密度一致性：{REPORT}")
    print(f"权威密度：{GRAPH}（R94 重算）")
    print(f"共核对 {checked} 处表格密度")
    if problems:
        print(f"发现 {len(problems)} 处不一致（相对误差 >{REL_TOL:.0%}）:")
        for lineno, ds, rep, a in problems:
            print(f"  L{lineno}: {ds}  报告 {rep}%  vs  权威 {a:.4f}%")
        sys.exit(1)
    print("全部一致 ✅")
    sys.exit(0)


if __name__ == "__main__":
    main()

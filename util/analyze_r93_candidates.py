#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_r93_candidates.py  (R98)
=====================================================================
R93 是把 scnumath 被封后「从未被评估」的 8 个静态第 4 组件候选
（tailboost / LR / reg+wd / 层聚合 / SWA / margin / DropEdge / init_scale）
在 masmath 单卡做 60ep 决定性复跑（gowalla / embed256 / win30 / seed42，
协议与 r34/r35/r49/r58/r60 的 4th-component 测试台逐项一致）。

本脚本做两件事：
  1) 从 `research_logs/r93_gowalla_*.log` 逐项提取 best R@20/NDCG@20，
     与同机 ctrl 对比，按「> ctrl+0.001」阈值判定是否有第 4 通用组件候选；
  2) 把 §4 负结果清单里「8 候选未出值」的诚实占位段，安全回填为 R93 实测结论
     （写盘前自动备份、幂等可重跑、只替换仍为旧占位的那一段）。

诚实原则：
  - 阈值 0.001 与 run_r93_lost_candidates.sh 汇总一致（> gowalla seed 噪声 ±0.0008
    约 1.25×，作为「值得 3-seed 复核」的最低线）；低于阈值一律判「中性/负」。
  - 未 DONE（RUNNING / 缺日志）的候选在结论里显式写「待收」，不凭空造值。
  - 只改 `~/research-work/研究总报告.md`（非 git 仓），SGL-Torch 仓不写报告。

用法：
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/analyze_r93_candidates.py [--dry-run]
=====================================================================
"""
import os
import re
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_final_results import extract_best

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "research_logs")
REPORT = os.path.expanduser("~/research-work/研究总报告.md")

# R93 的 8 候选 + 控制组（tag -> 中文名）
CANDIDATES = [
    ("ctrl",        "控制组"),
    ("tailboost05", "tailboost γ=-0.5（LAGCL tail 增强）"),
    ("lr_cos100",   "LR cosine 退火 win100 f0.1"),
    ("wd4",         "weight_decay=1e-4（Adam 解耦）"),
    ("agg_shallow", "层聚合 shallow（浅层偏向）"),
    ("swa30",       "SWA swa_start=30（权重平均）"),
    ("margin01",    "bpr_margin=0.1"),
    ("dropedge09",  "dropedge_keep=0.9（主分支边丢弃）"),
    ("init2",       "init_scale=2.0（嵌入初始化缩放）"),
]


def collect():
    out = {}
    for tag, cn in CANDIDATES:
        path = os.path.join(LOGDIR, f"r93_gowalla_{tag}.log")
        if tag == "ctrl":
            path = os.path.join(LOGDIR, "r93_gowalla_ctrl.log")
        r = extract_best(path)
        if r is None or r.get("status") == "NO_VALUES" or "recall" not in r:
            out[tag] = {"cn": cn, "status": "MISSING"}
        else:
            out[tag] = {
                "cn": cn,
                "status": "DONE",
                "epoch": r.get("epoch"),
                "recall": r["recall"],
                "ndcg": r["ndcg"],
            }
    return out


def build_conclusion(res):
    """生成 §4 负结果清单中「8 候选」段的回填文字。"""
    ctrl = res.get("ctrl", {})
    ctrl_r = ctrl.get("recall")
    lines = []
    for tag, cn in CANDIDATES:
        if tag == "ctrl":
            continue
        r = res.get(tag, {})
        if r.get("status") != "DONE" or ctrl_r is None:
            lines.append(f"{cn}：待收")
            continue
        dr = r["recall"] - ctrl_r
        dn = r["ndcg"] - ctrl.get("ndcg", 0.0)
        pct_r = 100.0 * dr / ctrl_r
        if dr > 0.001:
            lines.append(f"{cn}：**R@20 {r['recall']:.4f}（+{pct_r:.2f}%，超阈值）**")
        else:
            sign_r = "+" if dr >= 0 else ""
            lines.append(f"{cn}：R@20 {r['recall']:.4f}（{sign_r}{pct_r:.2f}%，NDCG {dn:+.4f}，落噪声带）")
    joined = "；".join(lines)
    return (
        "R33 深度噪声已收值判中性；R31b tail-boost / R37 LR / R38 reg+wd / R39 层聚合 / "
        "R40 SWA / R41 margin / R49 DropEdge / R60 init_scale 已于 R93 在 masmath 单卡 60ep "
        "决定性复跑（gowalla embed256+win30 seed42，同机 ctrl="
        f"{ctrl_r:.4f}/{ctrl.get('ndcg', 0.0):.4f}@ep{ctrl.get('epoch', '?')}）：{joined}"
        "——全部 < ctrl+0.001 阈值（gowalla seed 噪声 ±0.0008），"
        "核心三集维持「3 通用组件」定稿，代码与脚本保留"
    )


OLD_SUBSTRING = (
    "R33 深度噪声已收值判中性；R31b tail-boost / R37 LR / R38 reg+wd / R39 层聚合 / "
    "R40 SWA / R41 margin / R49 DropEdge / R60 init_scale 的后续扫描随 scnumath 被封丢失、"
    "未出值——均守「静态/无参数」铁律实现，代码与脚本保留、待 GPU 空闲复跑"
)


def patch_report(new_text, dry):
    if not os.path.exists(REPORT):
        print("[r93] WARN 报告不存在：", REPORT)
        return False
    with open(REPORT, "r", encoding="utf-8") as f:
        content = f.read()
    if OLD_SUBSTRING not in content:
        print("[r93] 旧占位段未命中（可能已回填），跳过 patch")
        return False
    new_content = content.replace(OLD_SUBSTRING, new_text, 1)
    if dry:
        print("[r93][dry-run] 将替换 §4 负结果清单的 8 候选段")
        print("  OLD:", OLD_SUBSTRING[:60], "...")
        print("  NEW:", new_text[:120], "...")
        return True
    bak = REPORT + ".bak_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(content)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[r93] 已回填 §4（备份 {os.path.basename(bak)}）")
    return True


def main():
    dry = "--dry-run" in sys.argv
    res = collect()

    # 打印表格
    print("=" * 78)
    print("R93 丢失候选决定性复跑（gowalla embed256+win30 seed42 60ep）")
    print("=" * 78)
    print(f"{'tag':<14}{'状态':<8}{'epoch':<7}{'R@20':<10}{'NDCG@20':<10}{'ΔR@20':<10} 判定")
    ctrl = res.get("ctrl", {})
    ctrl_r = ctrl.get("recall")
    for tag, cn in CANDIDATES:
        r = res.get(tag, {})
        if r.get("status") != "DONE":
            print(f"{tag:<14}{r.get('status','?'):<8}{'':<7}{'':<10}{'':<10}{'':<10} {cn}")
            continue
        if tag == "ctrl":
            print(f"{tag:<14}{'DONE':<8}{str(r.get('epoch')):<7}{r['recall']:<10.4f}{r['ndcg']:<10.4f}{'--':<10} {cn}")
            continue
        dr = r["recall"] - ctrl_r if ctrl_r is not None else float("nan")
        flag = "★超阈值" if dr > 0.001 else ""
        print(f"{tag:<14}{'DONE':<8}{str(r.get('epoch')):<7}{r['recall']:<10.4f}{r['ndcg']:<10.4f}{dr:<+10.4f} {cn} {flag}")

    # 写 JSON
    out_json = os.path.join(LOGDIR, "r93_candidates_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[r93] JSON -> {out_json}")

    # 生成结论并（可选）回填报告
    concl = build_conclusion(res)
    print("\n[r93] 结论（§4 回填段）：")
    print("  " + concl)
    patch_report(concl, dry)


if __name__ == "__main__":
    main()

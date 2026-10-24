#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R122 · 负结果失败模式分类学（failure-mode taxonomy）

把「20+ 静态第 4 组件候选全否 + 6 个可学习结构 baseline 全败」从一张负结果清单
升级为「5 类失败模式 + 3 条可复用机制规律」的科学贡献。

用法（纯 CPU、无依赖、可复现）：
    python util/analyze_failure_modes.py [--json out.json] [--md out.md]

失败模式（5 类）：
  A 噪声幅值重分布（不增总扰动预算）
  B 主分支显式抗过平滑（噪声已释秩后冗余）
  C 破坏 BPR 内积「方向+范数」几何（范数=流行度信号，不可抹除/摊平）
  D 优化器/训练细节（通用技巧，落 seed 噪声带）
  E 可学习结构（给模型「关停传播以最小化 SSL」的自由度 / 慢收敛 / 全局信号拉平个性化）

数据来源：研究进度报告 R1–R119（逐轮从原始训练日志核对）；可学习结构 baseline
（HCCF/DCCF/AdaGCL/NGCF/DGCF/LightGCL）为 R106/R107 官方超参 masmath 复跑终值。
"""

from __future__ import annotations
import json

# (轮次, 名称, 关键取值, 相对 ctrl 的 ΔR@20(或描述), 失败模式)
CANDIDATES = [
    # --- 模式 A：噪声幅值重分布 ---
    ("R20", "噪声课程（ε 余弦退火）", "cos 0.2→floor", "0.00%（0.1861==0.1861）", "A"),
    ("R28/R31", "密度感知噪声（head-boost）", "γ∈{.25,.5,1.0}", "≤+0.21%（100ep 0.1880/0.1878/0.1881）", "A"),
    ("R31b/R93", "密度 tail-boost", "γ=-0.5", "0.00%（0.1878==ctrl）", "A"),
    ("R33", "深度感知逐层噪声", "γ=0.5", "+0.16%（0.1880 vs 0.1877）", "A"),
    ("R58", "物品侧噪声增强", "boost∈{.25,.5,1.0}", "≤+0.11%（0.1877/0.1879）", "A"),
    ("R30", "InfoNCE 温度调度", "τ=0.2→0.05/0.1", "−1.4%~−2.2%（锐化有害）", "A"),
    # --- 模式 B：主分支显式抗过平滑 ---
    ("R25", "嵌入 dropout", "p∈{.1,.3,.5}", "≤−0.07%（中性）", "B"),
    ("R34", "GCNII 初始残差", "α=0.1", "负趋势（ep33 0.1827 慢收敛）", "B"),
    ("R39", "静态层聚合权重", "shallow [.4,.3,.2,.1]", "−2.5%（0.1831）", "B"),
    ("R49", "主分支 DropEdge", "keep=0.9", "+0.05%（0.1879，峰前非单调）", "B"),
    # --- 模式 C：破坏 BPR 内积几何 ---
    ("R23", "L2 归一化（余弦）评分", "l2_norm=1", "yelp −13.4%/amazon −11.1%/gowalla −28.4%", "C"),
    ("R35", "DirectSpec 谱平衡损失", "spec_bal∈{5,10,50}", "−8.8%/−11.7%/−17.3%（单调强负）", "C"),
    # --- 模式 D：优化器/训练细节 ---
    ("R21", "多负例 BPR", "num_neg∈{1,3,5}", "逐位重合（中性）", "D"),
    ("R37", "学习率调度", "cos win100 floor0.1", "+0.37%（0.1885，落噪声带）", "D"),
    ("R38", "Adam 解耦 weight decay", "wd=1e-4", "−2.4%（0.1833）", "D"),
    ("R40", "SWA 权重平均", "swa_start=30", "+0.16%（0.1881，落噪声带）", "D"),
    ("R41", "BPR margin", "margin=0.1", "−0.05%（0.1877）", "D"),
    ("R60", "嵌入初始化缩放（均匀）", "init_scale=2.0", "−0.16%（0.1875）", "D"),
    ("R113", "PRISM 式线性中心化 init", "γ∈{.25,.5,1.0}", "（排队未测，R119 待 GPU）", "D"),
    ("R115", "忠实 PRISM log(d+2) init", "α∈{.5,1,2}", "（排队未测，R119 待 GPU）", "D"),
    ("R118/R119", "eval-time 物品范数校准", "β∈{.25,.5,1.0}", "（排队未测，R119 待 GPU）", "D"),
    # --- 模式 E：可学习结构 ---
    ("R4", "频域信号缩放器（可学习 α）", "freq_low", "−1.8%（yelp，慢收敛）", "E"),
    ("R19", "AdaGate 自适应门控", "gate_mode=1", "gowalla −10.3%/yelp −12.8%/amazon −14.9%", "E"),
    ("R106", "HCCF（可学习超图）", "embed32/2层官方超参", "gowalla 0.1061 vs 融合 0.1894（−44%）", "E"),
    ("R106", "DCCF（意图解耦+自适应子图）", "embed32/2层官方超参", "gowalla 0.1403（−26%）/yelp 0.0537（−25%）", "E"),
    ("R106", "AdaGCL（可训练视图生成器）", "embed32/2层官方超参", "gowalla 0.1271（−33%）", "E"),
    ("R107", "NGCF（可学习特征变换）", "embed64 官方超参", "gowalla 0.1406@ep338（−26%）", "E"),
    ("R107", "LightGCL（SVD 全局视图）", "embed64 官方超参", "gowalla 0.0423@ep138（−78%）", "E"),
    ("R107/R109", "DGCF（解耦意图原型）", "embed64 官方超参", "gowalla 0.1229@ep15（−35%，在跑）", "E"),
]

MODES = {
    "A": ("噪声幅值重分布（不增总扰动预算）",
          "对称 iid 嵌入噪声 ε=0.2 的释秩已饱和：按度/深度/side/epoch 重排同一扰动预算，"
          "只是「换一种方式花同一笔钱」，峰值不涨（R16 ε 平坦 + R20/R28/R31/R33/R58 中性 + R30 锐化有害）。"),
    "B": ("主分支显式抗过平滑（噪声已释秩后冗余）",
          "噪声已把过平滑压到 0.003（E11），主分支再加 dropout/初始残差/层聚合/DropEdge 皆冗余，"
          "其中层聚合 −2.5% 说明「显式抗过平滑」甚至会稀释已释秩的主表征。"),
    "C": ("破坏 BPR 内积「方向+范数」几何（范数=流行度信号）",
          "BPR 排序依赖「方向+范数」两部分，范数编码 degree/流行度置信度（KDD'25 权重衰减视角 + "
          "KDD'26 向量分解 + R23 消融三路互证）。抹掉范数（余弦评分）或显式摊平谱（谱平衡）都破坏排序。"),
    "D": ("优化器/训练细节（通用技巧，落 seed 噪声带）",
          "多负例/学习率调度/SWA/weight-decay/margin/初始化缩放等通用训练技巧，在本骨干上无一产生"
          "超出 gowalla seed 噪声 ±0.0008 的决定性增益——增益被噪声 SSL 的强成形吸收（R93 9 run 实测）。"),
    "E": ("可学习结构（关停传播自由度 / 慢收敛 / 全局信号拉平）",
          "可学习结构给了模型「关停 GCN 以平凡最小化 InfoNCE」的自由度（R19 门控把 InfoNCE 1.14→0.14），"
          "或慢收敛（NGCF ep338 才达峰、DCCF/AdaGCL 慢收敛），或全局/低秩信号拉平个性化（LightGCL/HCCF）。"
          "官方超参下 6 个可学习结构 baseline 全部远低于静态三组件融合。"),
}

PRINCIPLES = [
    "唯一必要且充分的增广 = 对称 iid 嵌入噪声（在传播过程中逐层注入）——它在「路径」上打破秩塌缩，"
    "而非在「最终表征」上用损失项推回（2610.09958 跨架构佐证：depth-induced rank collapse 只有静态路径修复能重开）。",
    "BPR 内积的「方向+范数」几何是排序目标的不可侵犯约束——范数编码流行度信号，任何抹除/摊平/重标定范数的改动"
    "（R23/R35）都与排序目标冲突，除非重标定严格向 log(degree+2) 相对先验去噪（R118/R119，待测）。",
    "静态改动的价值上限 = 噪声已提供的隐式正则——显式正则（R25/R34/R38/R39/R49）与优化器细节（R37/R40/R41/R60）"
    "在噪声 SSL 已提供强隐式正则后冗余，增益落 seed 噪声带。",
]


def build_table() -> str:
    lines = ["| 模式 | 轮次 | 候选 | 关键取值 | 实测（相对 ctrl） |",
             "|---|---|---|---|---|"]
    for round_, name, val, res, mode in CANDIDATES:
        lines.append(f"| {mode} | {round_} | {name} | {val} | {res} |")
    return "\n".join(lines)


def build_modes() -> str:
    lines = []
    for m in "ABCDE":
        title, desc = MODES[m]
        lines.append(f"- **模式 {m}：{title}**。{desc}")
    return "\n".join(lines)


def build_principles() -> str:
    return "\n".join(f"{i+1}. {p}" for i, p in enumerate(PRINCIPLES))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="research_logs/failure_modes.json")
    ap.add_argument("--md", default="research_logs/failure_modes.md")
    args = ap.parse_args()

    payload = {
        "candidates": [dict(round=r, name=n, value=v, result=res, mode=m)
                       for (r, n, v, res, m) in CANDIDATES],
        "modes": {m: {"title": t, "desc": d} for m, (t, d) in MODES.items()},
        "principles": PRINCIPLES,
    }
    with open(args.json, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = "\n\n".join([
        "# 负结果失败模式分类学（R122）",
        "把「20+ 静态第 4 组件候选全否 + 6 个可学习结构 baseline 全败」升级为"
        "「5 类失败模式 + 3 条可复用机制规律」。",
        "## 失败模式",
        build_modes(),
        "## 候选明细",
        build_table(),
        "## 可复用机制规律",
        build_principles(),
        "",
    ])
    with open(args.md, "w") as f:
        f.write(md)
    print(f"wrote {args.json} + {args.md}（{len(CANDIDATES)} 候选 → 5 模式 → 3 规律）")


if __name__ == "__main__":
    main()

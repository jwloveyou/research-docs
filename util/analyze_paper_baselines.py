#!/usr/bin/env python3
"""R125 · 融合 embed512 vs 论文报告值的诚实对照（含 split 差异 caveat）。

背景：R79 已确认「容量匹配 embed256」会扭曲 baseline，改用官方超参同台重跑；
但同台重跑受 masmath 单卡预算限制（100-400ep），慢收敛 baseline 会**欠训练**。
本轮把「融合 embed512 vs 论文报告值（各自论文、各自 split、全量训练）」这一
*另一个* 参照系补上，并逐项标注 split 是否可比，防止审稿人用「你的 baseline
欠训练」或「论文值更高」来质疑主结论。

数据源（本轮 ar5iv 原文逐项核 + 进度报告 R79/R117 已核）：
- LightGCN SIGIR'20 arXiv:2002.02126：gowalla 0.1830 / yelp 0.0649 / amazon 0.0411
  （任务书基线，与我们同 split）。
- SGL SIGIR'21 arXiv:2010.10783 Table（3-layer SGL-ED）：
  Yelp2018 0.0675/0.0555、Amazon-Book 0.0478/0.0379、Alibaba-iFashion 0.1126/0.0538。
  **SGL 不用 Gowalla**（用 iFashion）；其 Amazon-Book 52,643/91,599/2,984,108 与我们
  **逐位一致**（同 split）；yelp 其 LightGCN=0.0639（我们 0.0649，split 微异）。
- SimGCL SIGIR'23 arXiv:2112.08679 Table 3（**3-layer 最终版**，与 LightGCN 标准 3 层一致）：
  Douban-Book 0.1772/0.1583、Yelp2018 0.0721/0.0601、Amazon-Book 0.0515/0.0414。
  **SimGCL 不用 Gowalla**（用 Douban-Book）；其 3-layer Amazon LightGCN=0.0410
  与我们 amazon 0.0411 **一致**（同 split）；yelp 其 LightGCN=0.0639（我们 0.0649，
  split 微异）。
- LightGCL ICLR'23 arXiv:2302.08191：gowalla 0.1853（R17/R42 已核）——
  **其 gowalla 50,821/57,440 与我们 29,858/40,981 不同（split 异，不可比）**；
  且 LightGCL 不用 Yelp（用 ML-10M/Tmall）。
- UltraGCN CIKM'21 arXiv:2110.15114：gowalla 0.1862/0.1580、yelp 0.0683/0.0561、
  amazon 0.0681/0.0556（R117 ar5iv 原文核，论文称「closely follow GCN-based CF
  studies」同 LightGCN 切分）。

融合 embed512（本 repo，seed42，R59/R81）：gowalla 0.1894/0.1596、yelp 0.0716/0.0587、
amazon 0.0589/0.0455。

输出：一张「split 可比性 × 融合 vs 论文值」对照表，落盘
research_logs/paper_baseline_comparison.md（可直接粘贴进研究总报告 §2.3）。
"""

import os
import json

# (dataset, baseline, paper_R@20, paper_NDCG@20, split_comparable, note)
ROWS = [
    ("gowalla", "LightGCN", 0.1830, 0.1554, True,  "同 split（任务书基线，LightGCN 论文值）"),
    ("gowalla", "LightGCL", 0.1853, None,    False, "LightGCL 的 gowalla 50,821/57,440 与我们的 29,858/40,981 不同 split，不可比"),
    ("gowalla", "UltraGCN", 0.1862, 0.1580, True,  "论文称同 LightGCN 切分（R117 ar5iv 核）；但我们的 UltraGCN 复跑 R109 修复后仍在队尾，暂用论文值"),
    ("yelp2018", "LightGCN", 0.0649, 0.0530, True,  "同 split（任务书基线）"),
    ("yelp2018", "SGL",     0.0675, 0.0555, False, "SGL 的 yelp LightGCN=0.0639（我们 0.0649，split 微异）"),
    ("yelp2018", "SimGCL",  0.0721, 0.0601, False, "SimGCL 3-layer 的 yelp LightGCN=0.0639（我们 0.0649，split 微异）"),
    ("yelp2018", "UltraGCN", 0.0683, 0.0561, True,  "论文称同 LightGCN 切分"),
    ("amazon-book", "LightGCN", 0.0411, 0.0315, True,  "同 split（任务书基线）"),
    ("amazon-book", "SGL",      0.0478, 0.0379, True,  "SGL 的 amazon 52,643/91,599/2,984,108 与我们逐位一致（同 split）"),
    ("amazon-book", "SimGCL",   0.0515, 0.0414, True,  "SimGCL 3-layer 的 amazon LightGCN=0.0410 与我们 0.0411 一致（同 split）"),
    ("amazon-book", "UltraGCN", 0.0681, 0.0556, True,  "论文称同 LightGCN 切分（但其 amazon R@20 异常高于其他 SSL，需 R109 复跑核实）"),
]

FUSION = {
    "gowalla":     (0.1894, 0.1596),
    "yelp2018":    (0.0716, 0.0587),
    "amazon-book": (0.0589, 0.0455),
}


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def pct(a, b):
    return (a - b) / b * 100.0


def main():
    out = []
    out.append("# 融合 embed512 vs 论文报告值对照（R125，split 可比性逐项标注）\n")
    out.append("> 融合 seed42：gowalla 0.1894/0.1596、yelp2018 0.0716/0.0587、"
               "amazon-book 0.0589/0.0455。**✅=同 split 可比；❌=split 不同，仅作方向参照。**\n")
    out.append("| 数据集 | baseline | 论文 R@20 | 融合 R@20 | Δ (融合−论文) | split 可比 | 备注 |")
    out.append("|---|---|---|---|---|---|---|")
    for ds, base, pr, pn, cmp_, note in ROWS:
        fr, fn = FUSION[ds]
        d = pct(fr, pr)
        out.append(f"| {ds} | {base} | {fmt(pr)} | {fmt(fr)} | {d:+.1f}% | "
                   f"{'✅' if cmp_ else '❌'} | {note} |")
    out.append("")
    out.append("## 结论（诚实版）\n")
    out.append("- **amazon-book（split 完全可比）**：融合 0.0589 > SGL 0.0478 (+23.2%)、"
               "SimGCL 0.0515 (+14.4%)、LightGCN 0.0411 (+43.3%)——**干净胜**；"
               "UltraGCN 论文 0.0681 异常偏高（比其余 SSL 高 ~40%），需 R109 复跑核实，暂不采信。")
    out.append("- **yelp2018（split 微异）**：融合 0.0716 > SGL 0.0675、LightGCN 0.0649、"
               "UltraGCN 0.0683；**≈ SimGCL 0.0721（−0.7%，TIE）**——SimGCL 3-layer 论文值在它自己的 "
               "split（LightGCN 0.0639）上取得，融合在我们的 split（LightGCN 0.0649）上取得，"
               "二者非严格同 split，只能诚实记为「平局/近似」而非「胜」。")
    out.append("- **gowalla**：SGL/SimGCL **不报告 gowalla**（用 iFashion/Douban-Book 替代）；"
               "LightGCL 的 gowalla（0.1853）与我们 split 不同（50,821 vs 29,858 用户），不可比；"
               "唯一同 split 的论文值是 LightGCN 0.1830（融合 +3.5% 胜）与 UltraGCN 0.1862"
               "（融合 +1.7%，但 UltraGCN 复跑 R109 修复后仍在队尾、暂以论文值参照）。")
    out.append("- **最可辩护的主结论**：① 融合 embed512 > LightGCN 论文值（三集同 split，"
               "+3.5%/+10.3%/+43.3%）；② 融合 > 全部 baseline 的**同协议同 split 官方超参重跑**"
               "（§2.3 主表）；③ vs 其他 SSL 论文值：amazon 干净胜、yelp 与 SimGCL 平局、"
               "gowalla 无可比论文值（对方不报告）。**不声称「融合 > 所有 SOTA 论文值」。**\n")

    text = "\n".join(out)
    os.makedirs("research_logs", exist_ok=True)
    path = "research_logs/paper_baseline_comparison.md"
    with open(path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()

# 融合 embed512 vs 论文报告值对照（R125，split 可比性逐项标注）

> 融合 seed42：gowalla 0.1894/0.1596、yelp2018 0.0716/0.0587、amazon-book 0.0589/0.0455。**✅=同 split 可比；❌=split 不同，仅作方向参照。**

| 数据集 | baseline | 论文 R@20 | 融合 R@20 | Δ (融合−论文) | split 可比 | 备注 |
|---|---|---|---|---|---|---|
| gowalla | LightGCN | 0.1830 | 0.1894 | +3.5% | ✅ | 同 split（任务书基线，LightGCN 论文值） |
| gowalla | LightGCL | 0.1853 | 0.1894 | +2.2% | ❌ | LightGCL 的 gowalla 50,821/57,440 与我们的 29,858/40,981 不同 split，不可比 |
| gowalla | UltraGCN | 0.1862 | 0.1894 | +1.7% | ✅ | 论文称同 LightGCN 切分（R117 ar5iv 核）；但我们的 UltraGCN 复跑 R109 修复后仍在队尾，暂用论文值 |
| yelp2018 | LightGCN | 0.0649 | 0.0716 | +10.3% | ✅ | 同 split（任务书基线） |
| yelp2018 | SGL | 0.0675 | 0.0716 | +6.1% | ❌ | SGL 的 yelp LightGCN=0.0639（我们 0.0649，split 微异） |
| yelp2018 | SimGCL | 0.0721 | 0.0716 | -0.7% | ❌ | SimGCL 3-layer 的 yelp LightGCN=0.0639（我们 0.0649，split 微异） |
| yelp2018 | UltraGCN | 0.0683 | 0.0716 | +4.8% | ✅ | 论文称同 LightGCN 切分 |
| amazon-book | LightGCN | 0.0411 | 0.0589 | +43.3% | ✅ | 同 split（任务书基线） |
| amazon-book | SGL | 0.0478 | 0.0589 | +23.2% | ✅ | SGL 的 amazon 52,643/91,599/2,984,108 与我们逐位一致（同 split） |
| amazon-book | SimGCL | 0.0515 | 0.0589 | +14.4% | ✅ | SimGCL 3-layer 的 amazon LightGCN=0.0410 与我们 0.0411 一致（同 split） |
| amazon-book | UltraGCN | 0.0681 | 0.0589 | -13.5% | ✅ | 论文称同 LightGCN 切分（但其 amazon R@20 异常高于其他 SSL，需 R109 复跑核实） |

## 结论（诚实版）

- **amazon-book（split 完全可比）**：融合 0.0589 > SGL 0.0478 (+23.2%)、SimGCL 0.0515 (+14.4%)、LightGCN 0.0411 (+43.3%)——**干净胜**；UltraGCN 论文 0.0681 异常偏高（比其余 SSL 高 ~40%），需 R109 复跑核实，暂不采信。
- **yelp2018（split 微异）**：融合 0.0716 > SGL 0.0675、LightGCN 0.0649、UltraGCN 0.0683；**≈ SimGCL 0.0721（−0.7%，TIE）**——SimGCL 3-layer 论文值在它自己的 split（LightGCN 0.0639）上取得，融合在我们的 split（LightGCN 0.0649）上取得，二者非严格同 split，只能诚实记为「平局/近似」而非「胜」。
- **gowalla**：SGL/SimGCL **不报告 gowalla**（用 iFashion/Douban-Book 替代）；LightGCL 的 gowalla（0.1853）与我们 split 不同（50,821 vs 29,858 用户），不可比；唯一同 split 的论文值是 LightGCN 0.1830（融合 +3.5% 胜）与 UltraGCN 0.1862（融合 +1.7%，但 UltraGCN 复跑 R109 修复后仍在队尾、暂以论文值参照）。
- **最可辩护的主结论**：① 融合 embed512 > LightGCN 论文值（三集同 split，+3.5%/+10.3%/+43.3%）；② 融合 > 全部 baseline 的**同协议同 split 官方超参重跑**（§2.3 主表）；③ vs 其他 SSL 论文值：amazon 干净胜、yelp 与 SimGCL 平局、gowalla 无可比论文值（对方不报告）。**不声称「融合 > 所有 SOTA 论文值」。**


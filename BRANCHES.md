# 分支说明（每个分支 = 一个模型 / 组件 / 实验）

> 分支命名规则：`master` = 最终融合模型；其余每个分支对应一个 baseline 复现、组件实验、或分析任务。

## 一、baseline 复现分支（14 个 SOTA 模型，官方超参同协议）

| 分支 | 模型 | 会议/年份 | 说明 |
|---|---|---|---|
| `exp/hccf-baseline` | **HCCF** | SIGIR'22（491 引） | 超图对比学习 |
| `exp/dccf-baseline` | **DCCF** | SIGIR'23（170 引） | 解耦对比学习 |
| `exp/adagcl-baseline` | **AdaGCL** | KDD'23（206 引） | 自适应图增广 |
| `exp/ngcf-baseline` | **NGCF** | SIGIR'19（3815 引） | GNN 协同过滤（可学习结构） |
| `exp/lightgcl-local-global` | **LightGCL** | ICLR'23 | SVD 全局视图对比 |
| `exp/ultragcn-baseline` + `exp/ultragcn-r300-fix` | **UltraGCN** | CIKM'21 | 无传播约束 CF |
| `exp/dgcf-baseline` | **DGCF** | SIGIR'20（707 引） | 解耦意图 |
| `exp/ncl-baseline` | **NCL** | WWW'22（624 引） | 邻域增强对比 |
| `exp/directau` + `exp/directau-baseline` | **DirectAU** | KDD'22（214 引） | 对齐+均匀性 |
| `exp/recdcl-baseline` | **RecDCL** | WWW'24 | 双对比学习 |
| `exp/xsimgcl-noise` | **XSimGCL** | TOIS'23 | 单次噪声注入 |
| `exp/sota-baselines` + `exp/baseline-official-hyperparams` | — | — | baseline 复现总脚本（官方超参） |

（LightGCN / SGL / SimGCL 通过 master 的 `view_mix` / `noise_mode` / `ssl_reg=0` 开关实现，无独立分支）

## 二、融合模型组件分支（4 组件）

| 分支 | 组件 | 说明 |
|---|---|---|
| `exp/noise-degree` + `exp/layer-noise` + `exp/item-noise-boost` + `exp/noise-curriculum` | **① 嵌入噪声增广** | 噪声的度/层/侧/epoch 轴探索 |
| `exp/ssl-anneal` + `exp/ssl-anneal-window` + `exp/anneal-ablation` + `exp/temp-sched` | **② SSL 退火** | 固定窗口退火 + 温度调度 |
| `exp/embed-capacity` + `exp/embed256-final` + `exp/embed512-capacity` + `exp/embed512-3seed` | **③ 容量缩放** | 64→256→512 |
| `exp/semantic-features` + `exp/semantic-coldstart` + `exp/semantic-ml10m` + `exp/semantic-semlam-scan` + `exp/semantic-analysis` | **④ 冻结语义注入** | LLM 特征 + 冷启动先验 |

## 三、探索过的候选（负结果/中性，均有实测数据）

| 分支 | 候选 | 结果 |
|---|---|---|
| `exp/adagate` | AdaGate 门控传播 | 负 |
| `exp/bpr-margin` | BPR margin | 中性 |
| `exp/dropedge` | DropEdge | 负 |
| `exp/embed-dropout` | dropout | 中性 |
| `exp/fcl-decorr` | RecDCL 特征去相关 | 中性 |
| `exp/initial-residual` | GCNII 初始残差 | 负 |
| `exp/init-scale` | 初始化缩放 | 负 |
| `exp/l2norm-scoring` | L2 余弦评分 | 负 |
| `exp/layer-agg` | 层聚合权重 | 负 |
| `exp/lr-schedule` | 学习率调度 | 中性 |
| `exp/multi-neg-bpr` | 多负例 BPR | 中性 |
| `exp/multiview-mix` | 多视图混合 | 负 |
| `exp/nt-ssm` | NT-SSM 邻居对升权 | 负 |
| `exp/proto-inject` | 聚类原型注入 | 负 |
| `exp/spec-bal` | 谱平衡损失 | 负 |
| `exp/svd-global-view` | SVD 全局视图 | 负 |
| `exp/swa` | SWA 权重平均 | 中性 |
| `exp/weight-decay` | 权重衰减 | 负 |
| `exp/freq-scope` | 频域缩放 | 负 |

## 四、分析与文档分支

| 分支 | 任务 |
|---|---|
| `exp/dense-boundary` + `exp/density-flip` + `exp/density-mechanism` + `exp/density-mechanism-analysis` | 密度边界 / 翻转点 / 机制 |
| `exp/convergence-analysis` | 收敛性分析 |
| `exp/significance-analysis` + `exp/component2-significance` | 统计显著性 |
| `exp/embedding-viz` + `exp/mech-crossds` | 可视化 + 跨数据集机制复证 |
| `exp/explainability` | 可解释性 |
| `exp/new-datasets` + `exp/new-datasets-2` + `exp/new-datasets-3` | 新数据集 |
| `exp/branches-doc` | 分支文档 |
| `exp/final-results-collector` | 结果收集器（收尾管线） |

## 快速定位

```
看最终方法 → master 分支 + 论文/研究总报告.md
看某个 baseline 怎么复现 → 对应 exp/<模型>-baseline 分支 + 复现脚本/
看某个组件的完整探索 → 对应组件分支
看失败原因 → 三、探索过的候选分支（每个都有负结果/中性的实测记录）
```

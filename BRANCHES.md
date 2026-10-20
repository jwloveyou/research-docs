# BRANCHES.md —— 各 git 分支说明（约 67 个，按时间倒序）

> 目的：让任何人（和未来的自己）快速看懂每个分支是什么、对应哪个实验（R几）、做了什么改动、结果（有效 / 负 / 中性）、是否准入融合、一句话结论。
>
> 图例：
> - ✅ **准入融合**：该组件进入最终 NoiseSGL 融合模型（§1 三/四组件之一）。
> - ❌ **负 / 未准入**：结果劣于控制组，或未达稳健增益，不进入融合。
> - ➖ **中性**：与控制组逐位接近（在 seed 噪声带内），不进入融合。
> - 📄 **纯文档 / 基建 / 脚本轮**：无模型改动或仅脚本/报告。
>
> 最终融合模型（R82 定稿）：`view_mix=nn`（嵌入噪声增广）+ `ssl_anneal=2, ssl_anneal_epochs=30`（固定窗口 SSL 退火）+ `embed_size=512`（容量缩放）+ **条件组件** `sem_emb=1, sem_lam=2.0`（冻结 LLM 语义注入，仅文本富集数据集 lastfm/ml-1m/ml-10m）。

| 分支名 | 轮次 | 做了什么改动 | 结果 | 是否准入 | 一句话结论 |
|---|---|---|---|---|---|
| `master` | R3/R4 | SimGCF 频域信号缩放器（可学习 freq alphas，`freq_scaler`/`freq_type`/`freq_mix`） | 负 | ❌ | 三数据集一致负（−1.8%/−8.5%/−5.2%）且慢收敛；「可学习结构改动有害」铁律起点 |
| `exp/freq-scope` | R4 | SimGCF 缩放器放置位置消融（`freq_on_main`/`freq_on_views`） | 负 | ❌ | 主/视图滤波器不一致=跌幅主因；慢收敛=主分支放置主因 |
| `exp/nt-ssm` | R5 | NT-SSM type-aware 邻居对升权（`nt_ssm`/`nt_ssm_alpha`，按流行度 BPR 逐对重加权） | 负 | ❌ | yelp −1.9%；对级权重分配轴无增益 |
| `exp/multiview-mix` | R6 | 视图组成消融（`view_mix`=nn/ne/nee：噪声 vs ED 结构视图混合） | 负 | ❌ | ED 结构视图严格劣于噪声视图，混入即稀释 |
| `exp/svd-global-view` | R7 | LightGCL 去噪全局视图（`view_mix` 增 g：rank-5 截断 SVD 全局视图） | 负 | ❌ | 全局视图经 InfoNCE 拉向低秩均值，过平滑/损个性化 |
| `exp/fcl-decorr` | R8 | RecDCL 特征维对比（`fcl_poly`/`fcl_bt`/`fcl_mom`，表征层去相关） | 中性偏负 | ❌ | 64 维骨干上 FCL 无正收益；RecDCL 增益依赖 2048 维大模型 |
| `exp/directau` | R9 | DirectAU 对齐+均匀性（`directau_align`/`directau_unif`/`directau_t`/`directau_k`） | 跨集不一致 | ❌ | yelp +0.6% 但 gowalla/amazon −0.6%/−2.1%，慢收敛正则器 |
| `exp/proto-inject` | R11 | NCL 式聚类原型注入（`proto_K`/`proto_lam`，k-means 凸混合进表征） | 中性偏负 | ❌ | 簇心平滑削弱个性化，与 R7 同构 |
| `exp/ssl-anneal` | R13 | SSL 权重余弦退火（`ssl_anneal`，λ 全程衰减至 0） | 正（NDCG） | ✅ 前身 | gowalla NDCG +1.6%（3-seed），R@20 不变；只救 NDCG |
| `exp/ssl-anneal-window` | R15 | 固定窗口 SSL 退火（`ssl_anneal_epochs`，W=30 轮衰减至 floor 后纯 BPR 精调） | **正** | ✅ **组件②** | gowalla R@20 +1.43%、NDCG +3.21%；首个同时改善 Recall+NDCG 的模块 |
| `exp/xsimgcl-noise` | R16 | XSimGCL 单次注入噪声 × ε 扫描（`noise_mode`） | 负 | ❌ | 单次注入劣于逐层；噪声注入位置/幅度轴封闭 |
| `exp/lightgcl-local-global` | R17 | 原版 LightGCL 局部-全局对比（`lightgcl`/`lightgcl_perlayer`） | 负 | ❌ | 三集一致负；全局低秩视图损个性化，重验证确认 |
| `exp/embed-capacity` | R18 | gowalla 容量 64→128（`embed_size` 本为 config） | **正** | ✅ **组件③ 起点** | gowalla 0.1809→0.1861，首次破 LightGCN 0.1830 |
| `exp/adagate` | R19 | AdaGate 自适应门控传播（`gate_mode`/`gate_proj`，GRU 式逐节点门控） | 强负 | ❌ | 可学习门控作弊关停传播→InfoNCE 平凡最小化；可学习结构有害铁律 |
| `exp/noise-curriculum` | R20 | 噪声课程（`noise_anneal`，ε 退火）+ L4 深度 + 机制脚本 | 中性/负 | ❌ | ε 退火中性；L4 快收敛低平台 |
| `exp/multi-neg-bpr` | R21 | 多负例 BPR（`num_negatives`，K∈{1,3,5}） | 中性 | ❌ | InfoNCE 已全图负例，BPR 多负例冗余 |
| `exp/new-datasets` | R22 | 新增 lastfm-2k + ml-1m 预处理（`prepare_new_datasets.py`） | 部分泛化 | 📄 | lastfm +4.1% 胜、ml-1m −3.3% 负（密度边界首现） |
| `exp/l2norm-scoring` | R23 | L2 归一化（余弦）评分（`l2_norm`）+ embed256 延伸 | 负/正 | ❌（l2n）| 余弦化抹掉范数=流行度信号，三集 −11%~−28% |
| `exp/embed256-final` | R24 | embed256 三 seed 定稿 | **正** | ✅ 组件③ 升级 | 128→256 三集全胜（amazon +9.4% 最猛） |
| `exp/embed-dropout` | R25 | 主分支嵌入 dropout（`dropout`） | 中性 | ❌ | 噪声 SSL 已提供等价正则，dropout 冗余 |
| `exp/new-datasets-2` | R26 | 新增 douban-book/ifashion/tmall + 修复采样 bug | 部分泛化 | 📄 | douban +4.6% 胜、ifashion 容量过拟合、tmall 早期胜 |
| `exp/new-datasets-3` | R27 | 新增 yelp_full/amazon-beauty/amazon-kindle | 部分泛化 | 📄 | kindle +148%、beauty +41.7%、yelp_full 容量过拟合 |
| `exp/noise-degree` | R28 | 密度感知噪声强度（`noise_deg_gamma`，head-boost） | 中性 | ❌ | 度缩放噪声 30ep/100ep 均中性，节点轴关闭 |
| `exp/sota-baselines` | R29 | 同协议 SOTA baseline 复现（5 baseline）+ 代码交叉审查 | 基建 | 📄 | 旧 embed256 容量匹配版（后 R79 判作废） |
| `exp/temp-sched` | R30/R30b/R31 | InfoNCE 温度调度（`ssl_temp_anneal`） | 负 | ❌ | τ=0.2 已近最优，降 τ 锐化有害，升 τ 软化中性 |
| `exp/generalization-capacity` | R32 | 泛化边界精化 + ifashion/yelp_full 降容修复排队 | 分析 | 📄 | 边界由「图密度」精化为「容量利用率」 |
| `exp/layer-noise` | R33 | 深度感知逐层噪声调度（`noise_layer_gamma`） | 中性 | ❌ | 深层更强扰动无增益，深度轴关闭 |
| `exp/initial-residual` | R34 | GCNII/APPNP 静态初始残差（`res_alpha`） | 负 | ❌ | 残差重注入稀释噪声释秩收益，结构轴关闭 |
| `exp/spec-bal` | R35 | DirectSpec 式谱平衡损失（`spec_bal`/`spec_bal_center`） | 强负 | ❌ | 损失层强制谱平坦与 BPR 内积冲突，−8.8%~−17.3% |
| `exp/lr-schedule` | R37 | 学习率调度（`lr_anneal`，linear/cosine） | 待/中性 | ❌ | 优化器步长轴，先验低 |
| `exp/weight-decay` | R38 | 正则化轴（`weight_decay` + `reg` 扫描） | 待/中性 | ❌ | 优化器正则化轴，先验低 |
| `exp/layer-agg` | R39 | 静态非均匀层聚合权重（`agg_weights`/`agg_alpha`） | 待/中性 | ❌ | 层聚合轴，先验低 |
| `exp/swa` | R40 | 随机权重平均 SWA（`swa_start`/`swa_n`） | 待/中性 | ❌ | 权重轨迹轴，先验低 |
| `exp/bpr-margin` | R41 | BPR 排序 margin（`bpr_margin`） | 待/中性 | ❌ | margin 轴，先验低 |
| `exp/dense-boundary` | R42 | 稠密数据集（jester/ml-10m）+ 文献核查 | 部分 | 📄 | jester 中性、ml-10m 在途，密度谱补端点 |
| `exp/hccf-baseline` | R43 | HCCF No.1 baseline（`hccf`，超图 HHᵀE） | 负（官方 32 维） | ❌ baseline | 可学习超图结构在本骨干低效，铁律反事实 |
| `exp/dccf-baseline` | R44 | DCCF No.1 baseline（`dccf`，意图解耦+自适应子图） | 负 | ❌ baseline | yelp −51%，可学习结构有害第三次复证 |
| `exp/explainability` | R45 | 图 2-hop 协同路径解释器（`util/explain.py`） | **正（交付）** | ✅ 论文模块 | 覆盖率 94.8-100%，GCN 层-2 扩散的白盒分解 |
| `exp/adagcl-baseline` | R46 | AdaGCL No.1 baseline（`adagcl`，VGAE+去噪生成器） | 待/慢收敛 | ❌ baseline | 可训练视图生成器，先验低 |
| `exp/semantic-features` | R47/R48 | LLM 语义特征构建 + 冻结语义注入开关（`sem_emb`/`sem_lam`） | **正** | ✅ **组件④** | 信息注入轴首个干净正增益 |
| `exp/semantic-analysis` | R50 | 语义特征谱分析（`analyze_semantic_features.py`） | 分析 | 📄 | 语义先验有效秩 44.4-71.8% |
| `exp/semantic-coldstart` | R52/R53 | 语义冷启动先验证据 + lastfm 3-seed 定案 | **正** | ✅ 组件④ 定稿 | lastfm sem_lam=1.0 +0.82% 3-seed 零交叉 |
| `exp/sensitivity-embed256` | R51 | embed256 ε×退火窗口敏感性扫描 | 待/中性 | 📄 | 任务#2 敏感性缺口（后随 embed512 升级更新） |
| `exp/anneal-ablation` | R54/R55 | 组件②「无退火」3-seed + 机制脚本修复 | 分析 | 📄 | 修 5 元组解包 bug + 跨集机制脚本 |
| `exp/significance-analysis` | R56 | 统计显著性分析（`analyze_significance.py`） | 分析 | 📄 | 配对 t+bootstrap CI+零交叉 |
| `exp/mech-crossds` | R57 | E11 机制跨集复证（`analyze_mechanism.py` 修 dataset 接线 bug） | 分析 | 📄 | lastfm 2.84× 释秩 + 1.74× 抗过平滑 |
| `exp/item-noise-boost` | R58 | 物品侧噪声增强（`noise_item_boost`） | 中性 | ❌ | 噪声 side 轴关闭 |
| `exp/embed512-capacity` | R59 | embed512 容量饱和点测试 | **正** | ✅ 组件③ 升级 | amazon +6.9% 决定性，容量未饱和 |
| `exp/init-scale` | R60 | 嵌入初始化缩放（`init_scale`） | 待/中性 | ❌ | 初始化轴，先验低 |
| `exp/convergence-analysis` | R61 | 收敛性分析（`analyze_convergence.py` + SVG 绘图） | 分析 | 📄 | win30 退火解耦快收敛与过拟合 |
| `exp/embedding-viz` | R62/R63/R67 | 可视化底层证据（t-SNE/谱能量/有效秩/过平滑 SVG） | 分析 | 📄 | 机制证据从数字升级为成图 |
| `exp/density-mechanism` | R64 | 密度边界机制修正 + ml-1m ε×ssl_reg 决定性扫描 | 分析 | 📄 | 推翻旧「无过平滑可抗」说，改为 InfoNCE 均匀性说 |
| `exp/ngcf-baseline` | R65 | NGCF 经典 baseline（`ngcf`） | 负 | ❌ baseline | 可学习变换铁律第 4 反事实 |
| `exp/density-mechanism-analysis` | R68 | 密度机制定量对照（`analyze_density_mechanism.py`） | 分析 | 📄 | 三集机制定量表 |
| `exp/ultragcn-baseline` | R66/R67 | UltraGCN 经典 baseline（`ultragcn`） | 协议错配→修复 | ❌ baseline | R=1 误用，R73 修 R=300 |
| `exp/ultragcn-r300-fix` | R73 | UltraGCN R=300 协议修复（`_ultragcn_batch_loop` 支持 R 负例） | 基建 | 📄 | 避免「假赢 baseline」 |
| `exp/dgcf-baseline` | R69 | DGCF 经典 baseline（`dgcf`，意图解耦） | 待 | ❌ baseline | 可学习意图原型，先验低 |
| `exp/ncl-baseline` | R70 | NCL 经典 baseline（`ncl`，结构+原型双对比） | 待 | ❌ baseline | 全局/簇级信号，先验低 |
| `exp/directau-baseline` | R71 | DirectAU standalone baseline（`directau`） | 待 | ❌ baseline | 无传播纯嵌入法反事实 |
| `exp/component2-significance` | R72 | 组件②「有/无」3-seed 显著性 | **正** | ✅ 消融闭合 | 四组件消融表全 3-seed 显著 |
| `exp/semantic-ml10m` | R74 | 语义组件④扩展到 ml-10m（第 3 文本数据集） | 分析 | 📄 | ml-10m 语义特征全覆 + 冷启动先验 |
| `exp/density-flip` | R75 | 密度翻转点定量 + 密度感知 ssl_reg 处方（`analyze_density_flip.py`） | 分析 | 📄 | 翻转区间 (2.30%, 5.07%)，ρ_flip≈3.69% |
| `exp/semantic-semlam-scan` | R76/R76b | sem_lam 敏感性曲线 + sem_lam=2.0 3-seed | **正** | ✅ 组件④ 升级 | 最优 lam=2.0，增益 +0.82%→+1.43% |
| `exp/recdcl-baseline` | R77/R78 | RecDCL baseline（配置式：`view_mix=ee` + FCL 官方系数） | 待 | ❌ baseline | 官方 2048 维，容量匹配 256 会低估 |
| `exp/baseline-official-hyperparams` | R79/R80 | 协议可比性审查 + 官方超参 baseline 重跑 | 基建 | 📄 | 停 4 错配训练 + 13 baseline 官方超参重跑 |
| `exp/embed512-3seed` | R81/R82 | embed512 三 seed 定案 + R82 失败任务重跑 | **正** | ✅ 组件③ 定稿 | 三集零交叉，embed_size 256→512 |

## 统计

- **准入融合的分支**：`exp/ssl-anneal-window`（组件②）、`exp/embed-capacity`/`exp/embed256-final`/`exp/embed512-capacity`/`exp/embed512-3seed`（组件③ 容量轴）、`exp/semantic-features`/`exp/semantic-coldstart`/`exp/semantic-semlam-scan`（组件④ 语义轴）、`exp/explainability`（可解释性模块，论文交付非消融组件）。
- **关键负结果（构成「static wins」铁律与「可学习结构有害」反事实）**：`exp/freq-scope`/`exp/nt-ssm`/`exp/multiview-mix`/`exp/svd-global-view`/`exp/fcl-decorr`/`exp/directau`/`exp/proto-inject`/`exp/xsimgcl-noise`/`exp/lightgcl-local-global`/`exp/adagate`/`exp/noise-curriculum`/`exp/multi-neg-bpr`/`exp/l2norm-scoring`/`exp/embed-dropout`/`exp/noise-degree`/`exp/temp-sched`/`exp/layer-noise`/`exp/initial-residual`/`exp/spec-bal`/`exp/item-noise-boost` 等。
- **忠实实现但（因 scnumath 被封 / 预算不足）终值不可得的 baseline 分支**：`exp/hccf-baseline`/`exp/dccf-baseline`/`exp/adagcl-baseline`/`exp/ngcf-baseline`/`exp/ultragcn-baseline`(+`-r300-fix`)/`exp/dgcf-baseline`/`exp/ncl-baseline`/`exp/directau-baseline`/`exp/recdcl-baseline`（官方超参重跑丢失，其旧 embed256 值作废，见总报告 §2.3）。

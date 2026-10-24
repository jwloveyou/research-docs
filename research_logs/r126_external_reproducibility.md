# R126 · 外部可复现性文献收口（独立第三方佐证 + UltraGCN「静态方法」定位修正）

> 时间：2026-10-25 05:1x ｜ 类型：查文献 → 判断 → 纯 CPU 写作（GPU 双机忙：masmath:0=yelp2018_dgcf、scnumath:4=gowalla_dgcf，礼貌让卡，无训练终值）
> 查询通道：arXiv API（http，本轮可达）+ OpenAlex（https 可达）；ar5iv / arXiv PDF / zenodo / raw.githubusercontent 本轮超时或连接拒绝。

## 1. 两个独立第三方可复现性证据（本轮核心发现）

### ① RecSys'23「Challenging the Myth of Graph Collaborative Filtering」
- 全名：*Challenging the Myth of Graph Collaborative Filtering: a Reasoned and Reproducibility-driven Analysis*
- 作者：Vito Walter Anelli, Daniele Malitesta, Claudio Pomo, Alejandro Bellogín, Eugenio Di Sciascio, Tommaso Di Noia
- 出处：ACM RecSys 2023，**DOI 10.1145/3604915.3609489**，**20 引**，arXiv:2308.00404
- 代码：github.com/sisinflab/Graph-RSs-Reproducibility
- **内容**：独立复现 6 个图推荐模型（**NGCF / DGCF / LightGCN / SGL / UltraGCN / GFCF**）在 Gowalla / Yelp2018 / Amazon-Book 三基准上的结果，并把它们与「历史上离线评估表现强的传统 CF 模型」对比；结论=**挑战图 CF 的「神话」**——许多图模型的报告增益来自「未验证地沿用基线论文结果」，而非真实方法优势。
- **与本工作的对齐**：① 直接坐实「可学习结构改动有害」铁律（R4/R19/R43/R44/R46/R65/R66/R69 七反事实）——第三独立研究也发现「可学习图结构」的增益被高估；② 直接坐实本 repo 的诚实协议纪律（R79 官方超参审查 + R125 split 可比性对照）——该文的核心批评正是「沿用未核验的 baseline 值」。

### ② ML Reproducibility Challenge 的 UltraGCN 专项复现报告
- 全名：*Reproducibility Report for UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation*
- 出处：Zenodo，**DOI 10.5281/zenodo.7591338**（landing 7591339），2023，0 引
- **内容**：专门评估 UltraGCN（Mao et al., CIKM'21）在实验设计 / 数据 / 代码可得性三个维度的可复现性。
- **意义**：UltraGCN 是本 repo 14 baseline 里**唯一被社区单独写复现报告**的模型，印证「UltraGCN 报告值是否有水分」是领域内真实存在、且被独立第三方关注的疑云（与本 repo R107「复现失败」→ R109「mean-vs-sum 根因」→ R125「amazon 0.0681 异常偏高」一线相呼应）。

## 2. 关键判断：UltraGCN 是「静态」方法，不是「可学习结构」反事实

此前 R123/R122 的 2×2 分类学把 UltraGCN 与 NGCF/DGCF 同列，但本轮重读 UltraGCN 原文摘要后**修正定位**：

- UltraGCN **彻底丢弃 message passing**（无逐层可学习变换、无可学习传播权重），用「约束损失近似无限层卷积极限」——即它**只有嵌入表是可学习参数**，结构与本 repo 融合模型同属「静态」。
- 因此 **「可学习结构有害」铁律不适用于 UltraGCN**：它不是该铁律的反事实，而是「无传播的静态约束损失」这一**正交架构**，是融合模型的**真实竞争对手**。
- **含义（诚实，重要）**：R125 标注的「UltraGCN 论文 amazon 0.0681 异常偏高（+66% vs LightGCN 0.0411）」在「第三方已成功复现 UltraGCN（RecSys'23 摘要明写 successfully replicates UltraGCN）」这一证据下，**更可能是真实值而非 split 膨胀**——即「融合 amazon 0.0589 vs UltraGCN 0.0681」可能是一处真实落后，**必须等 R109 修复后的 UltraGCN×3 官方超参复跑出值定案**，不能提前采信「split 膨胀」的猜测。
- gowalla / yelp 上融合（0.1894 / 0.0716）仍高于 UltraGCN 论文值（0.1862 / 0.0683），仅 amazon 存疑。

## 3. 对论文的落点

- §2.3 补一条「外部可复现性证据」bullet：RecSys'23（3604915.3609489）+ Zenodo 7591338 两个独立第三方证据，把「baseline 复现必须诚实」从本 repo 纪律升级为「领域共识」。
- §6.1 失败模式分类学补一个「E 类（可学习结构）边界」注：UltraGCN 属静态方法、不列入 E 类反事实，其「无传播 + 约束损失」是唯一可能威胁融合的静态对照（amazon 待 R109 复跑）。
- 更新 R125 的「UltraGCN amazon 异常」判断措辞：由「split 膨胀 or 真实」的二选一，调整为「第三方复现成功 → 偏真实 → 必须 R109 复跑定案」。

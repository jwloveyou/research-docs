# 基于图神经网络的稀疏协同过滤推荐方法

> 毕业论文项目 · 可复现代码与完整实验历史

## 项目简介

在 LightGCN 骨干上叠加 **3 个静态无参组件 + 1 个条件语义组件**，三核心数据集 Recall@20 相对官方超参 baseline 提升 **+3.5% ~ +43.3%**，13 数据集泛化验证稀疏图稳健受益，机制为"噪声抗过平滑 + 释放嵌入秩"，适用边界由图密度（ρ≈3.42%）定量刻画，零额外参数、27.6× 收敛加速。

## 融合模型（4 组件）

| 组件 | 做法 | 配置 |
|---|---|---|
| ① 嵌入噪声增广 | 逐层单位球方向扰动，双视图对比 | ε=0.2 |
| ② SSL 固定窗口退火 | SSL 权重 cosine 衰减到 0，后半程纯 BPR | W=30, λmax=0.1 |
| ③ 容量缩放 | 嵌入维度 64→512 | d=512 |
| ④ 冻结语义注入 | LLM 编码物品文本，冻结加权注入 | λsem=2.0（仅文本富集集） |

## 目录结构

```
research-docs/
├── README.md              ← 本文件（项目总览）
├── 论文/                   ← 论文文档
│   ├── 研究总报告.md        ← 最终论文（方法/结果/机制/边界）
│   ├── 研究任务书.md        ← 研究任务与指令
│   └── 研究进度报告.md      ← R1-R126 完整实验日志
├── 代码/
│   ├── model/general_recommender/NoiseSGL.py  ← 核心模型（融合+14 baseline，218KB）
│   ├── main.py            ← 入口
│   └── conf/NoiseSGL.ini  ← 配置
├── 复现脚本/               ← 14 baseline 官方超参复现脚本
├── 数据集/                 ← 13 数据集
└── BRANCHES.md            ← 分支说明（每个分支对应哪个模型/实验）
```

## 14 个 baseline（全部同协议官方超参复现）

**No.1 三强**：HCCF（SIGIR'22）、DCCF（SIGIR'23）、AdaGCL（KDD'23）
**轻量**：LightGCN（SIGIR'20）、SGL（SIGIR'21）、SimGCL（SIGIR'22）、XSimGCL（TOIS'23）
**经典**：NGCF（SIGIR'19）、LightGCL（ICLR'23）、UltraGCN（CIKM'21）、DGCF（SIGIR'20）、NCL（WWW'22）、DirectAU（KDD'22）、RecDCL（WWW'24）

## 复现方法

```bash
# 环境：conda create -n sgl python=3.9 && conda activate sgl && pip install -r requirements.txt
# 融合模型（三数据集）
python main.py --recommender=NoiseSGL --dataset=gowalla --embed_size=512 \
  --view_mix=nn --noise_eps=0.2 --ssl_anneal=2 --ssl_anneal_epochs=30
# baseline（官方超参）
bash research_logs/run_baselines_official.sh <gpu_id> <baseline_name>
```

## 分支命名规则

- `master` = 最终融合模型（3 通用 + 1 条件组件）
- 每个 baseline / 组件 / 实验一个分支，分支名即模型或实验名（详见 BRANCHES.md）

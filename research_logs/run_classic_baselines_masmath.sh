#!/usr/bin/env bash
# =============================================================================
# R102 · run_classic_baselines_masmath.sh —— 经典 baseline（LightGCL/NGCF/
#        UltraGCN/DGCF/NCL/DirectAU/RecDCL）官方超参在 masmath 单卡 GPU0 串行
#        复跑（scnumath 被封后补 task#10「经典模型大量复现」的最后缺口）。
#
# 背景：
#   R79 审查发现 13 个 baseline 全部被「容量匹配 embed256」扭曲；官方超参重跑
#   挂在 scnumath GPU7 后随封禁丢失（R82）。masmath 已收 4 轻量（LightGCN/SGL/
#   SimGCL/XSimGCL，R80/R82）+ No.1 三强（HCCF/DCCF/AdaGCL，R95 排队）；本脚本
#   补齐剩余的 7 个经典 baseline。
#
# 官方超参（逐项核官方仓库，与 run_baselines_official.sh 一致，R79/R100 已复核）：
#   LightGCL ICLR'23  HKUDS/LightGCL  d=64 gnn_layer=2 batch=256 lr=1e-3
#                     lambda1=0.2(=ssl_reg) q=5(=gl_rank) temp=0.2 lambda2=1e-7(=reg)
#   NGCF     SIGIR'19  xiangwang1223  embed=64 layer=3 batch=1024 reg=1e-5 drop=0.1
#                     gowalla/yelp lr=1e-4, amazon lr=5e-4
#   UltraGCN CIKM'21   论文实验设置  embed=64 R=300 lr=1e-4 init=N(0,1e-4)(normal,std0.01)
#   DGCF     SIGIR'20  xiangwang1223  embed=64 layer=3 K=4 batch=1024 lr=1e-3 reg=1e-4
#   NCL      WWW'22    RUCAIBox/NCL   embed=64 layer=3 batch=4096 ssl_reg=1e-6
#                     proto_reg=5e-8~1e-7 temp=0.05 alpha/clusters per-dataset
#                     lr gowalla=2e-3 else 1e-3
#   DirectAU KDD'22    论文/官方      embed=64 gamma=1 t=2 lr=1e-3 batch=1024（无 GCN）
#   RecDCL   WWW'24    THUDM/RecDCL  embedding=2048 n_layers=2 + FCL 官方系数
#                     (bt=1/bt_off=0.01/poly=0.2/a=1/degree=4/polyc=1e-7/mom=10/momentum=0.3)
#                     + BCL=SGL 式边丢弃(view_mix=ee)
#
# 协议与融合模型完全一致（task#14 已核）：同数据切分 / 全量排序 R@20·NDCG@20 /
# best-over-epochs / seed=42。baseline 各自用官方维度/层数/batch，融合用 embed512 ——
# 各自发挥最好水平（task#13）。
#
# 串行、独占 GPU0（R82 教训：并行共驻致 OOM + 磁盘配额失败）。
# RecDCL（embed2048）排最后：最重、OOM 风险最高，排尾不阻塞前 6 个轻量经典基线。
# 用法：bash research_logs/run_classic_baselines_masmath.sh
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L="research_logs/r102_classic_baselines_progress.txt"
[ -f "$L" ] || echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  # run <tag> <ds> <eps> <extra...>
  tag=$1; ds=$2; eps=$3; shift 3
  log="research_logs/r102_${ds}_${tag}.log"
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --use_mlc=0 --ssl_anneal=0 --ssl_anneal_epochs=0 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 "$@" > "$log" 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${ds}_${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

# ---- LightGCL（局部 vs SVD 全局低秩视图对比，ICLR'23）----
lightgcl () {
  run lightgcl gowalla     150 --embed_size=64 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-7 --lightgcl=1 --lightgcl_perlayer=1 --gl_rank=5 --ssl_reg=0.2 --ssl_temp=0.2 --ssl_ratio=0.1
  run lightgcl yelp2018     50 --embed_size=64 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-7 --lightgcl=1 --lightgcl_perlayer=1 --gl_rank=5 --ssl_reg=0.2 --ssl_temp=0.2 --ssl_ratio=0.1
  run lightgcl amazon-book  30 --embed_size=64 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-7 --lightgcl=1 --lightgcl_perlayer=1 --gl_rank=5 --ssl_reg=0.2 --ssl_temp=0.2 --ssl_ratio=0.1
}

# ---- NGCF（逐层可学习变换 W1/W2 + LeakyReLU，SIGIR'19）----
ngcf () {
  run ngcf gowalla     400 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.0001 --reg=1e-5 --ngcf=1 --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1
  run ngcf yelp2018    400 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.0001 --reg=1e-5 --ngcf=1 --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1
  run ngcf amazon-book 200 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.0005 --reg=1e-5 --ngcf=1 --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1
}

# ---- UltraGCN（无传播约束损失逼近无限层收敛态，CIKM'21）----
ultragcn () {
  run ultragcn gowalla     150 --embed_size=64 --lr=0.0001 --reg=1e-4 --param_init=normal --stddev=0.01 --ultragcn=1 --ultragcn_lambda=1.0 --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --batch_size=1024
  run ultragcn yelp2018    100 --embed_size=64 --lr=0.0001 --reg=1e-4 --param_init=normal --stddev=0.01 --ultragcn=1 --ultragcn_lambda=1.0 --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --batch_size=1024
  run ultragcn amazon-book 100 --embed_size=64 --lr=0.0001 --reg=1e-4 --param_init=normal --stddev=0.01 --ultragcn=1 --ultragcn_lambda=1.0 --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --batch_size=1024
}

# ---- DGCF（K=4 可学习意图原型 + 逐因子加权传播，SIGIR'20）----
dgcf () {
  run dgcf gowalla     200 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.001 --reg=1e-4 --dgcf=1 --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01
  run dgcf yelp2018    100 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.001 --reg=1e-4 --dgcf=1 --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01
  run dgcf amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.001 --reg=1e-4 --dgcf=1 --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01
}

# ---- NCL（结构对比 + 原型对比双对比，WWW'22）----
ncl () {
  run ncl gowalla     200 --embed_size=64 --n_layers=3 --batch_size=4096 --lr=0.002 --reg=1e-4 --ncl=1 --ncl_ssl_reg=1e-6 --ncl_proto_reg=5e-8 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=0.5 --ncl_num_clusters=10 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 --ncl_kmeans_seed=0
  run ncl yelp2018    100 --embed_size=64 --n_layers=3 --batch_size=4096 --lr=0.001 --reg=1e-4 --ncl=1 --ncl_ssl_reg=1e-6 --ncl_proto_reg=1e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=1.5 --ncl_num_clusters=2000 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 --ncl_kmeans_seed=0
  run ncl amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=4096 --lr=0.001 --reg=1e-6 --ncl=1 --ncl_ssl_reg=1e-6 --ncl_proto_reg=1e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=0.8 --ncl_num_clusters=2000 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 --ncl_kmeans_seed=0
}

# ---- DirectAU（无传播，纯嵌入对齐 + 均匀性，KDD'22）----
directau () {
  run directau gowalla     300 --embed_size=64 --lr=0.001 --reg=1e-4 --batch_size=1024 --directau=1 --directau_gamma=1.0 --directau_t=2.0
  run directau yelp2018    100 --embed_size=64 --lr=0.001 --reg=1e-4 --batch_size=1024 --directau=1 --directau_gamma=1.0 --directau_t=2.0
  run directau amazon-book 100 --embed_size=64 --lr=0.001 --reg=1e-4 --batch_size=1024 --directau=1 --directau_gamma=1.0 --directau_t=2.0
}

# ---- RecDCL（BCL=SGL 边丢弃 + FCL 特征维对比，WWW'24；官方 embed2048/2层）----
# 诚实注：官方 RecDCL 用 embedding=2048（R79-A 表）；R77 旧跑用 embed256 属「容量匹配
# 低估」，本脚本改为官方 2048 + n_layers=2。若 embed2048 在 amazon-book 上 OOM，run()
# 会记录 rc!=0 并继续，finalizer 将如实标注该 run FAILED、不造值。
recdcl () {
  run recdcl gowalla     150 --embed_size=2048 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=1e-4 --view_mix=ee --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --fcl_bt=1.0 --fcl_poly=0.2 --fcl_mom=10.0 --fcl_momentum=0.3 --fcl_off=0.0
  run recdcl yelp2018     50 --embed_size=2048 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=1e-4 --view_mix=ee --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --fcl_bt=1.0 --fcl_poly=0.2 --fcl_mom=10.0 --fcl_momentum=0.3 --fcl_off=0.0
  run recdcl amazon-book  30 --embed_size=2048 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=1e-4 --view_mix=ee --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --fcl_bt=1.0 --fcl_poly=0.2 --fcl_mom=10.0 --fcl_momentum=0.3 --fcl_off=0.0
}

lightgcl
ngcf
ultragcn
dgcf
ncl
directau
recdcl

echo "CLASSIC_BASELINES_ALLDONE $(date)" >> "$L"

# 简易汇总（剥时间戳前缀取 R@20/NDCG，与 collect_final_results.extract_best 同法）
echo "=== summary (best R@20 / NDCG@20) ===" >> "$L"
for f in research_logs/r102_*.log; do
  tag=$(basename "$f" .log | sed 's/r102_//')
  vals=$(grep -A2 "best_result" "$f" | tail -1 | sed 's/^[0-9-]* [0-9:.]*: *//' | awk '{printf "%s %s", $2, $3}')
  echo "$tag  $vals" >> "$L"
done
echo "(对照：融合 embed512 seed42 = gowalla 0.1894/0.1596, yelp 0.0716/0.0587, amazon 0.0589/0.0455)" >> "$L"

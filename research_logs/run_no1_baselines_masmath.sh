#!/usr/bin/env bash
# =============================================================================
# R95 · run_no1_baselines_masmath.sh —— No.1 baseline（HCCF/DCCF/AdaGCL）官方超参
#       在 masmath 单卡 GPU0 串行复跑（scnumath 被封后补 task#8「No.1 同台对比」）。
#
# 官方超参（逐项核官方仓库，与 run_baselines_official.sh 一致）：
#   HCCF  SIGIR'22  akaxlh/HCCF    latdim=32 gnn_layer=2 batch=256 lr=1e-3
#                 gowalla/yelp: reg=1e-5 ssl_reg=1e-4 temp=1.0 hyperNum=128 keepRate=0.5
#                 amazon:        reg=1e-2 ssl_reg=1e-7 temp=0.1
#   DCCF  SIGIR'23  HKUDS/DCCF     embed=32 n_layers=2 batch=1024 lr=1e-3
#                 n_intents=128 temp=1.0 emb_reg=2.5e-5 cen_reg=5e-3 ssl_reg=0.1（无全局 decay）
#   AdaGCL KDD'23  HKUDS/AdaGCL    latdim=32 gnn_layer=2 batch=4096 lr=1e-3 reg=1e-5
#                 ssl_reg=0.1 ib_reg=0.1 temp=0.5 lambda0=1e-4 gamma=-0.45 zeta=1.05
#                 init_temp=2.0 temp_decay=0.98 eps=1e-3
#
# 协议与融合模型完全一致（task#14 已核）：同数据切分 / 全量排序 R@20·NDCG@20 /
# best-over-epochs / seed=42。baseline 各自用官方维度（32），融合用 512 —— 各自发挥
# 最好水平（task#13）。
#
# 串行、独占 GPU0（R82 教训：并行共驻致 OOM + 磁盘配额失败）。
# 用法：bash research_logs/run_no1_baselines_masmath.sh
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L="research_logs/r95_no1_baselines_progress.txt"
[ -f "$L" ] || echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  # run <tag> <ds> <eps> <extra...>
  tag=$1; ds=$2; eps=$3; shift 3
  log="research_logs/r95_${ds}_${tag}.log"
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

# ---- HCCF（超图全局视图 + 可学习超图依赖矩阵，No.1 之首）----
hccf () {
  run hccf gowalla     100 --embed_size=32 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-5 --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-4 --hccf_temp=1.0
  run hccf yelp2018    100 --embed_size=32 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-5 --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-4 --hccf_temp=1.0
  run hccf amazon-book 100 --embed_size=32 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-2 --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-7 --hccf_temp=0.1
}

# ---- DCCF（意图解耦 + 可学习自适应子图）----
dccf () {
  run dccf gowalla     150 --embed_size=32 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=0.0 --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1
  run dccf yelp2018    100 --embed_size=32 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=0.0 --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1
  run dccf amazon-book 100 --embed_size=32 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=0.0 --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1
}

# ---- AdaGCL（可训练视图生成器：VGAE + 去噪网络，三优化器）----
adagcl () {
  run adagcl gowalla     200 --embed_size=32 --n_layers=2 --batch_size=4096 --lr=0.001 --reg=1e-5 --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3
  run adagcl yelp2018    100 --embed_size=32 --n_layers=2 --batch_size=4096 --lr=0.001 --reg=1e-5 --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3
  run adagcl amazon-book 100 --embed_size=32 --n_layers=2 --batch_size=4096 --lr=0.001 --reg=1e-5 --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3
}

hccf
dccf
adagcl

echo "NO1_BASELINES_ALLDONE $(date)" >> "$L"

# 简易汇总（剥时间戳前缀取 R@20/NDCG，与 collect_final_results.extract_best 同法）
echo "=== summary (best R@20 / NDCG@20) ===" >> "$L"
for f in research_logs/r95_*.log; do
  tag=$(basename "$f" .log | sed 's/r95_//')
  vals=$(grep -A2 "best_result" "$f" | tail -1 | sed 's/^[0-9-]* [0-9:.]*: *//' | awk '{printf "%s %s", $2, $3}')
  echo "$tag  $vals" >> "$L"
done
echo "(对照：融合 embed512 seed42 = gowalla 0.1894/0.1596, yelp 0.0716/0.0587, amazon 0.0589/0.0455)" >> "$L"

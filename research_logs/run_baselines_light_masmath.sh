#!/usr/bin/env bash
# =============================================================================
# R80 · run_baselines_light_masmath.sh
# 在 masmath GPU0 并行跑「主对比关键」的 4 个轻量 baseline（官方超参），
# 与 scnumath GPU7 的「No.1/经典 baseline 链」并行，避免官方超参重跑串行过慢。
#
#   lightgcn : 官方 embed64/3层/batch2048/lr1e-3/decay1e-4（主 baseline 验证，R79 已证
#              数据切分与 LightGCN 官方逐位一致，论文主值引用官方 0.0649/0.0411/0.1830）
#   sgl      : 官方 embed64/3层/batch2048/ED/ssl_reg0.1/ratio0.1/temp0.2
#   simgcl   : 官方 embed64/3层/batch2048/cl_rate0.1/eps0.1/tau0.2（逐层噪声）
#   xsimgcl  : 官方 embed64/3层/eps0.2/cl0.1/tau0.2（单次噪声）
#
# 评测协议与融合模型完全一致（全量排序 + NDCG@20 best-over-epochs，task#14 已核）。
# 结果写 research_logs/r80_light_baselines_masmath_progress.txt 与 r80_<ds>_<tag>.log
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd "$(dirname "$0")/.."
GPU=0
L="research_logs/r80_light_baselines_masmath_progress.txt"
[ -f "$L" ] || echo "BEGIN $(date)" > "$L"

run () {
  tag=$1; ds=$2; eps=$3; shift 3
  log="research_logs/r80_${ds}_${tag}.log"
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

# ---- LightGCN (官方 embed64/3层/batch2048/lr1e-3/decay1e-4; 验证轮，主值用论文) ----
run lightgcn gowalla     300 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0
run lightgcn yelp2018    100 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0
run lightgcn amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0

# ---- SGL (官方 embed64/3层/batch2048/ED/ssl_reg0.1/temp0.2/ratio0.1) ----
run sgl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=ee --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --noise_eps=0.2
run sgl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=ee --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --noise_eps=0.2
run sgl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=ee --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --noise_eps=0.2

# ---- SimGCL (官方 embed64/3层/batch2048/cl_rate0.1/eps0.1/tau0.2) ----
run simgcl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run simgcl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run simgcl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2

# ---- XSimGCL (官方 embed64/3层/eps0.2/cl0.1/tau0.2) ----
run xsimgcl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run xsimgcl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run xsimgcl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2

echo "R80_LIGHT_BASELINES_ALLDONE $(date)" >> "$L"

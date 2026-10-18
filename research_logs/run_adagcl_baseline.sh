#!/bin/bash
# AdaGCL baseline (task #8, third & last No.1): faithful Adaptive Graph Contrastive Learning
# for Recommendation (KDD'23, arXiv:2305.10837, official repo HKUDS/AdaGCL Model.py),
# implemented in NoiseSGL.py under the adagcl=1 switch (two trainable view generators:
# VGAE + graph-denoising net, IM InfoNCE + information-bottleneck loss + mean-BPR, three
# optimisers).
#
# Protocol (same as HCCF R43 / DCCF R44): LightGCN backbone, n_layers=3 (official 2, kept
# 3 for the same-protocol backbone depth used by all our No.1 baselines), seed=42,
# best-over-epochs R@20/NDCG@20 on test, lr=1e-3, batch=1024 (framework protocol; official
# 4096), stop_cnt=1000. AdaGCL-specific hyperparams from the official code: temp=0.5,
# ssl_reg=0.1, ib_reg=0.1, beta=0.1, lambda0=1e-4, gamma=-0.45, zeta=1.05, init_temp=2.0,
# temp_decay=0.98, reg=1e-5, denoise Adam eps=1e-3.
#
# *** embed_size=32 (AdaGCL OFFICIAL latdim) *** -- documented protocol deviation: AdaGCL's
# two trainable view generators (VGAE edge-probability decoder + per-layer denoising
# attention MLPs) run O(E*d) edge-wise MLPs, so the capacity-matched embed256 is ~64x the
# MLP cost and infeasible on this infrastructure (and 32 is the paper's own reported
# optimum). This is an honest note, not a hidden advantage: 32-dim is AdaGCL's best-case
# setting. A capacity-matched embed256 run may be attempted later if GPU time permits.
#
# Epoch budget (cost-reduced vs HCCF/DCCF because AdaGCL is ~3-5x DCCF's cost/epoch, its
# generators recompute O(E*d) per batch): gowalla 100ep / yelp 50ep / amazon 30ep.
#
# Usage: bash run_adagcl_baseline.sh <gpu_id>
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=$1
L=research_logs/r_adagcl_progress.txt
[ -f $L ] || echo "BEGIN $(date)" > $L

run () {
  tag=$1; ds=$2; eps=$3
  log=research_logs/r_adagcl_${ds}.log
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=32 --use_mlc=0 --gpu_id=$GPU \
    --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 \
    --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 \
    --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3 \
    > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run adagcl yelp2018     50
run adagcl amazon-book  30
run adagcl gowalla     100
echo "ADAGCL_ALLDONE $(date)" >> $L

#!/usr/bin/env bash
# R70: NCL classic baseline (task #10, WWW'22 arXiv:2202.06200) on the three core
# datasets. NCL = clean LightGCN backbone + structure-contrastive InfoNCE (layer-0
# center vs layer-2*hyper_layers context, paper eq.6) + prototypical-contrastive
# InfoNCE (node vs its k-means centroid, paper eq.10-11), trained with an EM k-means
# E-step every m_step=1 epoch and a prototype-loss warm-up (warm_up_step=20).
#
# Protocol (fair, matched to R29/R65/R66): embed_size=256 capacity-matched, n_layers=3,
# seed=42, best-over-epochs (R@20/NDCG@20 on the test set), batch=1024. NCL does NOT
# use our noise / SSL-anneal / semantic components (ncl=1 switches them off).
#
# Hyperparameters: official properties/NCL.yaml + per-dataset yaml (yelp/amazon-books/
# gowalla-merged), with ONE documented adjustment -- the ssl_reg / proto_reg are scaled
# by 4x (official batch 4096 -> our batch 1024) because NCL's InfoNCE losses are SUMMED
# over the batch, so the effective SSL:BPR gradient ratio is proportional to
# batch_size * reg_weight. The 4x keeps the official effective ratio.
#   gowalla : temp=0.05 ssl_reg=4e-6 alpha=0.5 proto_reg=2e-7 num_clusters=10  lr=2e-3
#   yelp    : temp=0.05 ssl_reg=4e-6 alpha=1.5 proto_reg=4e-7 num_clusters=2000 lr=1e-3
#   amazon  : temp=0.05 ssl_reg=4e-6 alpha=0.8 proto_reg=4e-7 num_clusters=2000 reg=1e-6
#   (official: ssl_reg=1e-6, proto_reg=5e-8/1e-7, lr gowalla=2e-3)
#
# Epoch budget: NCL is a LightGCN+GCL model (like SGL), so it gets the SSL-baseline
# matched budget (gowalla 200ep / yelp 100ep / amazon 100ep). stop_cnt=1000 ->
# best-over-epochs. k-means on 256-dim with num_clusters=2000 is ~1.5s/E-step (scipy),
# so it runs sequentially on a single (shared) GPU without issue.
set -e
GPU="${1:-0}"
cd "$(dirname "$0")/.."
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
PROG=research_logs/r70_ncl_progress.txt
LOG=research_logs/r70_ncl_$(date +%H%M%S).log

{
  echo "BEGIN $(date)"

  echo "START gowalla ncl gpu=$GPU ep=200 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=200 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ncl=1 --ncl_ssl_reg=4e-6 \
    --ncl_proto_reg=2e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=0.5 \
    --ncl_num_clusters=10 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 \
    --reg=1e-4 --lr=2e-3 --batch_size=1024 --gpu_id=$GPU
  echo "END   gowalla rc=$? $(date)"

  echo "START yelp2018 ncl gpu=$GPU ep=100 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ncl=1 --ncl_ssl_reg=4e-6 \
    --ncl_proto_reg=4e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=1.5 \
    --ncl_num_clusters=2000 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 \
    --reg=1e-4 --lr=1e-3 --batch_size=1024 --gpu_id=$GPU
  echo "END   yelp2018 rc=$? $(date)"

  echo "START amazon-book ncl gpu=$GPU ep=100 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=amazon-book --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ncl=1 --ncl_ssl_reg=4e-6 \
    --ncl_proto_reg=4e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=0.8 \
    --ncl_num_clusters=2000 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 \
    --reg=1e-6 --lr=1e-3 --batch_size=1024 --gpu_id=$GPU
  echo "END   amazon-book rc=$? $(date)"

  echo "R70_ALLDONE $(date)"
  echo "ALLDONE $(date)"
} > "$PROG" 2> "$LOG"

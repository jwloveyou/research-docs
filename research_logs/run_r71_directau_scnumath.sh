#!/usr/bin/env bash
# R71: DirectAU standalone classic baseline (task #10, KDD'22 arXiv:2206.12811) on the
# three core datasets — scnumath edition (python path /home/zhangxu/...). Same protocol
# as R29/R65/R66 baselines: embed256 capacity-matched, seed=42, best-over-epochs
# (R@20/NDCG@20 on the test set), lr=0.001, reg=1e-4, batch=1024. DirectAU has NO GCN
# propagation and NO full-graph InfoNCE -> light & fast per epoch. Budgets: gowalla
# 300ep / yelp2018 100ep / amazon-book 100ep (LightGCN-style longer budget for a
# pure-embedding method, since the uniformity term is a global property).
set -e
GPU="${1:-7}"
cd "$(dirname "$0")/.."
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
PROG=research_logs/r71_directau_progress.txt
LOG=research_logs/r71_directau_$(date +%H%M%S).log

{
  echo "BEGIN $(date)"
  echo "START gowalla directau gpu=$GPU ep=300 $(date)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=300 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --directau=1 --directau_gamma=1.0 \
    --directau_t=2.0 --reg=1e-4 --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   gowalla directau rc=$? $(date)"

  echo "START yelp2018 directau gpu=$GPU ep=100 $(date)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --directau=1 --directau_gamma=1.0 \
    --directau_t=2.0 --reg=1e-4 --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   yelp2018 directau rc=$? $(date)"

  echo "START amazon-book directau gpu=$GPU ep=100 $(date)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=amazon-book --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --directau=1 --directau_gamma=1.0 \
    --directau_t=2.0 --reg=1e-4 --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   amazon-book directau rc=$? $(date)"

  echo "R71_ALLDONE $(date)"
  echo "ALLDONE $(date)"
} > "$PROG" 2> "$LOG"

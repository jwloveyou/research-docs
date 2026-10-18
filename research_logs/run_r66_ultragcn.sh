#!/usr/bin/env bash
# R66: UltraGCN classic baseline (task #10, CIKM'21 arXiv:2110.15114) on the three
# core datasets. Same protocol as R29/R65 baselines: embed256 capacity-matched,
# seed=42, best-over-epochs (R@20/NDCG@20 on the test set), lr=0.001, reg=1e-4,
# batch=1024. UltraGCN has no GCN propagation -> light & fast, so run sequentially on
# a shared GPU0. Budgets: gowalla 150ep / yelp2018 100ep / amazon-book 100ep.
set -e
GPU="${1:-0}"
cd "$(dirname "$0")/.."
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
PROG=research_logs/r66_ultragcn_progress.txt
LOG=research_logs/r66_ultragcn_$(date +%H%M%S).log

{
  echo "BEGIN $(date)"
  echo "START gowalla ultragcn gpu=$GPU ep=150 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=150 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ultragcn=1 --ultragcn_lambda=1.0 \
    --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --reg=1e-4 --lr=0.001 \
    --batch_size=1024 --gpu_id=$GPU
  echo "END   gowalla rc=$? $(date)"

  echo "START yelp2018 ultragcn gpu=$GPU ep=100 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ultragcn=1 --ultragcn_lambda=1.0 \
    --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --reg=1e-4 --lr=0.001 \
    --batch_size=1024 --gpu_id=$GPU
  echo "END   yelp2018 rc=$? $(date)"

  echo "START amazon-book ultragcn gpu=$GPU ep=100 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=amazon-book --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ultragcn=1 --ultragcn_lambda=1.0 \
    --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --reg=1e-4 --lr=0.001 \
    --batch_size=1024 --gpu_id=$GPU
  echo "END   amazon-book rc=$? $(date)"

  echo "ALLDONE $(date)"
} > "$PROG" 2> "$LOG"

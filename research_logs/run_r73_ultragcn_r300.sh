#!/usr/bin/env bash
# R73: UltraGCN R=300 faithful re-run (protocol fix). The R66 run used num_negatives=1,
# which is NOT UltraGCN's protocol — UltraGCN (CIKM'21, arXiv:2110.15114) requires R=300
# negative samples per positive pair to counteract over-smoothing (paper "negative
# sampling ratio R to 300"). R=1 cripples its BCE signal and is therefore an INVALID
# baseline comparison; this re-run fixes the only mismatch (num_negatives 1 -> 300) while
# keeping every other knob identical to the R66 unified capacity-matched protocol
# (embed256, lr=1e-3, xavier init, reg=1e-4, batch=1024, seed=42, best-over-epochs).
set -e
GPU="${1:-0}"
cd "$(dirname "$0")/.."
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
PROG=research_logs/r73_ultragcn_r300_progress.txt
LOG=research_logs/r73_ultragcn_r300_$(date +%H%M%S).log

{
  echo "BEGIN $(date)"
  echo "START gowalla ultragcn_r300 gpu=$GPU ep=150 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=150 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ultragcn=1 --ultragcn_lambda=1.0 \
    --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --reg=1e-4 \
    --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   gowalla rc=$? $(date)"

  echo "START yelp2018 ultragcn_r300 gpu=$GPU ep=100 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ultragcn=1 --ultragcn_lambda=1.0 \
    --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --reg=1e-4 \
    --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   yelp2018 rc=$? $(date)"

  echo "START amazon-book ultragcn_r300 gpu=$GPU ep=100 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=amazon-book --seed=42 --epochs=100 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --ultragcn=1 --ultragcn_lambda=1.0 \
    --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --reg=1e-4 \
    --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   amazon-book rc=$? $(date)"
  echo "R73_ALLDONE $(date)"
} 2>&1 | tee -a "$PROG" "$LOG"

#!/bin/bash
# R44: DCCF No.1 baseline (task #8, second of the three No.1 baselines).
# Faithful HKUDS/DCCF (SIGIR'23, arXiv:2305.02759) in NoiseSGL.py under dccf=1.
#
# Protocol (same as HCCF R43 / SOTA baselines R29): LightGCN-local backbone, embed_size=256
# (capacity-matched to fusion), n_layers=3 (shared backbone depth; official DCCF uses 2,
# see report note), seed=42, best-over-epochs R@20/NDCG@20 on test, lr=1e-3, batch=1024,
# stop_cnt=1000. DCCF-specific hyperparams from the official parser: n_intents=128,
# temp=1.0, emb_reg=2.5e-5, cen_reg=5e-3, ssl_reg=1e-1. No annealing / noise / capacity
# beyond embed256 (DCCF has no such components).
#
# Epoch budget matches the SSL baselines: gowalla 150ep / yelp 50ep / amazon 30ep.
# Ordered yelp -> amazon -> gowalla so the small datasets finish first.
#
# Usage: bash research_logs/run_dccf_baseline.sh
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=0
L=research_logs/r44_dccf_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; eps=$2
  log=research_logs/r44_dccf_${ds}.log
  echo "START ${ds} gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --use_mlc=0 --gpu_id=$GPU \
    --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 \
    --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1 --reg=1e-7 > $log 2>&1
  rc=$?
  echo "END   ${ds} rc=$rc $(date)" >> $L
  echo "--- ${ds} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run yelp2018    50
run amazon-book 30
run gowalla   150
echo "DCCF_ALLDONE $(date)" >> $L

#!/bin/bash
# R54 (必补 must-not-miss): 组件② win30 退火「无」侧 3-seed 补完。
#   "有" side (win30: ssl_anneal=2/epochs=30) already 3-seed in R24:
#     gowalla embed256 seed42/43/44 = 0.1880/0.1884/0.1883 (mean 0.18823).
#   "无" side (fixed lambda: ssl_anneal=0) = this script: gowalla 100ep x seeds{42,43,44}.
#   Protocol EXACTLY matched to R24 (embed256, view_mix=nn, use_mlc=0, noise_eps=0.2,
#   ssl_reg=0.1, stop_cnt=500, eval best-over-epochs). Only ssl_anneal toggles 2->0.
#   Runs on masmath GPU0 (shared with my own DCCF + ml-1m-sem tasks; ~17GB free).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r54_noanneal_progress.txt
echo "BEGIN $(date)" > $L

run () {
  seed=$1
  tag=noanneal
  ds=gowalla
  log=research_logs/r54_${ds}_${tag}_s${seed}.log
  echo "START ${ds}_${tag}_s${seed} gpu=0 ep=100 ssl=0.1 anneal=0 $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=100 --verbose=1 \
    --stop_cnt=500 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=0 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_s${seed} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_s${seed} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

for s in 42 43 44; do run $s; done
echo "ALLDONE $(date)" >> $L

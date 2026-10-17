#!/bin/bash
# R25: main-branch embedding dropout scan (4th-component candidate), gowalla side.
# scnumath GPUs 6/7 (idle cards only; 0-5 occupied/partly used -> untouched).
# Fusion config: embed256 + win30 cosine SSL anneal + embedding-noise views.
# gowalla dropout in {0 (ctrl, reproduce R24 0.1880@ep61), 0.1, 0.3, 0.5}, 100ep each.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r25_scnumath_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; dr=$2; eps=$3
  tag="drop${dr}"
  log=research_logs/r25_gowalla_${tag}.log
  echo "START gowalla_${tag} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=gowalla \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=500 --embed_size=256 \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=$dr --gpu_id=$gpu > $log 2>&1
  echo "END   gowalla_${tag} rc=$? $(date)" >> $L
  grep "best_result" -A3 $log | tail -3 >> $L
}

# GPU6: ctrl -> 0.1
( run 6 0 100
  run 6 0.1 100 ) &
# GPU7: 0.3 -> 0.5
( run 7 0.3 100
  run 7 0.5 100 ) &
wait
echo "ALLDONE $(date)" >> $L

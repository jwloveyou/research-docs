#!/bin/bash
# R25: main-branch embedding dropout scan (4th-component candidate).
# masmath GPU0 sequential chain. Absolute python path (nohup-safe).
# Fusion config: embed256 + win30 cosine SSL anneal + embedding-noise views.
# dropout in {0 (ctrl, reproduce R24), 0.1, 0.3} for yelp; then amazon.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r25_masmath_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; eps=$2; dr=$3
  tag="drop${dr}"
  log=research_logs/r25_${ds}_${tag}.log
  echo "START ${ds}_${tag} ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=500 --embed_size=256 \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=$dr --gpu_id=0 > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  grep "best_result" -A3 $log | tail -3 >> $L
}

# yelp2018: dropout {0, 0.1, 0.3}, 20ep each (~35min each)
run yelp2018 20 0
run yelp2018 20 0.1
run yelp2018 20 0.3
# amazon-book: dropout {0, 0.1, 0.3}, 12ep each (~90min each)
run amazon-book 12 0
run amazon-book 12 0.1
run amazon-book 12 0.3
echo "ALLDONE $(date)" >> $L

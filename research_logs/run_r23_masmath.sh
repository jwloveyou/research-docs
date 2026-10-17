#!/bin/bash
# R23: L2-normalised (cosine) scoring ablation + embed256 capacity extension.
# masmath GPU0 (sequential chain). Absolute python path (nohup-safe, conda activate fails).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r23_masmath_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; ds=$2; eps=$3; emb=$4; l2n=$5
  log=research_logs/r23_${ds}_${tag}.log
  echo "START ${ds}_${tag} $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=500 --embed_size=$emb \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=$l2n --gpu_id=0 > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  grep "best_result" -A3 $log | tail -3 >> $L
}

# Primary: yelp ctrl (embed128, l2_norm=0)
run ctrl_emb128_l2n0 yelp2018 20 128 0
# Secondary: yelp embed256 (capacity extension, untested)
run emb256_l2n0 yelp2018 20 256 0
# Secondary: gowalla embed256 re-verify (known 0.1880@ep61 from R19/R20)
run emb256_l2n0 gowalla 100 256 0
echo "ALLDONE $(date)" >> $L

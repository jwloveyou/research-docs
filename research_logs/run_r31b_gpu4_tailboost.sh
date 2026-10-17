#!/bin/bash
# R31b (backup, LAGCL-inspired): tail-boost noise (gamma < 0) on gowalla.
#   LAGCL (arXiv:2309.11177) enhances TAIL nodes to fix degree disparity; this is the
#   OPPOSITE hypothesis to R28's head-boost (gamma>0 = more noise on high-degree nodes).
#   With gamma<0 the linear scale = 1 + gamma*(c - mean(c)) gives tail (low-degree) nodes
#   MORE noise and head nodes LESS noise. Run only if R31 head-boost scan is negative/neutral.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r31bg4_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; gamma=$5
  log=research_logs/r31bg4_${ds}_${tag}_g${gamma}.log
  echo "START ${ds}_${tag}_g${gamma} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=$gamma --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_g${gamma} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_g${gamma} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# tail-boost (gamma < 0): more noise on low-degree (tail) nodes
run 4 gowalla tailboost 30 -0.5
run 4 gowalla tailboost 30 -1.0

echo "ALLDONE $(date)" >> $L

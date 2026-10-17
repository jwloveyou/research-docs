#!/bin/bash
# R31c (follow-up): 100ep gowalla confirmation of density-aware noise at the real peak
# (R24 embed256 gowalla peaks ~ep61; the 30ep smoke test only shows early trajectory).
#   gamma=0 (ctrl) vs gamma=1.0 (best head-boost from smoke scan) vs gamma=0.5 (NDCG hint)
#   seed=42, embed256 + win30, matched to R24 protocol.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r31cg4_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; gamma=$5
  log=research_logs/r31cg4_${ds}_${tag}_g${gamma}.log
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

run 4 gowalla long 100 0.0
run 4 gowalla long 100 0.5
run 4 gowalla long 100 1.0

echo "ALLDONE $(date)" >> $L

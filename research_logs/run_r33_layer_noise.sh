#!/bin/bash
# R33 (4th-component candidate): depth-aware (layer-wise) noise schedule.
#   noise_layer_gamma=0  -> uniform per-layer noise magnitude (ctrl, backward compatible)
#   noise_layer_gamma>0  -> deeper layers get a stronger perturbation:
#       scale_k = eps * (1 + gamma * k/(L-1)),  k=0..L-1
#     Motivation: over-smoothing grows with depth (E11: the final LightGCN embedding
#     collapses toward the principal eigenvector), so deeper layers need a larger
#     re-diversifying noise. Static, parameter-free, orthogonal to the node axis
#     (noise_deg_gamma, R28) and the epoch axis (noise_anneal, R20); respects the
#     R1/R15/R18 'static wins' law and the R4/R6 'don't change the propagation
#     operator' constraint (the GCN operator A is untouched, only the per-layer
#     perturbation magnitude changes).
# Smoke test: gowalla embed256 + win30, seed=42, gamma in {0, 0.5, 1.0, 2.0}, 60ep.
#   (embed256 gowalla peaks ~ep61 per R24, so 60ep reaches the real peak.)
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r33_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; emb=$5; gamma=$6
  log=research_logs/r33_${ds}_${tag}_g${gamma}.log
  echo "START ${ds}_${tag}_g${gamma} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=$emb --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_layer_gamma=$gamma --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_g${gamma} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_g${gamma} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run $1 layer gowalla 60 256 0.0
run $1 layer gowalla 60 256 0.5
run $1 layer gowalla 60 256 1.0
run $1 layer gowalla 60 256 2.0

echo "ALLDONE $(date)" >> $L

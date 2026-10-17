#!/bin/bash
# R39 (4th-component candidate #13): static non-uniform layer-aggregation weights.
#   The fusion model currently mean-pools the L+1 layer embeddings (LightGCN default).
#   This scan replaces the uniform pool with a STATIC, parameter-free weighting of the
#   final layer pool (shared by main branch / noise views / eval, so the aggregation
#   operator stays identical, cf. R4/R6/R7):
#     uniform : mean-pooling (matched control, backward compatible)
#     shallow : w_k ∝ L+1-k  (favours layer-0; anti-over-smoothing)
#     deep    : w_k ∝ k+1    (favours the last layer)
#     geom    : w_k ∝ alpha^k, alpha=0.5 (geometric shallow-favoured)
#   Motivation: E11 shows over-smoothing accumulates with depth (cos(l0,lL)=0.46 even
#   with noise), so deeper layers carry a partially-collapsed signal; a shallow-favoured
#   static pool is a direct anti-over-smoothing measure complementing (not replacing) the
#   noise augmentation's rank-release. Literature: APPNP teleport (1810.05997) / GCNII
#   initial residual (2007.02133) / SGC layer-collapse analysis.
#
#   Protocol: gowalla embed256 + win30, seed=42, 60ep (R24 gowalla embed256 peaks ~ep61,
#   so 60ep reaches the real peak). uniform is a matched same-budget control.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r39_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; agg=$3
  log=research_logs/r39_gowalla_agg_${tag}.log
  echo "START gowalla_agg_${tag} gpu=$gpu ep=60 agg=$agg $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=60 --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
    --agg_weights=$agg --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   gowalla_agg_${tag} rc=$rc $(date)" >> $L
  echo "--- gowalla_agg_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run $1 uniform uniform
run $1 shallow shallow
run $1 deep    deep
run $1 geom    geom

echo "ALLDONE $(date)" >> $L

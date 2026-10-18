#!/bin/bash
# R49 (4th-component candidate #16): DropEdge-style edge dropout on the MAIN propagation
# branch (prediction/BPR branch only). dropedge_keep=1.0 keeps the full adjacency (ctrl,
# backward compatible); <1.0 drops (1-keep) of the bipartite training edges once per epoch
# and re-normalises, used ONLY by the main branch -- the noise views keep full norm_adj so
# the SSL signal is unchanged (respects R4/R6/R7). eval() still uses the full norm_adj
# (standard DropEdge train/test protocol, Rong et al. ICLR 2020).
# Mechanism: E11 shows depth-accumulated over-smoothing is still present in the main
# branch (cos(l0,lL)=0.46 even with noise); the noise only regularises the InfoNCE views,
# so the main branch currently has NO direct anti-over-smoothing. DropEdge breaks the graph
# connectivity during propagation and is the canonical static remedy. Orthogonal to the 3
# fusion components (noise R1 / SSL anneal R15 / capacity R18); distinct from R25 (input-
# embedding dropout, neutral) and R6/R7 (edge dropout on the CONTRAST views, harmful).
# Scan: gowalla embed256+win30 seed42 60ep (real peak ~ep61) keep in {1.0, 0.9, 0.8, 0.6}
#   + yelp (15ep) / amazon (12ep) keep in {1.0, 0.9} cross-dataset confirmation.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r49_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; keep=$5
  log=research_logs/r49_${ds}_${tag}_k${keep}.log
  echo "START ${ds}_${tag}_k${keep} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --dropedge_keep=$keep --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_k${keep} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_k${keep} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

gpu=$1

# Phase 1: gowalla keep-rate scan (60ep reaches the real embed256 peak ~ep61).
run $gpu gowalla de 60 1.0
run $gpu gowalla de 60 0.9
run $gpu gowalla de 60 0.8
run $gpu gowalla de 60 0.6

# Phase 2: cross-dataset confirmation at the a-priori most likely keep=0.9 (vs ctrl 1.0).
run $gpu yelp2018 de 15 1.0
run $gpu yelp2018 de 15 0.9
run $gpu amazon-book de 12 1.0
run $gpu amazon-book de 12 0.9

echo "R49_ALLDONE $(date)" >> $L

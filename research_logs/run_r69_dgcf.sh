#!/bin/bash
# R69 (task #10): DGCF classic-baseline same-protocol reproduction for the paper table.
# DGCF (SIGIR'20, arXiv:2007.01764) = the canonical 'disentangled intent' baseline: K
# learnable intent prototypes + per-intent weighted GCN propagation (routing
# p(u,i,k)=softmax_k((e_{u,k}^T c_k + e_{i,k}^T c_k)/tau)) + an independence
# (distance-correlation) regulariser. Pure BPR, no contrastive SSL. As a LEARNED-structure
# model it is a clean counterfactual to our 'static wins' iron law (R4/R19), alongside
# NGCF/HCCF/DCCF/AdaGCL.
#
# Protocol (fair, matched to R29/R65): embed_size=256 capacity-matched, n_layers=3,
# K=4 intents (paper default), tau=1.0, ind_reg=1e-2, seed=42, best-over-epochs
# (R@20/NDCG@20 on test, stop_cnt=1000), lr=1e-3, batch=1024, reg=1e-4. DGCF does NOT
# use our annealing / noise / semantic (those are our fusion components; dgcf=1 switches
# them off).
#
# Epoch budget: DGCF is a pure-BPR GNN (like NGCF/LightGCN), so it gets the LightGCN-style
# longer budget (gowalla 200ep / yelp2018 100ep / amazon-book 100ep).
#
# Usage: bash run_r69_dgcf.sh <gpu_id>
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=${1:-0}
L=research_logs/r69_dgcf_progress.txt
[ -f $L ] || echo "BEGIN $(date)" > $L
echo "BEGIN gpu=$GPU $(date)" >> $L

run () {
  ds=$1; eps=$2
  log=research_logs/r69_dgcf_${ds}.log
  echo "START ${ds} dgcf gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --use_mlc=0 --dgcf=1 \
    --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01 \
    --ssl_anneal=0 --ssl_anneal_epochs=0 --ssl_anneal_floor=0.0 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --gpu_id=$GPU > $log 2>&1
  rc=$?
  echo "END   ${ds} dgcf rc=$rc $(date)" >> $L
  echo "--- ${ds} dgcf best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run gowalla      200
run yelp2018     100
run amazon-book  100
echo "DGCF_ALLDONE $(date)" >> $L
echo "ALLDONE $(date)" >> $L

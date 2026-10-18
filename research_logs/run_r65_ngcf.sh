#!/bin/bash
# R65 (task #10): NGCF classic-baseline same-protocol reproduction for the paper table.
# NGCF (SIGIR'19, arXiv:1905.08108) is the direct predecessor of LightGCN and the most
# conspicuously missing classic baseline. Faithful architecture: per-layer learnable W1/W2
# feature transformations + LeakyReLU(0.2) + CONCATENATION of all L+1 layer embeddings
# (final dim (L+1)*embed_size), node/message dropout 0.1/0.1, pure mean-BPR (no SSL).
#
# Protocol (fair, matched to R29): same LightGCN-256 embedding size (embed_size=256, so
# NGCF's concat final dim = 1024 -- a CAPACITY ADVANTAGE for NGCF, i.e. conservative for
# our fusion claim), n_layers=3, same data split / seed=42 / eval (R@20,NDCG@20 on test,
# best-over-epochs), lr=1e-3, batch=1024, reg=1e-4. NGCF does NOT use our annealing /
# noise / semantic (those are our fusion components; ngcf=1 switches them off).
#
# Epoch budget: NGCF is a pure-BPR GNN (like LightGCN), so it gets the LightGCN-style
# longer budget (gowalla 200ep / yelp 100ep / amazon 100ep). stop_cnt=1000 -> best-over-epochs.
#
# Usage: bash run_r65_ngcf.sh <gpu_id>
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=${1:-0}
L=research_logs/r65_ngcf_progress.txt
[ -f $L ] || echo "BEGIN $(date)" > $L
echo "BEGIN gpu=$GPU $(date)" >> $L

run () {
  ds=$1; eps=$2
  log=research_logs/r65_ngcf_${ds}.log
  echo "START ${ds} ngcf gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --use_mlc=0 --ngcf=1 \
    --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1 \
    --ssl_anneal=0 --ssl_anneal_epochs=0 --ssl_anneal_floor=0.0 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --gpu_id=$GPU > $log 2>&1
  rc=$?
  echo "END   ${ds} ngcf rc=$rc $(date)" >> $L
  echo "--- ${ds} ngcf best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run gowalla      200
run yelp2018     100
run amazon-book  100
echo "R65_ALLDONE $(date)" >> $L
echo "ALLDONE $(date)" >> $L

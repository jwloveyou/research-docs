#!/bin/bash
# R47 (task #6): frozen LLM semantic item feature ablation.
#   sem_emb=0  -> pure ID (ctrl, = current fusion, backward compatible)
#   sem_emb=1  -> add sem_lam * item_sem (frozen TF-IDF-over-tags -> seeded Gaussian
#                 projection -> L2-norm) to the layer-0 item embedding.
# Protocol matches the fusion model (R22/R24): embed256 + win30 cosine SSL anneal +
# view_mix=nn + noise_eps=0.2, seed=42, best-over-epochs on test R@20/NDCG@20.
# Datasets (both carry public text metadata):
#   * lastfm-2k (fusion wins +4.1% over LightGCN): artist names + user tags.
#   * ml-1m    (fusion loses -3.3%, dense): movie titles + genres -- does the semantic
#              prior rescue the dense case / cold-start items?
# Usage: bash run_r47_sem_ablation.sh <gpu_id>   (gpu_id is the scnumath card index)
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r47_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; sem=$5; lam=$6
  log=research_logs/r47_${ds}_${tag}_sem${sem}_lam${lam}.log
  echo "START ${ds}_${tag}_sem${sem}_lam${lam} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --sem_emb=$sem --sem_lam=$lam --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_sem${sem}_lam${lam} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_sem${sem}_lam${lam} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# lastfm-2k: ctrl + 3 strengths (lam scan)
run $1 lastfm-2k sem 200 0 1.0
run $1 lastfm-2k sem 200 1 0.5
run $1 lastfm-2k sem 200 1 1.0
run $1 lastfm-2k sem 200 1 2.0
# ml-1m: ctrl + lam=1.0
run $1 ml-1m sem 200 0 1.0
run $1 ml-1m sem 200 1 1.0

echo "ALLDONE $(date)" >> $L

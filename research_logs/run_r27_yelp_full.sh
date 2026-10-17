#!/bin/bash
# R27: generalization validation on Yelp full (yelp_full, ~3.5M interactions, density 0.0075%).
# Protocol = R26: keep authors' train/test split, cold-start dropped, 0-indexed contiguous ids.
#   fusion = NoiseSGL 3 components (view_mix=nn + ssl_anneal=2/window=30 + embed_size=256)
#   lgn    = capacity-matched LightGCN baseline (ssl_reg=0, embed_size=256)
# equal epoch budget 15ep, seed=42. yelp_full is ~16min/ep at embed256 -> run on scnumath GPU5.
# Optionally chained: embed128 variants (capacity-robustness check) after the embed256 pair.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r27_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5; emb=$6
  log=research_logs/r27_${ds}_${tag}_emb${emb}.log
  echo "START ${ds}_${tag}_emb${emb} gpu=$gpu ep=$eps ssl=$ssl $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=1000 --embed_size=$emb \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_emb${emb} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_emb${emb} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# Primary (protocol-consistent): fusion embed256 vs lgn embed256, equal 15ep budget.
# (split from embed128 pair to stay well under the 12h forced-logout limit)
run 5 fusion yelp_full 15 0.1 256
run 5 lgn    yelp_full 15 0   256

echo "ALLDONE $(date)" >> $L

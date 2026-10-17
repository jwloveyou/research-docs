#!/bin/bash
# R27: yelp_full LightGCN baseline (capacity-matched embed256, ssl_reg=0), 15ep, seed=42.
# Launch AFTER the fusion emb256 run on GPU5 completes.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
log=research_logs/r27_yelp_full_lgn_emb256.log
echo "START yelp_full_lgn_emb256 $(date)" >> research_logs/r27_progress.txt
CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
  --recommender=NoiseSGL --dataset=yelp_full --seed=42 --epochs=15 --verbose=1 \
  --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
  --ssl_reg=0 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
  --l2_norm=0 --dropout=0 --gpu_id=5 > $log 2>&1
echo "END   yelp_full_lgn_emb256 rc=$? $(date)" >> research_logs/r27_progress.txt
grep 'best_result' -A3 $log | tail -4 >> research_logs/r27_progress.txt

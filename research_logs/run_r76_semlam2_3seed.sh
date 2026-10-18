#!/bin/bash
# R76b: 3-seed confirmation of the semantic-component optimal strength sem_lam=2.0
# (single-seed sensitivity scan found lam=2.0 > 1.0 on lastfm: 0.4238 vs 0.4209 R@20).
# Protocol identical to R53 3-seed (embed256 + win30 + view_mix=nn + noise_eps=0.2,
# 100ep quick signal, best-over-epochs). Compare against R53 sem0 3-seed (0.41827).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r76b_semlam2_3seed_progress.txt
echo "BEGIN $(date)" > $L
for s in 42 43 44; do
  log=research_logs/r76_lastfm-2k_semlam2_s${s}.log
  echo "START lastfm-2k sem_lam=2.0 seed=$s $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=lastfm-2k --seed=$s --epochs=100 --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --sem_emb=1 --sem_lam=2.0 --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   lastfm-2k sem_lam=2.0 seed=$s rc=$rc $(date)" >> $L
  echo "--- lastfm-2k sem_lam=2.0 seed=$s best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
done
echo "ALLDONE $(date)" >> $L

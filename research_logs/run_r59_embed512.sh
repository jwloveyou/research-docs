#!/bin/bash
# R59: capacity-scaling saturation test — embed256 -> embed512 on the three core datasets.
#
# Motivation: R18 (64->128), R23/R24 (128->256) gave monotonic R@20 gains on all three
# datasets (amazon +9.3% is the largest, suggesting 91.6K items still under-capacity at 256),
# so the capacity axis may not be saturated. This round probes the next point on the curve
# (embed512) to (a) establish the capacity boundary for the sensitivity analysis (task#2),
# and (b) check whether amazon (and possibly gowalla/yelp) gains further.
#
# Protocol = R24 (final fusion, seed42): view_mix=nn + noise_eps=0.2 + ssl_reg=0.1 +
# win30 cosine SSL anneal (ssl_anneal=2/epochs=30/floor=0) + stop_cnt=500 + best-over-epochs.
# ONLY change vs R24 = embed_size 256 -> 512 (static, no learnable params, respects R4/R19 law).
#
# Runs on masmath GPU0 (our own card, shared with DCCF / ml-1m sem / r54 noanneal / r57 mech —
# all our own jobs; memory stays well below 24G). Budgets: gowalla 60ep (reaches the embed256
# ep61 peak; if embed512 peaks later that is itself a finding), yelp 20ep, amazon 15ep.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r59_embed512_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; eps=$2
  log=research_logs/r59_${ds}_emb512_s42.log
  echo "START ${ds}_emb512 ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=512 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   ${ds}_emb512 rc=$rc $(date)" >> $L
  echo "--- ${ds}_emb512 best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run gowalla 60
run yelp2018 20
run amazon-book 15

echo "ALLDONE $(date)" >> $L

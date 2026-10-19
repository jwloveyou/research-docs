#!/bin/bash
# R78: embed512 3-seed verification (capacity-scaling 256 -> 512 upgrade decision).
#
# Motivation (critical, not just marginal): at embed256 the fusion model is
#   - yelp2018 : fusion 0.0709 vs SimGCL 0.0710 (tie / -0.1%, within noise)
#   - amazon-book : fusion 0.0551 vs SGL 0.0566 (**-2.7%, real deficit**)
#   - gowalla  : fusion 0.1880 > all baselines (clean win)
# embed512 single-seed (R59, seed42) already fixes yelp: 0.0716 > SimGCL 0.0710.
# amazon embed512 (seed42) is converging faster than embed256 and is expected to clear
# SGL 0.0566; this 3-seed run (seed43/44) settles the upgrade decision.
#
# Protocol = R59 (single-seed embed512) exactly, just seeds 43/44:
#   view_mix=nn + noise_eps=0.2 + ssl_reg=0.1 + win30 cosine SSL anneal
#   (ssl_anneal=2/epochs=30/floor=0) + stop_cnt=1000 + best-over-epochs + embed_size=512.
# ONLY change vs R24 (embed256 3-seed) = embed_size 256 -> 512 (static, no learnable params).
#
# Runs SEQUENTIALLY on one GPU (safer than parallel, avoids the R52-style contention bug).
# Budgets (same as R59): gowalla 60ep / yelp 20ep / amazon 15ep.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=${1:-6}
L=research_logs/r78_embed512_3seed_progress.txt
echo "BEGIN $(date) gpu=$GPU" > $L

run () {
  ds=$1; seed=$2; eps=$3
  log=research_logs/r78_${ds}_emb512_s${seed}.log
  echo "START ${ds}_emb512_s${seed} gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=512 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --gpu_id=$GPU > $log 2>&1
  rc=$?
  echo "END   ${ds}_emb512_s${seed} rc=$rc $(date)" >> $L
  echo "--- ${ds}_emb512_s${seed} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run gowalla 43 60
run gowalla 44 60
run yelp2018 43 20
run yelp2018 44 20
run amazon-book 43 15
run amazon-book 44 15

echo "ALLDONE $(date)" >> $L

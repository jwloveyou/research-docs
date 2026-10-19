#!/bin/bash
# R81: embed512 3-seed verification (capacity-scaling 256 -> 512 upgrade decision).
#
# Motivation: R59 (embed512, seed42) already shows all three core datasets BEAT embed256:
#   gowalla 0.1894/0.1596@ep29 vs 0.1880/0.1598@ep61 (R24 3-seed 0.18823)
#   yelp2018 0.0716/0.0587@ep4 vs 0.0709/0.0584 (R24 3-seed 0.07087)
#   amazon-book 0.0589/0.0455@ep15 vs 0.0551/0.0426 (R24 3-seed 0.05517)  << +6.9%, largest
# This run adds seed43/44 to settle the 256->512 upgrade with 3-seed zero-crossing.
#
# Protocol = R59 (single-seed embed512) exactly, just seeds 43/44:
#   view_mix=nn + noise_eps=0.2 + ssl_reg=0.1 + win30 cosine SSL anneal
#   (ssl_anneal=2/epochs=30/floor=0) + stop_cnt=1000 + best-over-epochs + embed_size=512.
# ONLY change vs R24 (embed256 3-seed) = embed_size 256 -> 512 (static, no learnable params).
#
# Runs SEQUENTIALLY on masmath GPU0 (shared with the R80 light-baseline chain — both are our
# own jobs; memory stays well below 24G). Budgets = R59: gowalla 60ep / yelp 20ep / amazon 15ep.
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=0
L=research_logs/r81_embed512_3seed_progress.txt
echo "BEGIN $(date) gpu=$GPU" > $L

run () {
  ds=$1; seed=$2; eps=$3
  log=research_logs/r81_${ds}_emb512_s${seed}.log
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

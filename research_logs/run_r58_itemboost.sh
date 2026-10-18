#!/bin/bash
# R58 (4th-component candidate #17): item-side noise boost (asymmetric noise magnitude).
#   noise_item_boost=0  -> symmetric noise on user/item rows (ctrl, backward compatible)
#   noise_item_boost>0  -> item rows get (1+boost)*noise_eps, user rows stay noise_eps
# Motivation: item embeddings live on the larger, heavier-skewed side (|I| >> |U|, a few
# head items dominate), so they suffer the rank-collapse / over-smoothing pathology (E11)
# MORE than user rows. A stronger re-diversifying perturbation on the item side is a
# targeted anti-collapse move, complementary to the symmetric noise's uniform rank release.
# Static & parameter-free (respects the R1/R15/R18 'static wins' law and R4/R6/R7). The
# operator stays shared across main branch / noise views / eval (only the per-node noise
# MAGNITUDE changes on the item rows, inside the existing perturb blocks).
# Literature: SimGCL (2112.08679, symmetric noise), XSimGCL (2209.02544), FairDAM/AdaGCL
# (degree-aware augmentation), LAGCL (2309.11177, tail boosting). The item/user asymmetry
# is the orthogonal "side axis" not covered by degree (node) / depth / epoch noise axes.
#
# Protocol: gowalla embed256 + win30, seed=42, boost in {0, 0.25, 0.5, 1.0}, 60ep (reaches
# the R24 embed256 gowalla peak ~ep61). If gowalla shows a robust gain, yelp/amazon 15ep
# confirmation follows automatically (best boost from the gowalla scan).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r58_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; boost=$5
  log=research_logs/r58_${ds}_${tag}_b${boost}.log
  echo "START ${ds}_${tag}_b${boost} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_item_boost=$boost --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_b${boost} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_b${boost} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# Phase 1: gowalla scan (reaches true peak ~ep61)
run $1 gowalla itemb 60 0.0
run $1 gowalla itemb 60 0.25
run $1 gowalla itemb 60 0.5
run $1 gowalla itemb 60 1.0

# Phase 2: cross-dataset confirmation (best boost; conservative default 0.5 if tie)
BEST=0.5
run $1 yelp2018 itemb 15 $BEST
run $1 amazon-book itemb 15 $BEST

echo "ALLDONE $(date)" >> $L

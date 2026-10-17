#!/bin/bash
# R35 (4th-component candidate #10): DirectSpec-style spectral balancing loss
# (ACM TORS 2025 "Balancing Embedding Spectrum for Recommendation").
#   spec_bal=0  -> off (ctrl, backward compatible)
#   spec_bal>0  -> minimise the scale-invariant spectral concentration of the batch
#                  embedding Gram (rows unit-normalised inside the loss), i.e. flatten
#                  the spectrum / raise the effective rank.
# Motivation: E11 showed the noise augmentation works by releasing embedding rank
# (41% -> 97% effective rank) and suppressing over-smoothing. DirectSpec directly
# optimises the spectrum distribution to make embeddings "span the entire embedding
# space", so this is the most mechanism-aligned *loss-level* candidate; it complements
# (does not replace) the noise augmentation's INDIRECT rank release. Static and
# parameter-free (no learnable weights; respects the R1/R15/R18 'static wins' law and
# the R4/R6/R7 'keep the propagation operator untouched' constraint). The loss lives on
# the fused main-branch batch embeddings only (representation layer), like R8/R9.
#
# Scan: gowalla embed256 + win30, seed=42, spec_bal in {0, 5, 10, 50}, 60ep
#   (R24 embed256 gowalla peaks ~ep61, so 60ep reaches the real peak; ctrl sb=0 is a
#   matched same-budget control in the same run). yelp/amazon confirmation is queued
#   separately once the gowalla-best coefficient is known (same protocol as R31/R34).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r35_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; sb=$5
  log=research_logs/r35_${ds}_${tag}_sb${sb}.log
  echo "START ${ds}_${tag}_sb${sb} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --spec_bal=$sb --spec_bal_center=1 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_sb${sb} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_sb${sb} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run $1 gowalla spec 60 0
run $1 gowalla spec 60 5
run $1 gowalla spec 60 10
run $1 gowalla spec 60 50

echo "ALLDONE $(date)" >> $L

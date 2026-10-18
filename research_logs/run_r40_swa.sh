#!/bin/bash
# R40 (4th-component candidate #14): stochastic weight averaging (SWA) of the embedding
# tables as a CV cross-domain regulariser. swa_start=0 keeps the standard single-best-
# snapshot protocol (ctrl, backward compatible); swa_start>0 accumulates an equal-weight
# average of the embedding weights over epochs [swa_start, E] and evaluates the RANKING
# with that average (not the last/best single snapshot).
# Motivation: R12c/R15/R23 document that the pure-BPR fine-tune phase (after the SSL
# weight hits floor at ep30) peaks (gowalla embed256 ~ep61) and then deterministically
# overfits; SWA flattens the late-stage loss surface. Static, parameter-free (respects
# the R1/R15/R18 'static wins' law); orthogonal to noise (R1) / SSL anneal (R15) /
# capacity (R18). Literature: SWA UAI'18 (arXiv:1803.05407) / EMA-momentum of BYOL
# (arXiv:2006.07733) & MoCo (arXiv:1911.05722) / Mean Teacher (arXiv:1703.01780).
#
# Scan: gowalla embed256 + win30, seed=42, swa_start in {0, 30, 50, 70}, 100ep
#   (R24 embed256 gowalla peaks ~ep61, so 100ep covers the climb + plateau + overfit
#   window; ctrl swa_start=0 is a matched same-budget control in the same run).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r40_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; swa=$5
  log=research_logs/r40_${ds}_${tag}_swa${swa}.log
  echo "START ${ds}_${tag}_swa${swa} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --swa_start=$swa --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_swa${swa} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_swa${swa} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run $1 gowalla swa 100 0
run $1 gowalla swa 100 30
run $1 gowalla swa 100 50
run $1 gowalla swa 100 70

echo "ALLDONE $(date)" >> $L

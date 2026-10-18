#!/bin/bash
# R60 (4th-component candidate #18): embedding initialization scale.
#   init_scale=1.0 -> standard initializer output (ctrl, backward compatible)
#   init_scale !=1  -> rescale the layer-0 user/item embedding tables right after
#                      reset_parameters, changing the initial embedding norm (and thus
#                      the initial BPR score scale + early gradient magnitude).
# Motivation (E11 / arXiv:2310.04400 embedding collapse / RankElastor 2605.23191):
#   the noise component releases embedding rank *during* training; init scale is the
#   complementary *initial-condition* knob on the same rank/norm geometry, and the
#   standard xavier_uniform bound sqrt(6/(fan_in+fan_out)) was never tuned for the
#   noise-SSL backbone. Static & parameter-free (respects R1/R15/R18 'static wins' law),
#   orthogonal to noise (R1) / SSL anneal (R15) / capacity (R18) / semantic (R47) and to
#   all 17 prior candidates (R20-R58).
# Protocol: gowalla embed256 + win30, seed=42, init_scale in {0.5, 1.0, 2.0, 5.0}, 60ep
#   (reaches the R24 embed256 gowalla peak ~ep61). If gowalla shows a robust gain,
#   yelp/amazon 15ep confirmation follows (best init_scale from the gowalla scan).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r60_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; iscale=$5
  log=research_logs/r60_${ds}_${tag}_is${iscale}.log
  echo "START ${ds}_${tag}_is${iscale} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --init_scale=$iscale --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_is${iscale} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_is${iscale} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# Phase 1: gowalla scan (reaches true peak ~ep61)
run $1 gowalla is 60 0.5
run $1 gowalla is 60 1.0
run $1 gowalla is 60 2.0
run $1 gowalla is 60 5.0

# Phase 2: cross-dataset confirmation (best init_scale; conservative default 1.0 if tie)
BEST=2.0
run $1 yelp2018 is 15 $BEST
run $1 amazon-book is 15 $BEST

echo "ALLDONE $(date)" >> $L

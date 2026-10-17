#!/bin/bash
# R37 (4th-component candidate #11): learning-rate schedule (linear / cosine decay).
#   lr_anneal=0  -> fixed lr (ctrl, backward compatible)
#   lr_anneal=1  -> linear decay  lr -> lr*lr_floor over lr_anneal_epochs, then hold
#   lr_anneal=2  -> cosine decay lr -> lr*lr_floor over lr_anneal_epochs, then hold
# Motivation: R12c/R15/R23 show the BPR fine-tune phase (SSL weight hits floor ~ep30)
# peaks (~ep61 for gowalla embed256) then deterministically overfits. A decaying lr
# shrinks the fine-tune step size to counter that late-stage overfitting. It is an
# OPTIMIZER-level schedule, orthogonal to the SSL-weight schedule (R13/R15, scales the
# InfoNCE magnitude) and to the noise/embedding axes, so it is an independent,
# static/parameter-free candidate. Literature: 'Optimal Linear Decay Learning Rate
# Schedules' (arXiv:2310.07831) + cosine schedule (Loshchilov & Hutter 2017).
#
# Scan: gowalla embed256 + win30, seed=42, 100ep (R24 gowalla peaks ~ep61, so 100ep
#   covers the peak AND the post-peak overfitting window that this candidate targets).
#   ctrl (anneal=0) is a matched same-budget control in the same run.
#   4 configs: ctrl / cosine(win100, floor0.1) / cosine(win200, floor0.1) / linear(win100, floor0.1).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r37_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; eps=$3; anneal=$4; win=$5; floor=$6
  log=research_logs/r37_gowalla_${tag}.log
  echo "START gowalla_${tag} gpu=$gpu ep=$eps anneal=$anneal win=$win floor=$floor $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 \
    --lr_anneal=$anneal --lr_anneal_epochs=$win --lr_floor=$floor --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   gowalla_${tag} rc=$rc $(date)" >> $L
  echo "--- gowalla_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# ctrl / cosine(win100, floor0.1) / cosine(win200, floor0.1) / linear(win100, floor0.1)
run $1 ctrl          100 0 0   0.1
run $1 cos_win100_f01 100 2 100 0.1
run $1 cos_win200_f01 100 2 200 0.1
run $1 lin_win100_f01 100 1 100 0.1

echo "ALLDONE $(date)" >> $L

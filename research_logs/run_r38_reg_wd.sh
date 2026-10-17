#!/bin/bash
# R38 (4th-component candidate #12): regularisation-axis scan.
#   Two knobs are scanned, both static / parameter-free (respect the R1/R15/R18
#   "static wins" law) and orthogonal to the 3 fusion components (noise / SSL anneal /
#   capacity) and to the 6 queued candidates:
#     * reg          : batch-embedding L2 loss weight (already in code, fixed 1e-4, NEVER scanned)
#     * weight_decay : Adam decoupled weight decay on the FULL embedding table
#                      (new knob; was always 0 before this round)
# Motivation: R12c/R15/R18 show a mild deterministic overfit in the pure-BPR fine-tune
#   phase (after ssl_anneal_epochs=30 the SSL weight is 0, peak ~ep61 then declines).
#   Stronger L2 regularisation directly counters that late-stage overfitting.
# Literature: 'Revisiting BPR: A Replicability Study' (arXiv:2409.14217) -- proper
#   regularisation/tuning is the main gap between BPR and SOTA; 'On the Overlooked
#   Pitfalls of Weight Decay' (arXiv:2011.11152); 'On the Theories Behind Hard Negative
#   Sampling' (arXiv:2302.03472) -- top-K connection of the BPR objective.
#
# Scan: gowalla embed256 + win30, seed=42, 60ep (R24 gowalla embed256 peaks ~ep61, so
#   60ep reaches the real peak). ctrl (reg=1e-4, wd=0) is a matched same-budget control
#   in the same run. 5 configs:
#     ctrl  reg=1e-4 wd=0      (fusion default)
#     reg0  reg=0    wd=0      (is the batch-L2 even needed?)
#     reg3  reg=1e-3 wd=0      (10x stronger batch-L2)
#     wd4   reg=1e-4 wd=1e-4   (decoupled Adam weight decay)
#     wd3   reg=1e-4 wd=1e-3   (stronger decoupled weight decay)
#   yelp/amazon confirmation is queued separately if gowalla shows a clear winner
#   (same protocol as R31/R34/R35).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r38_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; eps=$3; reg=$4; wd=$5
  log=research_logs/r38_gowalla_${tag}.log
  echo "START gowalla_${tag} gpu=$gpu ep=$eps reg=$reg wd=$wd $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
    --reg=$reg --weight_decay=$wd --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   gowalla_${tag} rc=$rc $(date)" >> $L
  echo "--- gowalla_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run $1 ctrl 60 1e-4 0.0
run $1 reg0 60 0    0.0
run $1 reg3 60 1e-3 0.0
run $1 wd4  60 1e-4 1e-4
run $1 wd3  60 1e-4 1e-3

echo "ALLDONE $(date)" >> $L

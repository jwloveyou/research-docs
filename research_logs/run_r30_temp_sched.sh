#!/bin/bash
# R30: InfoNCE temperature schedule (4th-component BACKUP candidate).
#   ssl_temp_anneal=0 -> fixed tau (ctrl, backward compatible)
#   1 = linear, 2 = cosine interpolation from ssl_temp_start to ssl_temp_end over
#   ssl_temp_anneal_epochs epochs, then holds. Operates on the InfoNCE *shape*
#   (which negatives dominate) rather than its overall weight (R13/R15 ssl_anneal),
#   hence orthogonal to all 3 fusion components. Static schedule, no learnable params.
# Literature: MM-TS (multi-modal temperature+margin schedules for long-tail CL,
#   arXiv:2603.08202, WACV'26); DySTreSS (dynamically scaled temperature, IEEE TAI'25);
#   RL temperature/curriculum scheduling.
# Smoke test: gowalla embed256 + win30, 30ep each, seed=42.
#   (Under win30 the SSL loss is active only for ep1-30, so the tau window == SSL window.
#    gowalla embed256 peaks ~ep61 per R24, so 30ep shows the early trajectory / direction.)
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r30_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; ep=$4; emb=$5; ta=$6; tae=$7; t0=$8; t1=$9
  log=research_logs/r30_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu ep=$ep tau_anneal=$ta tau=${t0}->${t1} win=${tae} $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$ep --verbose=1 \
    --stop_cnt=1000 --embed_size=$emb --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --ssl_temp_anneal=$ta --ssl_temp_anneal_epochs=$tae --ssl_temp_start=$t0 --ssl_temp_end=$t1 \
    --l2_norm=0 --dropout=0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# gowalla smoke scan (seed=42, 30ep, embed256 + win30). tau schedule over the 30-ep SSL window.
# ctrl + two fixed-tau controls (isolate "just lower/higher tau" from "schedule") + 3 schedules.
run $1 ctrl_tau020   gowalla 30 256 0 0 0.2 0.2
run $1 fix_tau010    gowalla 30 256 0 0 0.1 0.1
run $1 fix_tau030    gowalla 30 256 0 0 0.3 0.3
run $1 cos_020_005   gowalla 30 256 2 30 0.2 0.05   # harden (soft->sharp)
run $1 cos_020_050   gowalla 30 256 2 30 0.2 0.5    # soften (sharp->soft)
run $1 cos_050_005   gowalla 30 256 2 30 0.5 0.05   # strong harden

echo "ALLDONE $(date)" >> $L

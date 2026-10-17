#!/bin/bash
# R28 (4th-component candidate): density-aware (degree-scaled) noise strength.
#   noise_deg_gamma=0  -> uniform noise (ctrl, backward compatible)
#   noise_deg_gamma>0  -> per-node noise magnitude scale_i = 1 + gamma*(c_i - mean(c)),
#                         c_i = deg_i/max(deg); high-degree nodes over-smooth more (E11)
#                         hence get stronger perturbation, mean scale stays ~1.
# Literature: FairDAM (degree-aware masking GCL, 2026), AdaGCL (adaptive aug, SIGIR'23),
#             GCA (adaptive augmentation, 2023).
# Smoke test: gowalla embed256 + win30, 30ep each, seed=42, gamma in {0, 0.25, 0.5, 1.0}.
#   (embed256 gowalla peaks ~ep61 per R24, so 30ep shows the early trajectory / direction.)
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r28_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5; emb=$6; gamma=$7; seed=$8
  log=research_logs/r28_${ds}_${tag}_g${gamma}_s${seed}.log
  echo "START ${ds}_${tag}_g${gamma}_s${seed} gpu=$gpu ep=$eps ssl=$ssl $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=$emb --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=$gamma --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_g${gamma}_s${seed} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_g${gamma}_s${seed} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# gowalla gamma scan (seed=42, 30ep, embed256 + win30)
run $1 gamma gowalla 30 0.1 256 0.0 42
run $1 gamma gowalla 30 0.1 256 0.25 42
run $1 gamma gowalla 30 0.1 256 0.5 42
run $1 gamma gowalla 30 0.1 256 1.0 42

echo "ALLDONE $(date)" >> $L

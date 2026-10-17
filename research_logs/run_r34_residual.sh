#!/bin/bash
# R34 (4th-component candidate #9): GCNII/APPNP-style *static* initial-residual
# connection in the LightGCN propagation.
#   res_alpha=0  -> plain LightGCN propagation (ctrl, backward compatible)
#   res_alpha>0  -> h_{k+1} = (1-alpha)*(A h_k) + alpha*h_0
# Motivation: E11 showed over-smoothing is the core pathology the noise augmentation
# fixes INDIRECTLY (rank release 41%->97%, over-smoothing 0.033->0.003). The initial
# residual fixes it DIRECTLY by keeping the layer-0 personalised signal alive, so it is
# a complementary (not redundant) mechanism. Static & parameter-free (respects the
# R1/R15/R18 'static wins' law and the R4/R6/R7 'keep the propagation operator shared'
# constraint). Literature: GCNII ICML'20 (arXiv:2007.02133) / APPNP ICLR'19
# (arXiv:1810.05997) / Blurring-Sharpening SIGIR'23 / Cascading-Residual-GCN CIKM'23 /
# TSC (two-sided constraint against over-smoothing) KDD'24.
#
# Scan: gowalla embed256 + win30, seed=42, alpha in {0, 0.1, 0.2, 0.3, 0.5}, 60ep
#   (R24 embed256 gowalla peaks ~ep61, so 60ep reaches the real peak; ctrl alpha=0 is a
#   matched same-budget control in the same run).
#   yelp/amazon confirmation is queued separately once the gowalla-best alpha is known
#   (same protocol as R31: gowalla scan first, cross-dataset confirmation after).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r34_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; alpha=$5
  log=research_logs/r34_${ds}_${tag}_a${alpha}.log
  echo "START ${ds}_${tag}_a${alpha} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --res_alpha=$alpha --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_a${alpha} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_a${alpha} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run $1 gowalla res 60 0.0
run $1 gowalla res 60 0.1
run $1 gowalla res 60 0.2
run $1 gowalla res 60 0.3
run $1 gowalla res 60 0.5

echo "ALLDONE $(date)" >> $L

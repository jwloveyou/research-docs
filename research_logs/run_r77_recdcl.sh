#!/bin/bash
# R77 (task #10 收口): RecDCL standalone classic baseline (WWW'24, arXiv:2401.15635)
# -- the LAST missing baseline in the task#10 list (NGCF/LightGCN/DGCF/UltraGCN/SGL/NCL/
# SimGCL/HCCF/DirectAU/LightGCL/DCCF/AdaGCL/XSimGCL/RecDCL).
#
# RecDCL = LightGCN backbone + BPR + BCL (batch-wise CL, SGL-style edge-dropout views)
#          + FCL (feature-wise CL: Barlow-Twins cross-correlation + polynomial-kernel
#          within-group uniformity + momentum cross-modal alignment).
#
# Faithful reproduction note: the FCL machinery (fcl_bt/fcl_poly/fcl_mom + projector/
# predictor/BN, implemented in R8) reproduces the official RecDCL FCL term-by-term; here
# we activate it with the OFFICIAL coefficients (all_bt_coeff=1, bt_coeff=0.01,
# poly_coeff=0.2, a=1, degree=4, polyc=1e-7, mom_coeff=10, momentum=0.3) and the
# SGL-style edge-dropout BCL (view_mix=ee). The fcl_a/polyc/bt_off are hard-coded to the
# official values inside NoiseSGL.py (R8), so only bt/poly/mom coefficients + momentum
# need passing here.
#
# Protocol (matched to R29/R65/R66, same as the other SSL baselines): embed256
# capacity-matched, n_layers=3 (our backbone depth), seed=42, best-over-epochs
# (R@20/NDCG@20 on test), lr=1e-3, batch=1024, reg=1e-4, ssl_reg=0.1 (SGL BCL weight),
# ssl_anneal=0 (annealing is OUR fusion component, NOT RecDCL's), use_mlc=0.
# Budget (SSL baseline, matched to SGL/SimGCL/LightGCL): gowalla 150ep / yelp 50ep /
# amazon 30ep.
#
# HONEST boundary: official RecDCL uses embedding_size=2048 (paper reports on that
# regime); we use 256 for a capacity-matched same-protocol comparison, so RecDCL's
# numbers here are NOT expected to reach the paper's 2048-dim values (R8 already showed
# the FCL gain depends on the 2048-dim setting). This is the same conservative protocol
# used for all other baselines.
#
# Usage: bash run_r77_recdcl.sh <gpu_id>
set -e
GPU="${1:-0}"
cd "$(dirname "$0")/.."
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
PROG=research_logs/r77_recdcl_progress.txt
LOG=research_logs/r77_recdcl_$(date +%H%M%S).log

{
  echo "BEGIN $(date)"
  echo "START gowalla recdcl gpu=$GPU ep=150 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=150 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --view_mix=ee --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_temp=0.2 \
    --ssl_anneal=0 --ssl_anneal_epochs=0 --ssl_anneal_floor=0.0 \
    --fcl_bt=1.0 --fcl_poly=0.2 --fcl_mom=10.0 --fcl_momentum=0.3 \
    --fcl_off=0.0 --reg=1e-4 --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   gowalla recdcl rc=$? $(date)"

  echo "START yelp2018 recdcl gpu=$GPU ep=50 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=50 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --view_mix=ee --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_temp=0.2 \
    --ssl_anneal=0 --ssl_anneal_epochs=0 --ssl_anneal_floor=0.0 \
    --fcl_bt=1.0 --fcl_poly=0.2 --fcl_mom=10.0 --fcl_momentum=0.3 \
    --fcl_off=0.0 --reg=1e-4 --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   yelp2018 recdcl rc=$? $(date)"

  echo "START amazon-book recdcl gpu=$GPU ep=30 $(date)"
  $PY main.py --recommender=NoiseSGL --dataset=amazon-book --seed=42 --epochs=30 \
    --verbose=1 --stop_cnt=1000 --embed_size=256 --view_mix=ee --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_temp=0.2 \
    --ssl_anneal=0 --ssl_anneal_epochs=0 --ssl_anneal_floor=0.0 \
    --fcl_bt=1.0 --fcl_poly=0.2 --fcl_mom=10.0 --fcl_momentum=0.3 \
    --fcl_off=0.0 --reg=1e-4 --lr=0.001 --batch_size=1024 --gpu_id=$GPU
  echo "END   amazon-book recdcl rc=$? $(date)"

  echo "R77_ALLDONE $(date)"
  echo "ALLDONE $(date)"
} > "$PROG" 2> "$LOG"

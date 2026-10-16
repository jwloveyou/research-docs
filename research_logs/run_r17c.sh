#!/bin/bash
# R17c: LightGCL local-global contrast on yelp2018 + amazon-book (single-step, ssl_reg=0.001,
# the gentlest non-zero SSL found non-collapsing on gowalla). Completes the 3-dataset sweep.
# Compare vs established NoiseSGL ctrl: yelp2018 0.06814 / amazon-book 0.04584 (15ep seed42).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r17c_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; ds=$2; shift 2
  log=research_logs/r17c_${tag}.log
  echo "START ${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=15 --verbose=1 --stop_cnt=500 "$@" > $log 2>&1
  echo "END   ${tag} rc=$? $(date)" >> $L
  echo "--- ${tag} peak/final ---" >> $L
  $PY - $log >> $L << "PYEOF"
import re, sys
log = open(sys.argv[1]).read()
bestR = bestN = -1; bestRep = bestNep = -1
finR = finN = None; maxep = 0
for m in re.finditer(r"epoch (\d+):\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", log):
    ep = int(m.group(1)); p, r, n = float(m.group(2)), float(m.group(3)), float(m.group(4))
    if r > bestR: bestR, bestRep = r, ep
    if n > bestN: bestN, bestNep = n, ep
    maxep = max(maxep, ep); finR, finN = r, n
print("peakR@20=%.5f @ep%d | peakN@20=%.5f @ep%d | final(ep%d) R=%.5f N=%.5f" % (bestR, bestRep, bestN, bestNep, maxep, finR, finN))
PYEOF
}

# yelp2018: single-step LightGCL, ssl_reg=0.001 (ctrl=0.06814)
run yelp_ss001 yelp2018 --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.001
# amazon-book: single-step LightGCL, ssl_reg=0.001 (ctrl=0.04584)
run amazon_ss001 amazon-book --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.001

echo ALLDONE $(date) >> $L

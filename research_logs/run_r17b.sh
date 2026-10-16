#!/bin/bash
# R17b: LightGCL local-global contrast, ssl_reg scan. R17a showed ssl_reg=0.1 (single-step)
# collapses (R@20=0.066@15ep vs ctrl 0.1767): the deterministic global view is a strong
# regularizer. The paper's lambda_1 ~ 1e-6 (vs our batch-summed InfoNCE scale) => scan down.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r17b_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; shift
  log=research_logs/r17b_gowalla_${tag}.log
  echo "START ${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=15 --verbose=1 --stop_cnt=500 "$@" > $log 2>&1
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

# reference: ctrl_nn 0.17669@ep12 (R17a same-round); pure LightGCN 15ep 0.1281 (R12, still rising)
run reg01   --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.01
run reg001  --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.001
run reg0001 --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.0001

echo ALLDONE $(date) >> $L

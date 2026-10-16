#!/bin/bash
# R16 baseline calibration: pure LightGCN on gowalla (ssl_reg=0 -> BPR only).
# Critical missing baseline: does LightGCN in THIS repo reach ~0.1830 (paper value)
# at full training, or does the repo's split/eval give a different ceiling?
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r16_lgn_progress.txt
echo "BEGIN $(date)" > $L
log=research_logs/r16_gowalla_lgn_300ep.log
echo "START lgn300 $(date)" >> $L
$PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=300 --verbose=1 --stop_cnt=500 --noise_eps=0.2 --ssl_reg=0 > $log 2>&1
echo "END lgn300 rc=$? $(date)" >> $L
echo "--- best_result ---" >> $L
grep "best_result" -A3 $log | tail -4 >> $L
echo "--- peak/final ---" >> $L
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
echo ALLDONE $(date) >> $L

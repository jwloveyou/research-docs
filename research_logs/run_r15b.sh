#!/bin/bash
# R15b: gowalla win30 at 400ep (head goal: break LightGCN 0.1830 via long pure-BPR fine-tune)
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r15b_progress.txt
echo "BEGIN $(date)" > $L
log=research_logs/r15b_gowalla_win30_400ep.log
echo "START win30_400ep $(date)" >> $L
$PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=400 --verbose=1 --stop_cnt=500 --noise_eps=0.2 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 > $log 2>&1
echo "END   win30_400ep rc=$? $(date)" >> $L
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

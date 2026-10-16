#!/bin/bash
# R15: fixed-window SSL annealing on gowalla (head goal: push R@20 past LightGCN 0.1830).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r15_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; shift
  log=research_logs/r15_gowalla_${tag}.log
  echo "START ${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=150 --verbose=1 --stop_cnt=200 --noise_eps=0.2 "$@" > $log 2>&1
  echo "END   ${tag} rc=$? $(date)" >> $L
  echo "--- ${tag} best_result ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
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

# A = anneal over W=30 then 120ep pure-BPR fine-tune (the NEW mechanism)
# C = anneal over full 150ep (R13 behaviour stretched) -> isolates "window vs just-more-epochs"
# B = anneal over W=20 (window sensitivity)
run win30  --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30
run full150 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=0
run win20  --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=20
echo ALLDONE $(date) >> $L

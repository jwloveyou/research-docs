#!/bin/bash
# R18: gowalla embedding capacity scan (head goal: break LightGCN 0.1830).
# Hypothesis: noise-SSL basin ceiling ~0.180 may be limited by 64-dim capacity.
# Fusion config held fixed: view_mix=nn, noise_eps=0.2, ssl_anneal=2/cos, window=30, floor=0.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r18_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; dim=$2; eps=$3; shift 3
  log=research_logs/r18_gowalla_${tag}.log
  echo "START ${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=$eps \
    --verbose=1 --stop_cnt=500 --embed_size=$dim --noise_eps=0.2 \
    --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 "$@" > $log 2>&1
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

# A = embed 128 (main candidate), 200ep
# B = embed 256 (upper bound), 200ep
run emb128 128 200
run emb256 256 200
echo ALLDONE $(date) >> $L

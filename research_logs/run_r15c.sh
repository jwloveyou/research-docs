#!/bin/bash
# R15c: fixed-window SSL annealing quick check on yelp2018 / amazon-book (15ep, seed=42)
# verify the new ssl_anneal_epochs switch does not hurt the two datasets where it is not needed.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r15c_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; tag=$2; shift 2
  log=research_logs/r15c_${ds}_${tag}.log
  echo "START ${ds}_${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=15 --verbose=1 --stop_cnt=20 --noise_eps=0.2 "$@" > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  echo "--- ${ds}_${tag} best_result ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
  echo "--- ${ds}_${tag} peak ---" >> $L
  $PY - $log >> $L << "PYEOF"
import re, sys
log = open(sys.argv[1]).read()
bestR = bestN = -1; bestRep = bestNep = -1
for m in re.finditer(r"epoch (\d+):\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", log):
    ep = int(m.group(1)); p, r, n = float(m.group(2)), float(m.group(3)), float(m.group(4))
    if r > bestR: bestR, bestRep = r, ep
    if n > bestN: bestN, bestNep = n, ep
print("peakR@20=%.5f @ep%d | peakN@20=%.5f @ep%d" % (bestR, bestRep, bestN, bestNep))
PYEOF
}

# fixed-window anneal (win8) on yelp and amazon; ctrl (anneal=0) for same-session comparison
run yelp2018   win8   --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=8
run yelp2018   ctrl   --ssl_anneal=0
run amazon-book win8  --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=8
run amazon-book ctrl  --ssl_anneal=0
echo ALLDONE $(date) >> $L

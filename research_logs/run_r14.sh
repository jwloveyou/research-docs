#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r14_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; tag=$2; shift 2
  log=research_logs/r14_${ds}_${tag}.log
  echo "START ${ds}_${tag} $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=50 --verbose=1 --stop_cnt=60 --noise_eps=0.2 "$@" > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  echo "--- ${ds}_${tag} best_result ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
  echo "--- ${ds}_${tag} peak/final ---" >> $L
  python3 - $log >> $L << "PYEOF"
import re, sys
log = open(sys.argv[1]).read()
bestR = bestN = -1; bestRep = bestNep = -1
finR = finN = None
for m in re.finditer(r"epoch (\d+):\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", log):
    ep = int(m.group(1)); p, r, n = float(m.group(2)), float(m.group(3)), float(m.group(4))
    if r > bestR: bestR, bestRep = r, ep
    if n > bestN: bestN, bestNep = n, ep
    if ep == 50: finR, finN = r, n
print("peakR@20=%.5f @ep%d | peakN@20=%.5f @ep%d | final(ep50) R=%.5f N=%.5f" % (bestR, bestRep, bestN, bestNep, finR, finN))
PYEOF
}

run yelp2018    ctrl  --ssl_anneal=0
run yelp2018    cos   --ssl_anneal=2 --ssl_anneal_floor=0.0
run amazon-book ctrl  --ssl_anneal=0
run amazon-book cos   --ssl_anneal=2 --ssl_anneal_floor=0.0
echo ALLDONE $(date) >> $L

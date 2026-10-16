#!/bin/bash
# R18 seeds: 3-seed robustness for gowalla embed_size=128 (fusion change candidate).
# Waits for the emb128 200ep seed42 run to finish, then runs seed 43/44 (150ep),
# then quick yelp/amazon embed128 15ep regression checks.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r18_seeds_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; seed=$2; ds=$3; eps=$4; dim=$5; shift 5
  log=research_logs/r18_${ds}_${tag}.log
  echo "START ${ds}_${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=$eps \
    --verbose=1 --stop_cnt=500 --embed_size=$dim --noise_eps=0.2 \
    --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 "$@" > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  echo "--- ${ds}_${tag} best_result ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
  echo "--- ${ds}_${tag} peak/final ---" >> $L
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

# wait for the in-flight emb128 200ep seed42 run to finish (grep best_result line)
while ! grep -q "best_result" research_logs/r18_gowalla_emb128.log 2>/dev/null; do
  sleep 30
done
echo "emb128 seed42 done $(date)" >> $L
grep "best_result" -A3 research_logs/r18_gowalla_emb128.log | tail -4 >> $L

run emb128_s43 43 gowalla 150 128
run emb128_s44 44 gowalla 150 128

# quick regression checks: embed128 must not hurt yelp/amazon (already > baseline at 64)
run emb128 42 yelp2018 15 128
run emb128 42 amazon-book 15 128
echo ALLDONE $(date) >> $L

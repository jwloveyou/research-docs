#!/bin/bash
# R16b: ssl_reg scan on gowalla (win30 anneal). R15's priority-#1 hypothesis:
# the noise SSL's early (ep1-30) influence locks the representation into a ~0.180 basin;
# a WEAKER ssl_reg may leave a higher-peak basin for the subsequent pure-BPR fine-tune.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r16b_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; shift
  log=research_logs/r16b_gowalla_${tag}.log
  echo "START ${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=100 --verbose=1 --stop_cnt=200 --noise_eps=0.2 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 "$@" > $log 2>&1
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

# reference: ssl_reg=0.1 win30 -> peak 0.17926@ep92 (R15)
run reg005 --ssl_reg=0.05
run reg002 --ssl_reg=0.02

echo ALLDONE $(date) >> $L

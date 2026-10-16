#!/bin/bash
# R17: original LightGCL local-vs-global contrast (arXiv:2302.08191) on gowalla.
# R7 tested ONLY "noise view + SVD global view" mix (ng/nng), never the original
# "clean local view vs SVD global view" contrast. This round implements it faithfully:
#   global view = U_q S_q V_q^T applied to the local embeddings (bipartite SVD),
#   InfoNCE = local z_l vs global g_l (per-layer, paper eq.6, or single-step final).
# Budget: matched 15ep first (same as R1-R16 cross-method comparison), then longer.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r17_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; shift
  log=research_logs/r17_gowalla_${tag}.log
  echo "START ${tag} $(date)" >> $L
  $PY main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=15 --verbose=1 --stop_cnt=500 "$@" > $log 2>&1
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

# 1) same-round control: NoiseSGL nn, eps=0.2, ssl_reg=0.1 -> expect ~0.1767 (R12/R16)
run ctrl_nn      --noise_eps=0.2 --ssl_reg=0.1 --lightgcl=0
# 2) LightGCL single-step (final local vs global), ssl_reg=0.1 (default scale)
run lg_ss_01     --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.1
# 3) LightGCL single-step, weaker SSL (match ctrl SSL magnitude ~1.15 vs 1.71)
run lg_ss_007    --lightgcl=1 --lightgcl_perlayer=0 --ssl_reg=0.07
# 4) LightGCL per-layer (paper eq.6), SSL-strength-matched (per-layer loss ~6x -> /6)
run lg_pl_002    --lightgcl=1 --lightgcl_perlayer=1 --ssl_reg=0.02

echo ALLDONE $(date) >> $L

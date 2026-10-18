#!/usr/bin/env bash
# R77: queue the RecDCL standalone baseline (task #10 LAST missing baseline) on GPU7
# AFTER the current GPU7 chain (baseline p3 -> HCCF -> AdaGCL -> DGCF -> NCL -> DirectAU)
# completes. Two-gate guard per the R52 lesson:
#   1. wait for the DirectAU marker (r71_directau_progress.txt contains "ALLDONE");
#   2. wait for GPU7 to be truly idle (<1000 MiB, 2 consecutive checks ~60s apart).
# Then run run_r77_recdcl.sh 7 (gowalla 150ep / yelp 50ep / amazon 30ep).
set -u
cd "$(dirname "$0")"
MARK=r71_directau_progress.txt

echo "wait_r77_recdcl_gpu7: waiting for DirectAU ALLDONE ($MARK)"
while ! grep -q "ALLDONE" "$MARK" 2>/dev/null; do
  sleep 180
done
echo "wait_r77_recdcl_gpu7: DirectAU done, waiting for GPU7 idle"

while :; do
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo 99999)
  sleep 60
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo 99999)
  if [ "$u1" -lt 1000 ] && [ "$u2" -lt 1000 ]; then
    break
  fi
done
echo "wait_r77_recdcl_gpu7: GPU7 idle, launching RecDCL baseline"
bash run_r77_recdcl.sh 7

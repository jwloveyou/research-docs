#!/usr/bin/env bash
# R71: queue the DirectAU standalone baseline (task #10 last missing classic baseline)
# on GPU7 AFTER the current GPU7 chain (baseline p3 -> HCCF -> AdaGCL -> DGCF -> NCL)
# completes. Two-gate guard per the R52 lesson:
#   1. wait for the NCL marker (r70_ncl_progress.txt contains "ALLDONE");
#   2. wait for GPU7 to be truly idle (<1000 MiB, 2 consecutive checks ~60s apart).
# Then run run_r71_directau_scnumath.sh 7.
set -u
cd "$(dirname "$0")"
MARK=research_logs/r70_ncl_progress.txt

echo "wait_r71_directau_gpu7: waiting for NCL_ALLDONE ($MARK)"
while ! grep -q "ALLDONE" "$MARK" 2>/dev/null; do
  sleep 180
done
echo "wait_r71_directau_gpu7: NCL done, waiting for GPU7 idle"

while :; do
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo 99999)
  sleep 60
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo 99999)
  if [ "$u1" -lt 1000 ] && [ "$u2" -lt 1000 ]; then
    break
  fi
done
echo "wait_r71_directau_gpu7: GPU7 idle, launching DirectAU baseline"
bash run_r71_directau_scnumath.sh 7

#!/bin/bash
# R69 DGCF watcher: after the AdaGCL baseline (task #8, third No.1) finishes on GPU7, run
# the DGCF classic baseline (task #10) on GPU7. Guard: "ADAGCL_ALLDONE" must be present in
# the AdaGCL progress file AND GPU7 must be free for 2 consecutive checks. Marker matching
# uses plain strings (no dots); idle threshold uses <1000MiB (R52 lesson: small-graph runs
# occupy 1400-2900MiB, so <3000 would misjudge them as free).
cd ~/SGL-Torch
while true; do
  done_adagcl=$(grep -c "ADAGCL_ALLDONE" research_logs/r_adagcl_progress.txt 2>/dev/null || echo 0)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  if [ "$done_adagcl" -ge 1 ] && [ "${m1:-99999}" -lt 1000 ] && [ "${m2:-99999}" -lt 1000 ]; then break; fi
done
bash research_logs/run_r69_dgcf.sh 7 > research_logs/r69_dgcf_launcher.out 2>&1

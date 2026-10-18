#!/bin/bash
# R70 NCL watcher: after the DGCF classic baseline (task #10) finishes on GPU7, run the
# NCL classic baseline (task #10, WWW'22) on GPU7. Guard: "DGCF_ALLDONE" must be present
# in the DGCF progress file AND GPU7 must be free for 2 consecutive checks. Marker matching
# uses plain strings (no dots); idle threshold uses <1000MiB (R52 lesson).
cd ~/SGL-Torch
while true; do
  done_dgcf=$(grep -c "DGCF_ALLDONE" research_logs/r69_dgcf_progress.txt 2>/dev/null || echo 0)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  if [ "$done_dgcf" -ge 1 ] && [ "${m1:-99999}" -lt 1000 ] && [ "${m2:-99999}" -lt 1000 ]; then break; fi
done
bash research_logs/run_r70_ncl.sh 7 > research_logs/r70_ncl_launcher.out 2>&1

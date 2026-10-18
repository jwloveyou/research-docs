#!/bin/bash
# R51: queue the embed256 hyper-parameter sensitivity scan (noise eps + SSL anneal
# window) on GPU6 after the R47 frozen-LLM semantic quick-signal finishes.
# Two gates (avoids GPU contention):
#   1) R47 semantic quick-signal done (marker r47_sem_gpu6_all_done.txt).
#   2) GPU6 idle for 2 consecutive samples (<3000MiB, ~120s apart).
# Marker matching uses the plain string (no leading/trailing dots, per the R34 fix).
cd ~/SGL-Torch
while true; do
  if grep -q 'R47_SEM_ALLDONE' research_logs/r47_sem_gpu6_all_done.txt 2>/dev/null; then
    m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    sleep 120
    m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    if [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ]; then break; fi
  fi
  sleep 180
done
bash research_logs/run_r51_sensitivity.sh 6 > research_logs/r51_launcher.out 2>&1
echo "R51_ALLDONE $(date)" > research_logs/r51_gpu6_all_done.txt

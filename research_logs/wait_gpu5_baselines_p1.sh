#!/bin/bash
# R29: after yelp_full fusion->lgn finishes on GPU5, run SOTA-baseline part 1 on GPU5.
cd ~/SGL-Torch
while true; do
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null)
  if [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ]; then break; fi
done
bash research_logs/run_baselines_seed42.sh 5 1 > research_logs/r29_baselines_p1_launcher.out 2>&1

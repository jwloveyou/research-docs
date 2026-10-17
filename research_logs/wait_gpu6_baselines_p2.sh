#!/bin/bash
# R29: after tmall + yelp_full-emb128 finish on GPU6, run SOTA-baseline part 2 on GPU6.
cd ~/SGL-Torch
while true; do
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
  if [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ]; then break; fi
done
bash research_logs/run_baselines_seed42.sh 6 2 > research_logs/r29_baselines_p2_launcher.out 2>&1

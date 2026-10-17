#!/bin/bash
# R29: after beauty->kindle->r28 gamma->r28 ablation finish on GPU7, run baseline part 3.
# Guard: also require r28 ablation ALLDONE so we never collide with the r28 chain tail.
cd ~/SGL-Torch
while true; do
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  done_r28=$(grep -c 'ALLDONE' research_logs/r28_ablation_progress.txt 2>/dev/null)
  if [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ] && [ "$done_r28" -ge 1 ]; then break; fi
done
bash research_logs/run_baselines_seed42.sh 7 3 > research_logs/r29_baselines_p3_launcher.out 2>&1

#!/bin/bash
# R35 watcher: after the GPU4 chain (r31c -> r32 -> r33 -> r31b -> r34) finishes, run
# the DirectSpec-style spectral-balancing scan (4th-component candidate #10) on GPU4.
# GPU4 is only used by that chain, so no collision with GPU5/6/7.
cd ~/SGL-Torch
while true; do
  done=$(grep -c R34_ALLDONE research_logs/r34_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r35_spec_bal.sh 4 > research_logs/r35_launcher.out 2>&1
echo "R35_ALLDONE $(date)" > research_logs/r35_gpu4_all_done.txt

#!/bin/bash
# R34 watcher: after the GPU4 chain (r31c -> r32 -> r33 -> r31b) finishes, run the
# GCNII/APPNP static initial-residual scan (4th-component candidate #9) on GPU4.
# GPU4 is only used by that chain, so no collision with GPU5/6/7.
cd ~/SGL-Torch
while true; do
  done=$(grep -c R33_ALLDONE research_logs/r33_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r34_residual.sh 4 > research_logs/r34_launcher.out 2>&1
echo "R34_ALLDONE $(date)" > research_logs/r34_gpu4_all_done.txt

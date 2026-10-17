#!/bin/bash
# R38 watcher: after the GPU4 chain (r31c -> r32 -> r33 -> r31b -> r34 -> r35 -> r37)
# finishes, run the regularisation-axis scan (4th-component candidate #12) on GPU4.
# GPU4 is only used by that chain, so no collision with GPU5/6/7.
# Marker matching uses the plain R37_ALLDONE string (no leading/trailing dots).
cd ~/SGL-Torch
while true; do
  done=$(grep -c R37_ALLDONE research_logs/r37_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r38_reg_wd.sh 4 > research_logs/r38_launcher.out 2>&1
echo "R38_ALLDONE $(date)" > research_logs/r38_gpu4_all_done.txt

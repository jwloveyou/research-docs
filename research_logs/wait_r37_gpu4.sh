#!/bin/bash
# R37 watcher: after the GPU4 chain (r31c -> r32 -> r33 -> r31b -> r34 -> r35) finishes,
# run the learning-rate schedule scan (4th-component candidate #11) on GPU4.
# GPU4 is only used by that chain, so no collision with GPU5/6/7.
# Marker matching uses the plain R35_ALLDONE string (no leading/trailing dots).
cd ~/SGL-Torch
while true; do
  done=$(grep -c R35_ALLDONE research_logs/r35_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r37_lr_scan.sh 4 > research_logs/r37_launcher.out 2>&1
echo "R37_ALLDONE $(date)" > research_logs/r37_gpu4_all_done.txt

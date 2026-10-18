#!/bin/bash
# R40 watcher: after the GPU4 chain (… -> r39 layer-agg) finishes, run the SWA scan
# (4th-component candidate #14) on GPU4. GPU4 is only used by that chain, so no
# collision with GPU5/6/7. Marker matching uses the plain R39_ALLDONE string.
cd ~/SGL-Torch
while true; do
  done=$(grep -c R39_ALLDONE research_logs/r39_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r40_swa.sh 4 > research_logs/r40_launcher.out 2>&1
echo "R40_ALLDONE $(date)" > research_logs/r40_gpu4_all_done.txt

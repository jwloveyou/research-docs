#!/bin/bash
# R78: queue the embed512 3-seed verification on GPU6 after the R51 embed256
# sensitivity scan finishes (R51 is itself queued after the R47 semantic quick-signal).
#
# Two gates (avoids the R52-style contention bug):
#   1) R51 sensitivity done (marker research_logs/r51_gpu6_all_done.txt contains R51_ALLDONE).
#   2) GPU6 truly idle for 2 consecutive samples (<1000 MiB, ~120s apart).
# Marker matching uses plain strings (no leading/trailing dots, per the R34 watcher fix).
cd ~/SGL-Torch
while true; do
  if grep -q 'R51_ALLDONE' research_logs/r51_gpu6_all_done.txt 2>/dev/null; then
    m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    sleep 120
    m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    if [ "${m1:-99999}" -lt 1000 ] && [ "${m2:-99999}" -lt 1000 ]; then break; fi
  fi
  sleep 180
done
bash research_logs/run_r78_embed512_3seed.sh 6 > research_logs/r78_launcher.out 2>&1
echo "R78_ALLDONE $(date)" > research_logs/r78_gpu6_all_done.txt

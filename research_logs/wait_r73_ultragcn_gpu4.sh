#!/bin/bash
# R73 watcher: after the whole GPU4 4th-component chain finishes (the last stage is the
# r60 init-scale scan, which writes R60_ALLDONE into r60_gpu4_all_done.txt), run the
# UltraGCN R=300 faithful re-run on GPU4. Double gate (R52 lesson): (1) R60_ALLDONE marker
# AND (2) GPU4 truly idle (<1000MiB; small-graph embed256 tasks occupy ~1400-2900MiB).
cd ~/SGL-Torch
while true; do
  done=$(grep -c R60_ALLDONE research_logs/r60_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -lt 1 ]; then sleep 300; continue; fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
  mem=${mem:-99999}
  if [ "$mem" -lt 1000 ]; then break; fi
  sleep 300
done
bash research_logs/run_r73_ultragcn_r300.sh 4 > research_logs/r73_ultragcn_r300_launcher.out 2>&1
echo "R73_ALLDONE $(date)" > research_logs/r73_gpu4_all_done.txt

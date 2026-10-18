#!/bin/bash
# R60 watcher: after the full GPU4 4th-component chain finishes (r58_master -> r38 -> r39
# -> r40 -> r41 -> r49 DropEdge), run the init-scale scan (4th-component candidate #18) on
# GPU4. GPU4 is only used by that chain, so no collision with GPU5/6/7. Double gate (per
# the R52 lesson): (1) R49_ALLDONE marker present in r49_progress.txt AND (2) GPU4 truly
# idle (<1000MiB, since small-graph embed256 tasks occupy only ~1400-2900MiB and would be
# misread as idle by a <3000MiB threshold).
cd ~/SGL-Torch
while true; do
  done=$(grep -c R49_ALLDONE research_logs/r49_progress.txt 2>/dev/null || echo 0)
  if [ "$done" -lt 1 ]; then sleep 300; continue; fi
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | head -1)
  mem=${mem:-99999}
  if [ "$mem" -lt 1000 ]; then break; fi
  sleep 300
done
bash research_logs/run_r60_initscale.sh 4 > research_logs/r60_launcher.out 2>&1
echo "R60_ALLDONE $(date)" > research_logs/r60_gpu4_all_done.txt

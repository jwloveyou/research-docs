#!/bin/bash
# R49 watcher: after the GPU4 4th-component chain finishes (r31c -> r34 -> r35 -> r32 ->
# r33 -> r31b -> r37 -> r38 -> r39 -> r40 -> r41), run the DropEdge scan (4th-component
# candidate #16) on GPU4. GPU4 is only used by that chain. Marker matching uses the plain
# R41_ALLDONE string (no leading/trailing dots, per the R34 watcher fix).
cd ~/SGL-Torch
while true; do
  done=$(grep -c R41_ALLDONE research_logs/r41_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r49_dropedge.sh 4 > research_logs/r49_launcher.out 2>&1

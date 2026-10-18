#!/bin/bash
# R42 watcher: after SOTA-baseline part 1 (gowalla LightGCN/SGL/SimGCL) finishes on GPU5,
# run the dense-dataset generalization-boundary runs (jester + ml-10m + ml-1m) on GPU5.
# GPU5 is only used by that chain (p1), so no collision with GPU4/6/7.
# Marker matching uses the plain "DONE part=1" string (no leading/trailing dots).
cd ~/SGL-Torch
while true; do
  done_p1=$(grep -c "DONE part=1" research_logs/r29_baselines_progress.txt 2>/dev/null)
  done_p1=${done_p1:-0}
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null)
  m1=${m1:-99999}
  sleep 120
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null)
  m2=${m2:-99999}
  if [ "$done_p1" -ge 1 ] && [ "$m1" -lt 3000 ] && [ "$m2" -lt 3000 ]; then break; fi
done
bash research_logs/run_r42_dense.sh 5 > research_logs/r42_launcher.out 2>&1
echo "R42_ALLDONE $(date)" > research_logs/r42_gpu5_all_done.txt

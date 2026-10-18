#!/bin/bash
# HCCF baseline watcher: after SOTA-baseline part 3 finishes on GPU7, run the HCCF
# baseline (task #8) on GPU7. Guard: "DONE part=3" must be present in the baseline
# progress file AND GPU7 must be free for 2 consecutive checks (avoids the transient
# free window between chained runs). Marker matching uses plain strings (no dots).
cd ~/SGL-Torch
while true; do
  done_p3=$(grep -c "DONE part=3" research_logs/r29_baselines_progress.txt 2>/dev/null || echo 0)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  if [ "$done_p3" -ge 1 ] && [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ]; then break; fi
done
bash research_logs/run_hccf_baseline.sh 7 > research_logs/r_hccf_launcher.out 2>&1

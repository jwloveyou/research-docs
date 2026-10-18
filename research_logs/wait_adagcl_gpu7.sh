#!/bin/bash
# AdaGCL baseline watcher: after the HCCF baseline (task #8) finishes on GPU7, run the
# AdaGCL baseline (third No.1) on GPU7. Guard: "HCCF_ALLDONE" must be present in the HCCF
# progress file AND GPU7 must be free for 2 consecutive checks. Marker matching uses plain
# strings (no dots).
cd ~/SGL-Torch
while true; do
  done_hccf=$(grep -c "HCCF_ALLDONE" research_logs/r_hccf_progress.txt 2>/dev/null || echo 0)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  sleep 90
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null)
  if [ "$done_hccf" -ge 1 ] && [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ]; then break; fi
done
bash research_logs/run_adagcl_baseline.sh 7 > research_logs/r_adagcl_launcher.out 2>&1

#!/bin/bash
# R52: corrected re-queue of the R47 frozen-LLM semantic quick-signal on GPU6.
# Root cause of the premature fire in wait_r47_sem_gpu6.sh (R48): the idle gate used
# "<3000 MiB", but XSimGCL-gowalla (baseline part 2) only occupies ~2300 MiB, so the
# watcher misread GPU6 as idle and launched the semantic quick-signal on top of the
# running XSimGCL (two jobs contending for one card, results of both would be polluted).
# Fix: (1) additionally gate on baseline part-2 completion ("DONE part=2" marker);
#      (2) tighten the idle threshold to <1000 MiB (a truly idle card shows ~0-500 MiB,
#          while the smallest real run we launch uses >=~1442 MiB, so 1000 separates them).
# Marker matching uses plain strings (no leading/trailing dots, per the R34 watcher fix).
cd ~/SGL-Torch
while true; do
  done_p2=$(grep -c "DONE part=2" research_logs/r29_baselines_progress.txt 2>/dev/null || echo 0)
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
  sleep 120
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
  if [ "$done_p2" -ge 1 ] && [ "${m1:-99999}" -lt 1000 ] && [ "${m2:-99999}" -lt 1000 ]; then break; fi
  sleep 180
done
bash research_logs/run_r47_sem_quick.sh 6 > research_logs/r47_sem_launcher_v2.out 2>&1
echo "R47_SEM_ALLDONE $(date)" > research_logs/r47_sem_gpu6_all_done.txt

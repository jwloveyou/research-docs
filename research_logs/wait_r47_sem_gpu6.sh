#!/bin/bash
# R48: queue the R47 frozen-LLM semantic ablation on GPU6 after baseline part 2 finishes.
# Two gates before running (avoids wasting GPU on stale features):
#   1) LLM enrichment ALLDONE  -> ml-1m/lastfm item_sem.npy have been rebuilt with
#      native + DeepSeek LLM tags (higher effective rank than the 18-genre-only version).
#   2) GPU6 idle (2 consecutive <3000MiB samples) -> baseline p2 (LightGCL/XSimGCL/LGN)
#      has finished and released the card.
# Then run the staged semantic ablation:
#   * lastfm-2k 100ep quick signal: sem0 (ctrl) vs sem1 lam=1.0
#   * ml-1m     100ep quick signal: sem0 (ctrl) vs sem1 lam=1.0
#   (full 200ep + lam scan is left to the next agent based on this signal.)
cd ~/SGL-Torch
while true; do
  if grep -q '\[r47-llm\] ALLDONE' research_logs/r47_llm_enrich.out 2>/dev/null; then
    m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    sleep 120
    m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null)
    if [ "${m1:-99999}" -lt 3000 ] && [ "${m2:-99999}" -lt 3000 ]; then break; fi
  fi
  sleep 180
done
bash research_logs/run_r47_sem_quick.sh 6 > research_logs/r47_sem_launcher.out 2>&1
echo "R47_SEM_ALLDONE $(date)" > research_logs/r47_sem_gpu6_all_done.txt

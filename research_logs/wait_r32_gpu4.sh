#!/bin/bash
# R32 watcher: after r31c (gowalla gamma 100ep review, GPU4) finishes, run the
# generalization capacity-fix experiments on GPU4:
#   1) ifashion embed64/128 (fusion + lgn)
#   2) yelp_full lgn-emb256 (missing baseline) + emb64 pair
# GPU4 is only used by the r31c chain, so this does not collide with the GPU5/6/7 chains.
cd ~/SGL-Torch
while true; do
  done=$(grep -c .ALLDONE. research_logs/r31cg4_progress.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r32_ifashion_capacity.sh > research_logs/r32_ifashion_launcher.out 2>&1
bash research_logs/run_r32_yelp_full_emb64.sh > research_logs/r32_yelp_full_launcher.out 2>&1
echo "R32_ALLDONE $(date)" > research_logs/r32_gpu4_all_done.txt

#!/bin/bash
# R48: re-prioritized GPU4 chain.
#   r31c (density-noise 100ep, running) -> r34 (initial residual) -> r35 (spectral balance)
#   -> r32 (ifashion/yelp_full capacity fix) -> r33+r31b (layer noise + tailboost)
#   -> r37 (LR schedule) -> (r38/r39/r40/r41 unchanged, triggered by existing wait_r38).
# Rationale: task#1 (4th component) is top priority; r34/r35 are the two highest-prior
# 4th-component candidates (E11-aligned), so they run before the generalization fixes.
# Marker matching uses plain strings (no leading/trailing dots) per the R34 watcher fix.
cd ~/SGL-Torch

echo "[r48-master] waiting for r31c ALLDONE $(date)"
while true; do
  if grep -q ALLDONE research_logs/r31cg4_progress.txt 2>/dev/null; then break; fi
  sleep 180
done

echo "[r48-master] r34 initial residual $(date)"
bash research_logs/run_r34_residual.sh 4 > research_logs/r34_launcher.out 2>&1
echo "R34_ALLDONE $(date)" > research_logs/r34_gpu4_all_done.txt

echo "[r48-master] r35 spectral balance $(date)"
bash research_logs/run_r35_spec_bal.sh 4 > research_logs/r35_launcher.out 2>&1
echo "R35_ALLDONE $(date)" > research_logs/r35_gpu4_all_done.txt

echo "[r48-master] r32 capacity fix (ifashion + yelp_full) $(date)"
bash research_logs/run_r32_ifashion_capacity.sh > research_logs/r32_ifashion_launcher.out 2>&1
bash research_logs/run_r32_yelp_full_emb64.sh > research_logs/r32_yelp_full_launcher.out 2>&1
echo "R32_ALLDONE $(date)" > research_logs/r32_gpu4_all_done.txt

echo "[r48-master] r33 layer noise + r31b tailboost $(date)"
bash research_logs/run_r33_layer_noise.sh 4 > research_logs/r33_launcher.out 2>&1
bash research_logs/run_r31b_gpu4_tailboost.sh 4 > research_logs/r33b_launcher.out 2>&1
echo "R33_ALLDONE $(date)" > research_logs/r33_gpu4_all_done.txt

echo "[r48-master] r37 lr schedule $(date)"
bash research_logs/run_r37_lr_scan.sh 4 > research_logs/r37_launcher.out 2>&1
echo "R37_ALLDONE $(date)" > research_logs/r37_gpu4_all_done.txt

echo "[r48-master] ALLDONE $(date)"

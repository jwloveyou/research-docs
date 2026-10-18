#!/bin/bash
# R58 master: re-prioritized GPU4 chain (replaces the R48 master, which is killed because
# r34 initial-residual is already clearly negative/slow-convergent at alpha=0.1 — see
# r34_progress.txt — so alphas 0.2/0.3/0.5 are skipped to save ~3h and reach the two
# E11-aligned 4th-component candidates faster).
#
#   r35 (spectral balance, candidate #10) -> r58 (item-side noise boost, candidate #17)
#   -> r32 (ifashion/yelp_full capacity fix) -> r33+r31b (layer noise + tailboost)
#   -> r37 (LR schedule) -> R37_ALLDONE (downstream r38..r49 watchers unchanged).
#
# Marker matching uses plain strings (no leading/trailing dots).
cd ~/SGL-Torch

echo "[r58-master] r35 spectral balance $(date)"
bash research_logs/run_r35_spec_bal.sh 4 > research_logs/r35_launcher.out 2>&1
echo "R35_ALLDONE $(date)" > research_logs/r35_gpu4_all_done.txt

echo "[r58-master] r58 item-side noise boost $(date)"
bash research_logs/run_r58_itemboost.sh 4 > research_logs/r58_launcher.out 2>&1
echo "R58_ALLDONE $(date)" > research_logs/r58_gpu4_all_done.txt

echo "[r58-master] r32 capacity fix (ifashion + yelp_full) $(date)"
bash research_logs/run_r32_ifashion_capacity.sh 4 > research_logs/r32_ifashion_launcher.out 2>&1
bash research_logs/run_r32_yelp_full_emb64.sh 4 > research_logs/r32_yelp_full_launcher.out 2>&1
echo "R32_ALLDONE $(date)" > research_logs/r32_gpu4_all_done.txt

echo "[r58-master] r33 layer noise + r31b tailboost $(date)"
bash research_logs/run_r33_layer_noise.sh 4 > research_logs/r33_launcher.out 2>&1
bash research_logs/run_r31b_gpu4_tailboost.sh 4 > research_logs/r33b_launcher.out 2>&1
echo "R33_ALLDONE $(date)" > research_logs/r33_gpu4_all_done.txt

echo "[r58-master] r37 lr schedule $(date)"
bash research_logs/run_r37_lr_scan.sh 4 > research_logs/r37_launcher.out 2>&1
echo "R37_ALLDONE $(date)" > research_logs/r37_gpu4_all_done.txt

echo "[r58-master] ALLDONE $(date)"

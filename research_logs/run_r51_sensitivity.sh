#!/bin/bash
# R51: hyper-parameter sensitivity curves at the FINAL fusion config (embed256).
#
# Gap being filled (task #2 "超参敏感性"): the two key fusion hyper-parameters
#   (a) noise magnitude  noise_eps   (R16 scanned {0.05,0.1,0.2,0.3} only at embed64)
#   (b) SSL anneal window ssl_anneal_epochs (R15 tuned win30 only at embed64)
# were tuned at embed64, but the final fusion model runs embed_size=256 (R24).
# At embed256 the convergence geometry changed materially (gowalla peak moved from
# ep195@emb128 to ep61@emb256 with a faster post-peak decline, R24), so the optimal
# eps / anneal window may have shifted. This script produces the missing sensitivity
# curves at the exact fusion protocol.
#
# Protocol: identical to the fusion model (view_mix=nn, ssl_reg=0.1, ssl_anneal=2
# cosine->floor=0, embed_size=256, seed=42, best-over-epochs on the test set).
#   gowalla 60ep reaches the real peak (ep61) -> the decisive curves.
#   yelp 20ep / amazon 12ep match the R24 budgets (their peaks are ep5-7 / ep11-12).
# Control (eps=0.2 & win=30) is re-run in the SAME batch for a matched-budget
# comparison (R23/R24 control = 0.1880 / 0.0709 / 0.0551).
#
# Run: bash research_logs/run_r51_sensitivity.sh <gpu_id>
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r51_sensitivity_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; window=$5; nepoch=$6
  log=research_logs/r51_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu eps=$eps win=$window ep=$nepoch $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$nepoch --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=$eps \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=$window \
    --l2_norm=0 --dropout=0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# --- gowalla (decisive): eps curve {0.1, 0.2, 0.3} + anneal-window curve {15, 30, 60} ---
# shared control = (eps=0.2, win=30) -> 5 distinct runs
run $1 gowalla eps01_win30 0.1 30 60
run $1 gowalla ctrl_eps02_win30 0.2 30 60
run $1 gowalla eps03_win30 0.3 30 60
run $1 gowalla eps02_win15 0.2 15 60
run $1 gowalla eps02_win60 0.2 60 60

# --- yelp / amazon (confirm flatness at embed256, matched R24 budgets) ---
run $1 yelp2018 eps01_win30 0.1 30 20
run $1 yelp2018 eps03_win30 0.3 30 20
run $1 amazon-book eps01_win30 0.1 30 12
run $1 amazon-book eps03_win30 0.3 30 12

echo "ALLDONE $(date)" >> $L

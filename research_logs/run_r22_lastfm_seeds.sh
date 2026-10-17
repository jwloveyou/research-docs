#!/bin/bash
# R22b: 3-seed confirmation of the lastfm-2k positive result (fusion vs LightGCN).
# seed43 on GPU6, seed44 on GPU7 (GPU4/5 still busy with ml-1m). 200ep each.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r22_lastfm_seeds_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; seed=$3; ssl=$4
  ds=lastfm-2k
  log=research_logs/r22_${ds}_${tag}_s${seed}.log
  echo "START ${ds}_${tag}_s${seed} gpu=$gpu $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=$seed --epochs=200 --verbose=1 --stop_cnt=1000 --embed_size=128 \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --gpu_id=$gpu > $log 2>&1
  echo "END   ${ds}_${tag}_s${seed} rc=$? $(date)" >> $L
  echo "--- ${ds}_${tag}_s${seed} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

( run 6 fusion 43 0.1; run 6 fusion 44 0.1 ) &
( run 7 lgn 43 0; run 7 lgn 44 0 ) &
wait
echo "ALLDONE $(date)" >> $L

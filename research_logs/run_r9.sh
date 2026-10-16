#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r9_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/r9_$1.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
  grep "best_result" -A2 research_logs/r9_$1.log | tail -3 >> $L
}
run ctrl ""
run align01 "--directau_align=0.1"
run align05 "--directau_align=0.5"
run align05_unif01 "--directau_align=0.5 --directau_unif=0.1"
echo ALLDONE >> $L

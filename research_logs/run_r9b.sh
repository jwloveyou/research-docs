#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r9b_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=$3 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/r9b_$1.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
  grep "best_result" -A2 research_logs/r9b_$1.log | tail -3 >> $L
}
run align005 "--directau_align=0.05" 42
run align02  "--directau_align=0.2"  42
run align01_s43 "--directau_align=0.1" 43
run align01_s44 "--directau_align=0.1" 44
run align01_unif05 "--directau_align=0.1 --directau_unif=0.05" 42
echo ALLDONE >> $L

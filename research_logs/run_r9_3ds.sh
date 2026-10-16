#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r9_3ds_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=$1 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 --directau_align=0.1 > research_logs/r9_3ds_$1_align01.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
  grep "best_result" -A2 research_logs/r9_3ds_$1_align01.log | tail -3 >> $L
}
run gowalla
run amazon-book
echo ALLDONE >> $L

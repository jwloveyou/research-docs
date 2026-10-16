#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r11_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/r11_$1.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
  grep "best_result" -A2 research_logs/r11_$1.log | tail -3 >> $L
}
run ctrl ""
run lam01 "--proto_K=1000 --proto_lam=0.1 --proto_interval=1"
run lam02 "--proto_K=1000 --proto_lam=0.2 --proto_interval=1"
run lam005 "--proto_K=1000 --proto_lam=0.05 --proto_interval=1"
echo ALLDONE >> $L

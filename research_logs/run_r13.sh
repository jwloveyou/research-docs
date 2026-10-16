#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r13_progress.txt
echo "BEGIN $(date)" > $L
run () {
  tag=$1; shift
  echo "START $tag $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=50 --verbose=1 --stop_cnt=60 --noise_eps=0.2 "$@" > research_logs/r13_$tag.log 2>&1
  echo "END   $tag rc=$? $(date)" >> $L
  echo "--- $tag best ---" >> $L
  grep "best_result" -A3 research_logs/r13_$tag.log | tail -4 >> $L
}
run ctrl  --ssl_anneal=0
run lin   --ssl_anneal=1 --ssl_anneal_floor=0.0
run cos   --ssl_anneal=2 --ssl_anneal_floor=0.0
echo ALLDONE $(date) >> $L

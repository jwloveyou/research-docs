#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r13b_progress.txt
echo "BEGIN $(date)" > $L
run () {
  tag=$1; shift
  echo "START $tag $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=gowalla --seed=$tag --epochs=50 --verbose=1 --stop_cnt=60 --noise_eps=0.2 --ssl_anneal=2 --ssl_anneal_floor=0.0 > research_logs/r13b_cos_s$tag.log 2>&1
  echo "END   s$tag rc=$? $(date)" >> $L
  echo "--- cos s$tag best ---" >> $L
  grep "best_result" -A3 research_logs/r13b_cos_s$tag.log | tail -4 >> $L
  echo "--- cos s$tag ep50 ---" >> $L
  grep -oE "epoch 50:.*" research_logs/r13b_cos_s$tag.log >> $L
}
run 43
run 44
echo ALLDONE $(date) >> $L

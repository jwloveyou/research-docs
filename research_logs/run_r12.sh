#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r12_progress.txt
echo "BEGIN $(date)" > $L
run () {
  tag=$1; shift
  echo "START $tag $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 "$@" > research_logs/r12_$tag.log 2>&1
  echo "END   $tag rc=$? $(date)" >> $L
  grep "best_result" -A3 research_logs/r12_$tag.log | tail -4 >> $L
  grep -oE "epoch 1[0-5]:.*" research_logs/r12_$tag.log | tail -1 >> $L
}
run lgn      --noise_eps=0   --ssl_reg=0
run eps005   --noise_eps=0.05
run eps010   --noise_eps=0.1
run eps020   --noise_eps=0.2
run eps030   --noise_eps=0.3
echo ALLDONE >> $L

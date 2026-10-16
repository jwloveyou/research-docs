#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r12b_progress.txt
echo "BEGIN $(date)" > $L
run () {
  tag=$1; seed=$2; eps=$3
  echo "START $tag $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=gowalla --seed=$seed --epochs=15 --verbose=1 --stop_cnt=40 --noise_eps=$eps > research_logs/r12b_$tag.log 2>&1
  echo "END   $tag rc=$? $(date)" >> $L
  grep "best_result" -A3 research_logs/r12b_$tag.log | tail -4 >> $L
}
run eps010_s43 43 0.1
run eps010_s44 44 0.1
run eps020_s43 43 0.2
run eps020_s44 44 0.2
echo ALLDONE >> $L

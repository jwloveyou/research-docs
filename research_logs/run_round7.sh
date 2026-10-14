#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/round7_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/$1_yelp2018_s42.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
}
run svd_nn  "--view_mix=nn"
run svd_ng  "--view_mix=ng"
run svd_nng "--view_mix=nng"
echo ALLDONE >> $L

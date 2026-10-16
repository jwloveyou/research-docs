#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/fusion_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=$3 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/$1_yelp2018_s$3.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
}
run fusion_bpr "--ssl_reg=0 --view_mix=nn" 42
run fusion_nn  "--view_mix=nn" 43
run fusion_nn  "--view_mix=nn" 44
echo ALLDONE >> $L

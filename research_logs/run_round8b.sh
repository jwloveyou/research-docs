#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/round8b_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=$3 --epochs=15 --verbose=1 --stop_cnt=40 --view_mix=nn $2 > research_logs/$1_yelp2018_s$3.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
}
run fcl_mom_s43 "--fcl_mom=5.0" 43
run fcl_mom_s44 "--fcl_mom=5.0" 44
echo ALLDONE >> $L

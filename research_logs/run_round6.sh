#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
L=research_logs/round6_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/$1_yelp2018_s42.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
}
run mv_nn  "--view_mix=nn"
run mv_ne  "--view_mix=ne"
run mv_nee "--view_mix=nee"
echo ALLDONE >> $L

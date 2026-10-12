#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
L=research_logs/round4_progress.txt
run () {
  echo "START $1 $2 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=$1 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 $3 > research_logs/${2}_${1}_s42.log 2>&1
  echo "END   $1 $2 rc=$? $(date)" >> $L
}
run yelp2018 freq_views_only "--freq_scaler=1 --freq_type=low --freq_on_main=0 --freq_on_views=1"
run gowalla  freq_views_only "--freq_scaler=1 --freq_type=low --freq_on_main=0 --freq_on_views=1"
run yelp2018 freq_main_only  "--freq_scaler=1 --freq_type=low --freq_on_main=1 --freq_on_views=0"
echo ALLDONE >> $L

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
LOG=research_logs/clean_progress.txt
echo "clean driver start $(date)" >> $LOG
run () {  # $1 dataset  $2 tag  $3 extra  $4 epochs
  echo "START $1 $2 epochs=$4 $(date)" >> $LOG
  python main.py --recommender=NoiseSGL --dataset=$1 --seed=42 --epochs=$4 --verbose=1 --stop_cnt=40 $3 > research_logs/${2}_${1}_s42.log 2>&1
  if grep -q "epoch ${4}:" research_logs/${2}_${1}_s42.log; then
    echo "OK    $1 $2 $(date)" >> $LOG
  else
    echo "SHORT $1 $2 $(date)" >> $LOG
  fi
}
# Phase 1 : three-dataset NoiseSGL baseline (control, no freq scaler)
run amazon-book ctrl "--freq_scaler=0" 15
run gowalla     ctrl "--freq_scaler=0" 15
# Phase 2 : frequency scaler (low-pass, representative) on the two new datasets
run amazon-book freq_low "--freq_scaler=1 --freq_type=low" 15
run gowalla     freq_low "--freq_scaler=1 --freq_type=low" 15
# Phase 3 : fair-shot longer training for yelp freq-low (was still rising at ep15)
run yelp2018    freq_low_e30 "--freq_scaler=1 --freq_type=low" 30
echo DONE > research_logs/clean_done.flag
echo "clean driver done $(date)" >> $LOG

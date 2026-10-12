#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
LOG=research_logs/multids_progress.txt
while [ ! -f research_logs/multids_done.flag ]; do sleep 30; done
echo "multids done; freq-long yelp start $(date)" >> $LOG
python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=30 --verbose=1 --stop_cnt=40 --freq_scaler=1 --freq_type=low > research_logs/freq_low_s42_e30.log 2>&1
echo "freq long low done $(date)" >> $LOG
python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=30 --verbose=1 --stop_cnt=40 --freq_scaler=1 --freq_type=high > research_logs/freq_high_s42_e30.log 2>&1
echo "freq long high done $(date)" >> $LOG
echo DONE > research_logs/round3b_done.flag

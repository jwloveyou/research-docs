#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
python main.py --recommender=NoiseSGL --seed=42 --epochs=40 --verbose=1 --stop_cnt=40 --freq_scaler=1 --freq_type=low  > research_logs/freq_low_40_s42.log  2>&1 &
P1=$!
python main.py --recommender=NoiseSGL --seed=42 --epochs=40 --verbose=1 --stop_cnt=40 --freq_scaler=1 --freq_type=high > research_logs/freq_high_40_s42.log 2>&1 &
P2=$!
wait $P1; wait $P2
python main.py --recommender=NoiseSGL --seed=42 --epochs=15 --verbose=1 --stop_cnt=20 --freq_scaler=1 --freq_type=low --freq_mix=0.5 > research_logs/freq_low_mix50_s42.log 2>&1
python main.py --recommender=NoiseSGL --seed=42 --epochs=15 --verbose=1 --stop_cnt=20 --freq_scaler=1 --freq_type=low --freq_mix=0.25 > research_logs/freq_low_mix25_s42.log 2>&1
echo DONE > research_logs/round3b_done.flag

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
python main.py --recommender=NoiseSGL --seed=42 --epochs=15 --verbose=1 --stop_cnt=20 --freq_scaler=0 > research_logs/freq_ctrl_s42.log 2>&1
python main.py --recommender=NoiseSGL --seed=42 --epochs=15 --verbose=1 --stop_cnt=20 --freq_scaler=1 --freq_type=low > research_logs/freq_low_s42.log 2>&1
python main.py --recommender=NoiseSGL --seed=42 --epochs=15 --verbose=1 --stop_cnt=20 --freq_scaler=1 --freq_type=high > research_logs/freq_high_s42.log 2>&1
echo DONE > research_logs/round3_done.flag

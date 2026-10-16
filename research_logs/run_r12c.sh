#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
export CUDA_VISIBLE_DEVICES=0
L=research_logs/r12c_progress.txt
echo "BEGIN $(date)" > $L
echo "START eps020_50ep $(date)" >> $L
python main.py --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=50 --verbose=1 --stop_cnt=40 --noise_eps=0.2 > research_logs/r12c_eps020_50ep.log 2>&1
echo "END   eps020_50ep rc=$? $(date)" >> $L
grep "best_result" -A3 research_logs/r12c_eps020_50ep.log | tail -4 >> $L
echo ALLDONE >> $L

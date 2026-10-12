#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sgl
cd ~/SGL-Torch
L=research_logs/round5_progress.txt
echo "BEGIN $(date)" > $L
run () {
  echo "START $1 $(date)" >> $L
  python main.py --recommender=NoiseSGL --dataset=yelp2018 --seed=42 --epochs=15 --verbose=1 --stop_cnt=40 $2 > research_logs/$1_yelp2018_s42.log 2>&1
  echo "END   $1 rc=$? $(date)" >> $L
}
run ntssm_ctrl "--nt_ssm=0"
run ntssm_a05 "--nt_ssm=1 --nt_ssm_alpha=0.5"
echo ALLDONE >> $L

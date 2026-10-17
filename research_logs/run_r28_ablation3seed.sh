#!/bin/bash
# R28b (必补, must-not-miss): 3-seed "有/无" ablation for components ① and ②.
#   ① noise augmentation OFF = pure LightGCN (ssl_reg=0, embed256, win30 irrelevant).
#      gowalla 100ep x seeds{42,43,44} (matched to R24 "有" 100ep 3-seed) + amazon 12ep x 3.
#   ② win30 annealing OFF = fixed lambda (ssl_anneal=0, embed256, noise on).
#      gowalla 100ep x seeds{42,43,44} (matched to R24 "有" = ssl_anneal=2/epochs=30).
# Protocol identical to R24 (embed256, view_mix=nn, noise_eps=0.2, ssl_reg=0.1).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r28_ablation_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5; emb=$6; anneal=$7; seed=$8
  log=research_logs/r28_${ds}_${tag}_s${seed}.log
  echo "START ${ds}_${tag}_s${seed} gpu=$gpu ep=$eps ssl=$ssl anneal=$anneal $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=$emb --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=$anneal --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_s${seed} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_s${seed} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# ① noise aug OFF (LightGCN, ssl_reg=0): gowalla 100ep x3 + amazon 12ep x3
for s in 42 43 44; do run $1 lgn gowalla 100 0 256 0 $s; done
for s in 42 43 44; do run $1 lgn amazon-book 12 0 256 0 $s; done
# ② win30 annealing OFF (ssl_anneal=0, fixed lambda): gowalla 100ep x3
for s in 42 43 44; do run $1 noanneal gowalla 100 0.1 256 0 $s; done

echo "ALLDONE $(date)" >> $L

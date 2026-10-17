#!/bin/bash
# R27 secondary: amazon-beauty generalization (SELFRec, density 0.065%, 22.4K users).
# Small dataset -> run both embed256 (protocol-consistent) and embed128 (capacity check).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r27_amazon_beauty_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5; emb=$6
  log=research_logs/r27_${ds}_${tag}_emb${emb}.log
  echo "START ${ds}_${tag}_emb${emb} gpu=$gpu ep=$eps ssl=$ssl $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=1000 --embed_size=$emb \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_emb${emb} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_emb${emb} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run 7 fusion amazon-beauty 15 0.1 256
run 7 lgn    amazon-beauty 15 0   256
run 7 fusion amazon-beauty 15 0.1 128
run 7 lgn    amazon-beauty 15 0   128
echo "ALLDONE $(date)" >> $L

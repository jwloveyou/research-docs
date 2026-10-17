#!/bin/bash
# R27 secondary: amazon-kindle generalization (SELFRec, density 0.0112%, 138K users, 98K items).
# Medium sparse graph (~1.5M interactions); run embed256 pair (protocol-consistent) at 10ep.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r27_amazon_kindle_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5; emb=$6
  log=research_logs/r27_${ds}_${tag}_emb${emb}.log
  echo "START ${ds}_${tag}_emb${emb} gpu=$gpu ep=$eps ssl=$ssl $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=$emb --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_emb${emb} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_emb${emb} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run 7 fusion amazon-kindle 10 0.1 256
run 7 lgn    amazon-kindle 10 0   256
echo "ALLDONE $(date)" >> $L

#!/bin/bash
# R27 secondary: yelp_full capacity-robustness check (embed128 pair, on GPU6 after tmall).
# Motivation: R26 ifashion finding -> embed256 overfits ultra-sparse graphs (300K users x avg 4.2
# interactions); yelp_full is also ultra-sparse (0.0075%, avg 10.1 interactions/user), so we
# additionally test whether embed128 (fewer params) generalises better than embed256 here.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r27_emb128_progress.txt
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

run 6 fusion yelp_full 15 0.1 128
run 6 lgn    yelp_full 15 0   128
echo "ALLDONE $(date)" >> $L

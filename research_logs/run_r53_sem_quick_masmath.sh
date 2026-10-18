#!/bin/bash
# R53: clean semantic ablation (sem_emb=0 vs 1) run LOCALLY on masmath GPU0,
# sharing the card with the DCCF baseline (both are our own jobs; lastfm/ml-1m
# are tiny so contention is negligible and memory is far below 24G).
#   sem_emb=0 -> pure ID (ctrl, = current fusion)
#   sem_emb=1 -> add sem_lam*item_sem (frozen TF-IDF[+LLM tags] -> seeded Gaussian
#                projection -> L2-norm) to the layer-0 item embedding.
# Protocol = R47/R22: embed256 + win30 cosine SSL anneal + view_mix=nn + noise_eps=0.2,
# seed=42. 100ep quick signal (embed256 converges earlier than embed128 used in R22).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r53_sem_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; tag=$2; eps=$3; sem=$4; lam=$5
  log=research_logs/r53_${ds}_${tag}_sem${sem}_lam${lam}.log
  echo "START ${ds}_${tag}_sem${sem}_lam${lam} ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --sem_emb=$sem --sem_lam=$lam --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_sem${sem}_lam${lam} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_sem${sem}_lam${lam} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# lastfm-2k quick signal: ctrl vs lam=1.0
run lastfm-2k sem 100 0 1.0
run lastfm-2k sem 100 1 1.0
# ml-1m quick signal: ctrl vs lam=1.0 (dense case -- can the frozen semantic prior rescue it?)
run ml-1m sem 100 0 1.0
run ml-1m sem 100 1 1.0

echo "ALLDONE $(date)" >> $L

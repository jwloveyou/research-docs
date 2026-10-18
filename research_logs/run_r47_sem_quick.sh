#!/bin/bash
# R48 staged semantic ablation (quick signal, 100ep, seed=42).
#   sem_emb=0 -> pure ID (ctrl, = current fusion)
#   sem_emb=1 -> add sem_lam * item_sem (frozen TF-IDF[+LLM-tags] -> seeded Gaussian
#                projection -> L2-norm) to layer-0 item embedding.
# Protocol = R47/R22: embed256 + win30 cosine SSL anneal + view_mix=nn + noise_eps=0.2.
# 100ep is a trajectory-level signal (lastfm peak is ~ep132 in R22; ml-1m peak ~ep109).
# If sem1 clearly leads at ep100 -> next agent escalates to full 200ep + lam scan.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r47_sem_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; sem=$5; lam=$6
  log=research_logs/r47_${ds}_${tag}_sem${sem}_lam${lam}.log
  echo "START ${ds}_${tag}_sem${sem}_lam${lam} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py     --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1     --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2     --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30     --l2_norm=0 --dropout=0 --sem_emb=$sem --sem_lam=$lam --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_sem${sem}_lam${lam} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_sem${sem}_lam${lam} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# lastfm-2k quick signal: ctrl vs lam=1.0
run $1 lastfm-2k sem 100 0 1.0
run $1 lastfm-2k sem 100 1 1.0
# ml-1m quick signal: ctrl vs lam=1.0 (dense case -- does the semantic prior rescue it?)
run $1 ml-1m sem 100 0 1.0
run $1 ml-1m sem 100 1 1.0

echo "ALLDONE $(date)" >> $L

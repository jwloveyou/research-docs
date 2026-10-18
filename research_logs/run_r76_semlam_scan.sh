#!/bin/bash
# R76: semantic component (component 4) sem_lam sensitivity scan.
# Motivation (GrIS arXiv:2610.01533 + "Cross-Stage Decoupling of Semantic and
# Collaborative Signals" arXiv:2609.13678): the frozen semantic prior strength
# sem_lam must be BALANCED against the collaborative signal; sem_lam was only
# ever tested at 1.0 (R53: +0.82%/+0.73% on lastfm 3-seed). This scan fills the
# missing sensitivity curve for the semantic component (task#2 sensitivity item).
# Protocol = R53 exactly: embed256 + win30 cosine SSL anneal + view_mix=nn +
# noise_eps=0.2 + seed=42, 100ep quick signal. sem_emb=1 fixed, vary sem_lam.
# sem_lam=0.0 (=R53 sem0) and 1.0 (=R53 sem1) already have single-seed values;
# we re-run 1.0 for consistent timing and add {0.5, 2.0, 4.0}.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r76_semlam_progress.txt
echo "BEGIN $(date)" > $L

run () {
  ds=$1; lam=$2; eps=$3
  tag=$(echo $lam | tr '.' '_')
  log=research_logs/r76_${ds}_semlam${tag}.log
  echo "START ${ds} sem_lam=${lam} ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --sem_emb=1 --sem_lam=$lam --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   ${ds} sem_lam=${lam} rc=$rc $(date)" >> $L
  echo "--- ${ds} sem_lam=${lam} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# lastfm-2k (positive semantic dataset): 4-point scan {0.5, 1.0, 2.0, 4.0}
run lastfm-2k 0.5 100
run lastfm-2k 1.0 100
run lastfm-2k 2.0 100
run lastfm-2k 4.0 100

echo "ALLDONE $(date)" >> $L

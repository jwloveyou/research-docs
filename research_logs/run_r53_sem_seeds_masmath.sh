#!/bin/bash
# R53b: 3-seed confirmation of the lastfm semantic ablation (sem_emb=0 vs 1).
# R53 single-seed (42): sem0 0.4183/0.3658 vs sem1 0.4209/0.3683 (+0.62%/+0.79%).
# Complete seeds 43/44 to judge significance against lastfm seed noise.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r53_sem_seeds_progress.txt
echo "BEGIN $(date)" > $L

run () {
  seed=$1; ds=$2; sem=$3
  log=research_logs/r53_${ds}_s${seed}_sem${sem}_lam1.0.log
  echo "START ${ds}_s${seed}_sem${sem} $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=100 --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --sem_emb=$sem --sem_lam=1.0 --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   ${ds}_s${seed}_sem${sem} rc=$rc $(date)" >> $L
  echo "--- ${ds}_s${seed}_sem${sem} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

for seed in 43 44; do
  run $seed lastfm-2k 0
  run $seed lastfm-2k 1
done

echo "ALLDONE $(date)" >> $L

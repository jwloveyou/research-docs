#!/bin/bash
# R30 PREVIEW on GPU4 (idle): quick 4-config x 30ep signal for the InfoNCE temperature
# schedule 4th-component candidate, ahead of the full 6-config scan queued on GPU7.
# Writes to a SEPARATE progress file so it never clobbers the queued r30_progress.txt.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r30_preview_gpu4_progress.txt
echo "BEGIN $(date)" > $L
run () {
  tag=$1; ta=$2; tae=$3; t0=$4; t1=$5
  log=research_logs/r30_preview_gowalla_${tag}.log
  echo "START ${tag} tau_anneal=$ta tau=${t0}->${t1} $(date)" >> $L
  CUDA_VISIBLE_DEVICES=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=30 --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --ssl_temp_anneal=$ta --ssl_temp_anneal_epochs=$tae --ssl_temp_start=$t0 --ssl_temp_end=$t1 \
    --l2_norm=0 --dropout=0 --gpu_id=4 > $log 2>&1
  rc=$?
  echo "END   ${tag} rc=$rc $(date)" >> $L
  echo "--- ${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}
run ctrl_tau020 0 0 0.2 0.2
run cos_020_005 2 30 0.2 0.05
run cos_020_050 2 30 0.2 0.5
run fix_tau010 0 0 0.1 0.1
echo "ALLDONE $(date)" >> $L

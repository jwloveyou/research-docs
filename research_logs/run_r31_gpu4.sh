#!/bin/bash
# R31 (this round, GPU4): density-aware noise strength (noise_deg_gamma) — the last
# remaining 4th-component candidate — evaluated on GPU4 in parallel with the GPU7 chain.
#   gowalla gamma scan {0, 0.25, 0.5, 1.0} x 30ep (seed=42, embed256 + win30)
#   -> pick best gamma (highest R@20), then yelp / amazon {0, best} confirmation.
# Uses SEPARATE progress/log files (r31g4_*) so it never collides with the GPU7 chain.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r31g4_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; tag=$3; eps=$4; gamma=$5
  log=research_logs/r31g4_${ds}_${tag}_g${gamma}.log
  echo "START ${ds}_${tag}_g${gamma} gpu=$gpu ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=$gamma --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag}_g${gamma} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag}_g${gamma} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# 1) gowalla gamma scan (the head-on target, most sensitive to the 4th component)
run 4 gowalla gamma 30 0.0
run 4 gowalla gamma 30 0.25
run 4 gowalla gamma 30 0.5
run 4 gowalla gamma 30 1.0

# 2) pick best gamma among the >0 values (highest Recall@20)
best=$(for g in 0.25 0.5 1.0; do
  r=$(grep -A3 "best_result" research_logs/r31g4_gowalla_gamma_g${g}.log | grep -oE '0\.[0-9]{4}' | sed -n '2p')
  echo "$r $g"
done | sort -k1,1 -r | head -1 | awk '{print $2}')
echo "BESTGAMMA=$best" >> $L

# 3) yelp / amazon confirmation: gamma=0 vs best gamma
run 4 yelp2018 gamma 15 0.0
run 4 yelp2018 gamma 15 $best
run 4 amazon-book gamma 12 0.0
run 4 amazon-book gamma 12 $best

echo "ALLDONE $(date)" >> $L

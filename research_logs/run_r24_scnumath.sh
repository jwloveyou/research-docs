#!/bin/bash
# R24: embed256 3-seed finalization (capacity-scaling fusion upgrade 128->256).
# Reuses seed42 values from R23 (yelp 0.0709@ep6 / amazon 0.0551@ep11 / gowalla 0.1880@ep61).
# Runs seeds 43,44 for the three datasets to make a clean 3-seed table.
# scnumath GPUs 4-7 (one task per GPU; GPUs 0-3 occupied by others -> untouched).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r24_scnumath_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; seed=$3; eps=$4
  log=research_logs/r24_${ds}_emb256_s${seed}.log
  echo "START ${ds}_emb256_s${seed} gpu=$gpu eps=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=$seed --epochs=$eps --verbose=1 --stop_cnt=500 --embed_size=256 \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --gpu_id=$gpu > $log 2>&1
  echo "END   ${ds}_emb256_s${seed} rc=$? $(date)" >> $L
  grep "best_result" -A3 $log | tail -3 >> $L
}

# GPU4: gowalla seed43 (100ep, ~1h)
run 4 gowalla 43 100 &
# GPU5: gowalla seed44 (100ep, ~1h)
run 5 gowalla 44 100 &
# GPU6: yelp seed43 -> seed44 (20ep each, ~35min each)
( run 6 yelp2018 43 20
  run 6 yelp2018 44 20 ) &
# GPU7: amazon seed43 -> seed44 (12ep each, ~90min each)
( run 7 amazon-book 43 12
  run 7 amazon-book 44 12 ) &
wait
echo "ALLDONE $(date)" >> $L

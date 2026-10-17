#!/bin/bash
# R23: L2-normalised (cosine) scoring ablation + embed256 capacity extension.
# scnumath GPUs 4-7 (one task per GPU; GPUs 0-3 occupied by others -> untouched).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r23_scnumath_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; emb=$5; l2n=$6
  log=research_logs/r23_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=500 --embed_size=$emb \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=$l2n --gpu_id=$gpu > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  grep "best_result" -A3 $log | tail -3 >> $L
}

# GPU4: gowalla ctrl (embed128, l2n=0, 200ep to capture ep195 peak)
run 4 ctrl_emb128_l2n0 gowalla 200 128 0 &
# GPU5: gowalla l2n=1 (embed128, 200ep)
run 5 emb128_l2n1 gowalla 200 128 1 &
# GPU6: amazon ctrl 15ep -> amazon embed256 12ep (secondary capacity extension)
( run 6 ctrl_emb128_l2n0 amazon-book 15 128 0
  run 6 emb256_l2n0 amazon-book 12 256 0 ) &
# GPU7: yelp l2n=1 20ep -> amazon l2n=1 15ep
( run 7 emb128_l2n1 yelp2018 20 128 1
  run 7 emb128_l2n1 amazon-book 15 128 1 ) &
wait
echo "ALLDONE $(date)" >> $L

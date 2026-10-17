#!/bin/bash
# R32: yelp_full capacity fix + missing baseline.
#   (1) lgn embed256 10ep — the LightGCN baseline for yelp_full was never run (GPU5 was
#       grabbed by the baseline chain after fusion emb256 finished); fill it so we can
#       judge whether fusion emb256 peak 0.1043@ep3 actually beats LightGCN.
#   (2) fusion+lgn embed64 10ep — capacity-robustness check for this ultra-sparse large
#       graph (0.0075%, 267K users x avg 10.1 interactions), since emb256/128 both overfit.
# ~20min/ep (full-graph InfoNCE O(batch x N_nodes) dominates) -> 10ep x 3 = ~10h total.
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r32_yelp_full_emb64_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5; emb=$6
  log=research_logs/r32_${ds}_${tag}_emb${emb}.log
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

# missing LightGCN baseline (matched capacity to fusion emb256)
run 4 lgn    yelp_full 10 0   256
# capacity fix: embed64 pair
run 4 fusion yelp_full 10 0.1 64
run 4 lgn    yelp_full 10 0   64

echo "ALLDONE $(date)" >> $L

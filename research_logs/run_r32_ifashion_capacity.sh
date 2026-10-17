#!/bin/bash
# R32: ifashion capacity fix — does the noise-SSL fusion still beat LightGCN on the
# ultra-sparse iFashion graph (density 0.0051%, ~300K users x avg 4.2 interactions)
# once capacity is reduced from embed256 (which R26 showed overfits: peak ep3 then
# monotonic decline) to embed64 / embed128?
# Protocol = R26: matched split, cold-start dropped, 0-indexed ids; seed=42.
#   fusion = NoiseSGL 3 components (view_mix=nn + ssl_anneal=2/window=30 + embed_size)
#   lgn    = capacity-matched LightGCN baseline (ssl_reg=0), given a more generous
#            epoch budget (40ep) because LightGCN converges ~10x slower (R12/R16).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r32_ifashion_progress.txt
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

# fusion at two reduced capacities (the R26 overfit was at embed256)
run 4 fusion ifashion 20 0.1 64
run 4 fusion ifashion 20 0.1 128
# capacity-matched LightGCN baselines, generous epoch budget (slow convergence)
run 4 lgn    ifashion 40 0   64
run 4 lgn    ifashion 40 0   128

echo "ALLDONE $(date)" >> $L

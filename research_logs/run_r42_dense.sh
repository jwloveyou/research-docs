#!/bin/bash
# R42: dense-dataset generalization boundary (task #7). Pin down the density
# flip-point of the noise-augmentation method. ml-1m (5.07%) already fails; we add
#   * ml-10m : moderate density (~2-3%), near the suspected flip point
#   * jester : EXTREME density (72.5%), a definite-fail extreme point
# Protocol identical to R22 (10-core + global 80/20 seed 42, pair format) but with
# the CURRENT fusion capacity embed_size=256 (R24) for a capacity-matched comparison.
#   fusion = NoiseSGL 3 components: view_mix=nn + ssl_anneal=2/window=30 + embed256
#   lgn    = capacity-matched LightGCN (ssl_reg=0, embed256, longer budget ~3x)
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r42_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5
  log=research_logs/r42_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu ep=$eps ssl=$ssl $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# jester (extreme dense 72.5%, 25K nodes, 1.8M interactions -> fast): fusion 100ep vs LightGCN 200ep
run $1 fusion jester 100 0.1
run $1 lgn    jester 200 0
# ml-10m (moderate dense ~1.5-2%, ~79K nodes, ~9.9M interactions -> slow): reduced budget
run $1 fusion ml-10m 50 0.1
run $1 lgn    ml-10m 100 0
# ml-1m re-verify at embed256 (R22 was embed128; for a consistent density curve): fusion 60ep vs lgn 150ep
run $1 fusion ml-1m 60 0.1
run $1 lgn    ml-1m 150 0

echo "ALLDONE $(date)" >> $L

#!/bin/bash
# HCCF baseline (task #8, No.1): faithful Hypergraph Contrastive Collaborative Filtering
# (SIGIR'22, arXiv:2204.12200, official repo akaxlh/HCCF torchVersion/Model.py),
# implemented in NoiseSGL.py under the hccf=1 switch.
#
# Protocol (same as the other SOTA baselines, R29): LightGCN-local backbone, embed_size=256
# (capacity-matched to fusion), n_layers=3, seed=42, best-over-epochs R@20/NDCG@20 on test,
# lr=1e-3, batch=1024, reg=1e-7 (HCCF official), stop_cnt=1000. HCCF-specific hyperparams
# from the official code: hyperNum=128, keepRate=0.5 (edge-drop keep prob), ssl_reg=1e-3
# (contrast weight), temp=1.0. No annealing / noise / capacity beyond embed256 (HCCF has no
# such components), so the comparison isolates the hypergraph-contrast mechanism.
#
# Epoch budget matches the SSL baselines: gowalla 150ep / yelp 50ep / amazon 30ep.
#
# Usage: bash run_hccf_baseline.sh <gpu_id>
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=$1
L=research_logs/r_hccf_progress.txt
[ -f $L ] || echo "BEGIN $(date)" > $L

run () {
  tag=$1; ds=$2; eps=$3
  log=research_logs/r_hccf_${ds}.log
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --use_mlc=0 --gpu_id=$GPU \
    --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-3 --hccf_temp=1.0 \
    --reg=1e-7 > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run hccf gowalla     150
run hccf yelp2018     50
run hccf amazon-book  30
echo "HCCF_ALLDONE $(date)" >> $L

#!/bin/bash
# R29 (task #4): same-protocol SOTA baseline reproduction for the paper comparison table.
#
# Baselines (all already validated by CPU smoke test, R29):
#   LightGCN  = ssl_reg=0                              (LightGCN, SIGIR'20, arXiv:2002.02126)
#   SGL       = view_mix=ee (2 edge-dropout views)     (SGL, SIGIR'21, arXiv:2010.10783)
#   SimGCL    = view_mix=nn + noise_mode=0 (no anneal) (SimGCL, SIGIR'23, arXiv:2206.07869)
#   LightGCL  = lightgcl=1 (local vs SVD-global)       (LightGCL, ICLR'23, arXiv:2302.08191)
#   XSimGCL   = view_mix=nn + noise_mode=1 (no anneal) (XSimGCL, SIGIR'23, arXiv:2209.02544)
#
# Protocol (fair, matched): LightGCN backbone, embed_size=256 (capacity-matched to fusion),
#   n_layers=3, same data split / seed=42 / eval (R@20,NDCG@20 on test), lr=1e-3, batch=1024,
#   reg=1e-4, ssl_temp=0.2, ssl_ratio=0.1. Baselines do NOT use our annealing (component 2);
#   LightGCN uses ssl_reg=0. stop_cnt=1000 -> report best-over-epochs.
#
# Epoch budget: LightGCN is known to converge ~10x slower (R12/R16), so it gets a longer
#   budget (gowalla 300ep / yelp & amazon 100ep); SSL baselines get fusion-matched budgets
#   (gowalla 150ep / yelp 50ep / amazon 30ep) where they already peak.
#
# Usage: bash run_baselines_seed42.sh <gpu_id> <part>
#   part=1 : gowalla {LightGCN, SGL, SimGCL}
#   part=2 : gowalla {LightGCL, XSimGCL} + yelp/amazon LightGCN
#   part=3 : yelp/amazon {SGL, SimGCL, LightGCL, XSimGCL}
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=$1
PART=${2:-1}
L=research_logs/r29_baselines_progress.txt
[ -f $L ] || echo "BEGIN $(date)" > $L
echo "BEGIN part=$PART gpu=$GPU $(date)" >> $L

run () {
  tag=$1; ds=$2; eps=$3; extra="$4"
  log=research_logs/r29_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --use_mlc=0 --noise_eps=0.2 \
    --ssl_anneal=0 --ssl_anneal_epochs=0 --ssl_anneal_floor=0.0 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --gpu_id=$GPU $extra > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

if [ "$PART" = "1" ]; then
  run lgn    gowalla     300 "--view_mix=nn --ssl_reg=0"
  run sgl    gowalla     150 "--view_mix=ee --ssl_reg=0.1"
  run simgcl gowalla     150 "--view_mix=nn --ssl_reg=0.1 --noise_mode=0"
elif [ "$PART" = "2" ]; then
  run lgcl   gowalla     150 "--lightgcl=1 --lightgcl_perlayer=1 --ssl_reg=0.001"
  run xsimgcl gowalla    150 "--view_mix=nn --ssl_reg=0.1 --noise_mode=1"
  run lgn    yelp2018    100 "--view_mix=nn --ssl_reg=0"
  run lgn    amazon-book 100 "--view_mix=nn --ssl_reg=0"
elif [ "$PART" = "3" ]; then
  run sgl    yelp2018     50 "--view_mix=ee --ssl_reg=0.1"
  run sgl    amazon-book  30 "--view_mix=ee --ssl_reg=0.1"
  run simgcl yelp2018     50 "--view_mix=nn --ssl_reg=0.1 --noise_mode=0"
  run simgcl amazon-book  30 "--view_mix=nn --ssl_reg=0.1 --noise_mode=0"
  run lgcl   yelp2018     50 "--lightgcl=1 --lightgcl_perlayer=1 --ssl_reg=0.001"
  run lgcl   amazon-book  30 "--lightgcl=1 --lightgcl_perlayer=1 --ssl_reg=0.001"
  run xsimgcl yelp2018    50 "--view_mix=nn --ssl_reg=0.1 --noise_mode=1"
  run xsimgcl amazon-book 30 "--view_mix=nn --ssl_reg=0.1 --noise_mode=1"
else
  echo "unknown part $PART" >> $L; exit 1
fi

echo "DONE part=$PART $(date)" >> $L

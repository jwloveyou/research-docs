#!/bin/bash
# R41 (4th-component candidate #15): BPR ranking-margin scan.
#   bpr_margin=0 keeps plain BPR (backward compatible). margin>0 subtracts gamma from
#   the BPR logits: loss = -log sigmoid(pos - neg - gamma), requiring a STRONGER
#   separation between the positive and the negative item and sharpening the ranking
#   signal. Static, parameter-free, orthogonal to the 3 fusion components (noise / SSL
#   anneal / capacity) and to every previously queued candidate: it modifies the BPR
#   margin DIRECTLY rather than adding an auxiliary loss (unlike R8/R9/R21/R35).
# Literature: 'Revisiting BPR' (arXiv:2409.14217) -- BPR loss design / tuning is the
#   main gap vs SOTA; contrastive-margin / hinge ranking losses (CML, HNS) -- a margin
#   sharpens top-K ranking. This is the FIRST loss-side change that touches the BPR
#   margin itself instead of stacking a new regulariser.
#
# Scan (protocol identical to R38/R39, embed256 + win30, seed=42):
#   gowalla 60ep  margin in {0(ctrl), 0.1, 0.2, 0.3, 0.5}   (R24 peak ~ep61)
#   yelp    20ep  margin in {0, 0.1, 0.3}                    (R24 yelp peaks ep5-7)
#   amazon  12ep  margin in {0, 0.1, 0.3}                    (R24 amazon peaks ep11-12)
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r41_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; ds=$2; eps=$3; margin=$4
  tag="m${margin}"
  log=research_logs/r41_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu ep=$eps margin=$margin $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 --weight_decay=0.0 \
    --bpr_margin=$margin --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# gowalla (main dataset; the BPR fine-tune phase is longest there)
for m in 0.0 0.1 0.2 0.3 0.5; do run $1 gowalla 60 $m; done
# yelp / amazon quick confirmation
for m in 0.0 0.1 0.3; do run $1 yelp2018 20 $m; done
for m in 0.0 0.1 0.3; do run $1 amazon-book 12 $m; done

echo "ALLDONE $(date)" >> $L

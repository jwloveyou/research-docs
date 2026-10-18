#!/bin/bash
# R64: decisive density-boundary mechanism scan on ml-1m (dense, 5.07% — the ONE
# dataset where NoiseSGL loses to LightGCN, -3.3% R@20 in R22).
#
# Why this experiment (R64 hypothesis, grounded in r55 mechanism data + XSimGCL 2209.02544):
#   r55 mechanism (now COMPLETE) shows that on ml-1m the noise SSL STILL releases
#   effective rank (64.4 -> 119.2, +85%) and STILL suppresses over-smoothing
#   (smooth_final 0.2504 -> 0.0518, -79%), i.e. the E11 mechanism WORKS on dense
#   graphs too -- yet R@20 drops -3.3%. This FALSIFIES the old §5.3 claim "dense
#   graphs do not over-smooth, so noise has no anti-over-smoothing benefit".
#   Correct hypothesis: the harm on dense graphs is the InfoNCE uniformity
#   objective (XSimGCL 2209.02544: CL makes embeddings more uniform / mitigates
#   popularity bias), which fights the strong popularity/concentration signal that
#   BPR ranking relies on in dense graphs.
#
# Decisive test (isolate noise magnitude vs InfoNCE strength):
#   LGN     ssl_reg=0            (pure LightGCN reference, peak 0.2621@ep109)
#   ctrl    eps=0.2  reg=0.1     (current fusion -> ~0.2535@ep7)
#   reg001  eps=0.2  reg=0.01    (weak InfoNCE: if this recovers toward LGN -> InfoNCE is the harm)
#   reg0001 eps=0.2  reg=0.001   (very weak InfoNCE)
#   eps005  eps=0.05 reg=0.1     (small noise: if this recovers -> noise magnitude is the harm)
#
# All configs: embed256 + win30 cosine SSL anneal + view_mix=nn + use_mlc=0, seed=42.
# ml-1m fusion peaks @ep7 then deterministically overfits, so 40ep is plenty for the
# fusion variants; LGN needs ~100ep to approach its 0.2621 peak.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r64_dense_progress.txt
echo "BEGIN $(date)" > $L

run () {
  tag=$1; eps=$2; reg=$3; ep=$4
  log=research_logs/r64_ml1m_${tag}.log
  echo "START ml-1m_${tag} eps=$eps reg=$reg ep=$ep $(date)" >> $L
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=ml-1m --seed=42 --epochs=$ep --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=$eps \
    --ssl_reg=$reg --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --sem_emb=0 --gpu_id=0 > $log 2>&1
  rc=$?
  echo "END   ml-1m_${tag} rc=$rc $(date)" >> $L
  echo "--- ml-1m_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
  # also record the full best@epoch line for peak-epoch tracking
  grep "best@" $log | tail -2 >> $L
}

run lgn_reg0   0.2  0.0    100   # pure LightGCN reference
run ctrl_e02_r01 0.2  0.1   40    # current fusion (expected ~0.2535@ep7)
run reg001     0.2  0.01  40    # weak InfoNCE
run reg0001    0.2  0.001 40    # very weak InfoNCE
run eps005     0.05 0.1   40    # small noise

echo "ALLDONE $(date)" >> $L

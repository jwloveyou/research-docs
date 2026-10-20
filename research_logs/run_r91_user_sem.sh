#!/usr/bin/env bash
# =============================================================================
# R91 · user-side frozen semantic feature ablation (5th-component candidate,
#       semantic axis user counterpart). Literature: "Turning Semantics into
#       Topology: LLM-Driven Attribute Augmentation for CF" (arXiv:2602.21099)
#       -- embedding methods rely on *unilateral* (item-side) augmentation;
#       a user-side prior completes the semantic axis.
#
# Protocol (matches R76b final semantic config EXACTLY, only adds user_sem_emb):
#   lastfm-2k / ml-1m, embed256, win30 cosine SSL anneal (ssl_anneal=2 /
#   ssl_anneal_epochs=30 / floor=0), view_mix=nn, noise_eps=0.2, ssl_reg=0.1,
#   seed=42, best-over-epochs, full ranking + NDCG@20.
#   ctrl  = item-only semantic (sem_emb=1, sem_lam=2.0, user_sem_emb=0)
#           == R76b 3-seed mean 0.42423/0.37097 (lastfm-2k).
#   cand  = item + user semantic (user_sem_emb=1, user_sem_lam in {0.5,1.0,2.0})
#
# NOTE: masmath GPU0 is currently running the R82 serial rerun chain. DO NOT
# launch this script until R82_RERUN_ALLDONE (or launch on an idle card).
# Sequential, single GPU, no co-resident jobs (R82 lesson: parallel co-residency
# caused CUDA OOM + disk-quota failures).
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=0
L=research_logs/r91_user_sem_progress.txt
echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  ds=$1; seed=$2; eps=$3; ulam=$4
  tag="uls${ulam}_s${seed}"
  log="research_logs/r91_${ds}_${tag}.log"
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=$seed --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --embed_size=256 --view_mix=nn --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 \
    --ssl_anneal_floor=0.0 --sem_emb=1 --sem_lam=2.0 \
    --user_sem_emb=1 --user_sem_lam=$ulam --l2_norm=0 --dropout=0.0 > "$log" 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${ds}_${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

# ---- lastfm-2k: user_sem_lam scan (single seed 42, 100ep) ----
run lastfm-2k 42 100 0.5
run lastfm-2k 42 100 1.0
run lastfm-2k 42 100 2.0

# ---- ml-1m: item-only vs item+user (single seed 42, 100ep) ----
run ml-1m 42 100 0.5
run ml-1m 42 100 1.0

# ---- 3-seed confirmation of the best user_sem_lam on lastfm-2k (seed 43/44; 42 reuse above) ----
# (fill BEST_ULAM after the scan: 1.0 default; adjust if the scan shows 0.5 or 2.0 better)
BEST_ULAM=1.0
run lastfm-2k 43 100 $BEST_ULAM
run lastfm-2k 44 100 $BEST_ULAM

echo "R91_USER_SEM_ALLDONE $(date)" >> "$L"

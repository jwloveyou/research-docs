#!/usr/bin/env bash
# =============================================================================
# R82 · rerun of tasks that failed on 10-20 due to two transient causes:
#   (a) CUDA OOM  — amazon-book lightgcn ran while R81 embed512 (15.7GB) was co-resident
#   (b) disk quota exceeded (Errno 122) — killed simgcl/xsimgcl yelp+amazon and amazon emb512 s44
# scnumath is now sealed, so everything reruns SEQUENTIALLY on masmath GPU0 (single 4090).
# GPU0 is verified idle before launch. No parallel co-resident jobs this time.
#
# Order (most important for the paper's main table first):
#   1 yelp2018_simgcl      50ep   (main-table baseline)
#   2 amazon-book_simgcl   30ep
#   3 gowalla_xsimgcl      150ep
#   4 yelp2018_xsimgcl     50ep
#   5 amazon-book_xsimgcl  30ep
#   6 amazon-book_emb512 s44 15ep  (finishes embed512 3-seed zero-crossing)
#   7 amazon-book_lightgcn 100ep  (last; may miss deadline — headline uses paper 0.0411)
#
# All baseline runs use OFFICIAL hyperparams (R79): embed64 / n_layers3 / batch2048 / lr1e-3 /
# reg1e-4; SGL view_mix=ee ratio0.1 temp0.2; SimGCL eps0.1 noise_mode0; XSimGCL eps0.2 noise_mode1.
# The emb512 run uses the R81 fusion protocol exactly (only amazon s44 was missing).
# Eval protocol identical to fusion (full ranking + NDCG@20 best-over-epochs).
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
GPU=0
L=research_logs/r82_rerun_failed_progress.txt
echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  tag=$1; ds=$2; eps=$3; shift 3
  log="research_logs/r82_${ds}_${tag}.log"
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --use_mlc=0 --ssl_anneal=0 --ssl_anneal_epochs=0 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 "$@" > "$log" 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${ds}_${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

# ---- SimGCL (官方 embed64/3层/batch2048/cl_rate0.1/eps0.1/tau0.2, 逐层噪声) ----
run simgcl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run simgcl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2

# ---- XSimGCL (官方 embed64/3层/eps0.2/cl0.1/tau0.2, 单次噪声) ----
run xsimgcl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run xsimgcl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
run xsimgcl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2

# ---- amazon-book embed512 s44 (fusion config, finishes 3-seed) ----
log="research_logs/r82_amazon-book_emb512_s44.log"
echo "START amazon-book_emb512_s44 gpu=$GPU ep=15 $(date)" >> "$L"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
  --recommender=NoiseSGL --dataset=amazon-book --seed=44 --epochs=15 --verbose=1 \
  --stop_cnt=1000 --embed_size=512 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
  --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
  --l2_norm=0 --dropout=0 --gpu_id=$GPU > "$log" 2>&1
rc=$?
echo "END   amazon-book_emb512_s44 rc=$rc $(date)" >> "$L"
echo "--- amazon-book_emb512_s44 best ---" >> "$L"
grep "best_result" -A3 "$log" | tail -4 >> "$L"

# ---- amazon-book LightGCN (官方 embed64; OOM 重跑, 最后) ----
run lightgcn amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0

echo "R82_RERUN_ALLDONE $(date)" >> "$L"

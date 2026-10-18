#!/bin/bash
# R74: ml-10m frozen-semantic-injection ablation (component④ on a 3rd text-rich dataset).
# Extends R47/R53 (lastfm +0.82%, ml-1m neutral) to MovieLens-10M (1.47% density),
# the middle point of the density spectrum between lastfm (2.30%, benefits) and
# ml-1m (5.07%, neutral/hurt). Item text = title + genre + user tags (movies.dat +
# tags.dat from ml-10m.zip), encoded by util/build_semantic_features.py (TF-IDF ->
# frozen seeded Gaussian projection -> row L2-normalised), the same pipeline as
# lastfm/ml-1m.
#
# Protocol IDENTICAL to run_r42_dense.sh fusion config (embed256 + view_mix=nn +
# win30 SSL anneal + noise_eps=0.2), only the sem_emb switch differs (0 vs 1).
#
# ⚠️ COST: ml-10m full-graph InfoNCE is ~40 min/epoch (9.99M interactions, ~79.6K
# nodes, O(batch x N_nodes) negatives), so a 15ep ablation ≈ 2 x 10h. Run only on a
# genuinely free GPU (see R27/R36 efficiency note).
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r74_ml10m_sem_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; sem=$2; ep=$3
  log=research_logs/r74_ml-10m_sem${sem}.log
  echo "START ml-10m sem_emb=$sem gpu=$gpu ep=$ep $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=ml-10m --seed=42 --epochs=$ep --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 \
    --sem_emb=$sem --sem_lam=1.0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ml-10m sem_emb=$sem rc=$rc $(date)" >> $L
  echo "--- ml-10m sem_emb=$sem best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

run "$1" 0 15
run "$1" 1 15

echo "ALLDONE $(date)" >> $L

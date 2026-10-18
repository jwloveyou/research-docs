#!/bin/bash
# R62 bottom-level visualisation evidence: over-smoothing curve + spectral energy
# distribution + 2-D embedding projection (PCA + t-SNE), for NoiseSGL vs LightGCN.
# Runs sequentially on masmath GPU0 (shared with our other tasks, low memory).
set -u
cd ~/SGL-Torch
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
OUT=research_logs/r62_viz_progress.txt

echo "BEGIN $(date)" > "$OUT"

run_viz () {
  local ds=$1 ssl=$2 tag=$3 ep=$4
  echo "START ${tag} ds=${ds} ssl=${ssl} ep=${ep} $(date)" >> "$OUT"
  "$PY" research_logs/analyze_embedding_viz.py \
    --dataset="$ds" --ssl_reg="$ssl" --embed_size=128 --epochs="$ep" \
    --tag="$tag" --gpu_id=0 --seed=42 >> "research_logs/r62_${tag}.log" 2>&1
  echo "END   ${tag} rc=$? $(date)" >> "$OUT"
}

run_viz gowalla 0.1 gowalla_noise 30
run_viz gowalla 0.0 gowalla_lgn   30
run_viz lastfm-2k 0.1 lastfm_noise 30
run_viz lastfm-2k 0.0 lastfm_lgn   30

echo "ALLDONE $(date)" >> "$OUT"

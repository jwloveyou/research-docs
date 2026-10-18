#!/bin/bash
# R63: auto-generate the gowalla + lastfm bottom-level visualisation SVGs the
# moment their viz_<tag>.json inputs exist (the R62 chain is still running on
# GPU0 and each run takes ~2h under contention). Pure CPU, no GPU contention.
set -u
cd ~/SGL-Torch
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
OUT=research_logs/r63_vizplot_progress.txt

echo "BEGIN $(date)" > "$OUT"

plot_if_ready () {
  local ds=$1
  local noise=research_logs/viz_${ds}_noise.json
  local lgn=research_logs/viz_${ds}_lgn.json
  local svg=research_logs/embedding_viz_${ds}.svg
  if [ -f "$noise" ] && [ -f "$lgn" ] && [ ! -f "$svg" ]; then
    echo "PLOT ${ds} $(date)" >> "$OUT"
    "$PY" util/plot_embedding_viz.py --noise "$noise" --lgn "$lgn" --out "$svg" >> "$OUT" 2>&1
    echo "DONE ${ds} rc=$? $(date)" >> "$OUT"
  fi
}

# poll for up to 6 hours (the remaining chain: gowalla_lgn -> lastfm_noise -> lastfm_lgn)
for _ in $(seq 1 2160); do
  plot_if_ready gowalla
  plot_if_ready lastfm
  if [ -f research_logs/embedding_viz_gowalla.svg ] && [ -f research_logs/embedding_viz_lastfm-2k.svg ]; then
    echo "ALLDONE $(date)" >> "$OUT"
    exit 0
  fi
  sleep 10
done
echo "TIMEOUT $(date)" >> "$OUT"

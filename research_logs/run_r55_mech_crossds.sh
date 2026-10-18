#!/bin/bash
# R55: cross-dataset mechanism re-verification (E11 extension)
# Compare embedding-collapse / over-smoothing / effective-rank between
#   NoiseSGL (ssl_reg=0.1, embed128) and pure LightGCN (ssl_reg=0, embed128)
# on a BENEFIT dataset (lastfm-2k, sparse) vs a HURT dataset (ml-1m, dense).
# Hypothesis (paper §5.3): noise releases rank on sparse graphs (big eff-rank gap)
# but not on dense graphs (LightGCN already high-rank -> gap ~0).
# Protocol: embed128 (consistent with E11 on gowalla), 40ep, seed42, win30 anneal.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
ROOT=/home/Yehaikun/SGL-Torch
cd "$ROOT"
PROG=$ROOT/research_logs/analyze_mechanism.py
OUT=$ROOT/research_logs/r55_mech_progress.txt
: > "$OUT"
run() {  # $1=tag $2=dataset $3=ssl_reg
  echo "START $1 ds=$2 ssl=$3 $(date '+%F %T')" >> "$OUT"
  $PY "$PROG" --dataset="$2" --ssl_reg="$3" --embed_size=128 --epochs=40 \
      --log_every=2 --tag="$1" --gpu_id=0 --seed=42 >> "$ROOT/research_logs/r55_$1.log" 2>&1
  rc=$?
  echo "END   $1 rc=$rc $(date '+%F %T')" >> "$OUT"
  tail -1 "$ROOT/research_logs/mech_$1.jsonl" >> "$OUT" 2>/dev/null
}
run r55_lastfm_noise lastfm-2k 0.1
run r55_lastfm_lgn   lastfm-2k 0
run r55_ml1m_noise  ml-1m    0.1
run r55_ml1m_lgn    ml-1m    0
echo "ALLDONE $(date '+%F %T')" >> "$OUT"

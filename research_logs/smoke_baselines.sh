#!/bin/bash
# R29 smoke test: verify every SOTA-baseline config maps to a valid code path.
# epochs=0 -> model __init__ only (create_adj_mat + SVD + build), no training loop.
# Run on CPU (CUDA_VISIBLE_DEVICES="") so no GPU is touched.
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
DS=yelp2018

smoke () {
  tag=$1; shift
  log=research_logs/smoke_baseline_${tag}.log
  echo "=== smoke $tag: $@ ==="
  CUDA_VISIBLE_DEVICES="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$DS --seed=42 --epochs=0 --verbose=1 \
    --stop_cnt=1000 --embed_size=256 --use_mlc=0 --noise_eps=0.2 \
    --ssl_anneal=0 --l2_norm=0 --dropout=0 --gpu_id=0 "$@" > $log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "  OK ($rc)"; else echo "  FAIL ($rc)"; tail -20 $log; fi
}

# 1) LightGCN (ssl_reg=0)
smoke lgn    --view_mix=nn --ssl_reg=0
# 2) SGL (two edge-dropout views)
smoke sgl    --view_mix=ee --ssl_reg=0.1
# 3) SimGCL (two embedding-noise views, no anneal)
smoke simgcl --view_mix=nn --ssl_reg=0.1 --noise_mode=0
# 4) LightGCL (clean-local vs SVD-global)
smoke lgcl   --lightgcl=1 --lightgcl_perlayer=1 --ssl_reg=0.001
# 5) XSimGCL (single-shot noise)
smoke xsimgcl --view_mix=nn --ssl_reg=0.1 --noise_mode=1
# 6) fusion (control, sanity)
smoke fusion --view_mix=nn --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 --noise_mode=0

echo "ALL SMOKE DONE"

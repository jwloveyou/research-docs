#!/bin/bash
# R22: generalization on two extra public benchmarks (MovieLens-1M, LastFM-2K).
# Protocol matches existing 3 datasets: 10-core + global 80/20 (seed 42), pair format.
#   fusion = NoiseSGL 3 components: view_mix=nn + ssl_anneal=2/window=30 + embed_size=128
#   lgn    = capacity-matched LightGCN baseline (ssl_reg=0, embed_size=128)
# All seed=42, 200 epochs, one task per free GPU (4/5/6/7).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r22_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5
  log=research_logs/r22_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=1000 --embed_size=128 \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --gpu_id=$gpu > $log 2>&1
  echo "END   ${ds}_${tag} rc=$? $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# ml-1m: fusion (GPU4) vs LightGCN baseline (GPU5), 200ep
run 4 fusion ml-1m 200 0.1 &
run 5 lgn    ml-1m 200 0   &
# lastfm-2k: fusion (GPU6) vs LightGCN baseline (GPU7), 200ep
run 6 fusion lastfm-2k 200 0.1 &
run 7 lgn    lastfm-2k 200 0   &
wait
echo "ALLDONE $(date)" >> $L

#!/bin/bash
# R26: generalization validation on 3 more public benchmarks
#   douban-book (SELFRec), ifashion (SELFRec), tmall (DCCF).
# Protocol: pre-split standard benchmark train/test (kept authors' split),
#   cold-start test pairs dropped, ids remapped to 0-indexed contiguous.
#   fusion = NoiseSGL final 3 components: view_mix=nn + ssl_anneal=2/window=30 + embed_size=256
#   lgn    = capacity-matched LightGCN baseline (ssl_reg=0, embed_size=256)
# All seed=42, equal epoch budget per dataset (fusion converges fast; LightGCN slow).
PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch
L=research_logs/r26_progress.txt
echo "BEGIN $(date)" > $L

run () {
  gpu=$1; tag=$2; ds=$3; eps=$4; ssl=$5
  log=research_logs/r26_${ds}_${tag}.log
  echo "START ${ds}_${tag} gpu=$gpu ep=$eps ssl=$ssl $(date)" >> $L
  CUDA_VISIBLE_DEVICES=$gpu $PY main.py --recommender=NoiseSGL --dataset=$ds \
    --seed=42 --epochs=$eps --verbose=1 --stop_cnt=1000 --embed_size=256 \
    --view_mix=nn --use_mlc=0 --noise_eps=0.2 \
    --ssl_reg=$ssl --ssl_anneal=2 --ssl_anneal_floor=0.0 --ssl_anneal_epochs=30 \
    --l2_norm=0 --dropout=0 --gpu_id=$gpu > $log 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> $L
  echo "--- ${ds}_${tag} best ---" >> $L
  grep "best_result" -A3 $log | tail -4 >> $L
}

# GPU4: douban-book fusion -> tmall fusion
# GPU6: douban-book lgn    -> tmall lgn
# GPU7: ifashion   fusion  -> ifashion lgn
( run 4 fusion douban-book 200 0.1; run 4 fusion tmall 150 0.1 ) &
( run 6 lgn    douban-book 200 0;   run 6 lgn    tmall 150 0   ) &
( run 7 fusion ifashion 20 0.1;     run 7 lgn    ifashion 20 0 ) &
wait
echo "ALLDONE $(date)" >> $L

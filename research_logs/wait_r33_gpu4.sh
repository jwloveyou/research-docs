#!/bin/bash
# R33 watcher: after the GPU4 chain (r31c -> r32 generalization capacity fix) finishes,
# run the depth-aware (layer-wise) noise smoke scan on GPU4.
# GPU4 is only used by the r31g4 -> r31c -> r32 chain, so no collision with GPU5/6/7.
cd ~/SGL-Torch
while true; do
  done=$(grep -c .R32_ALLDONE. research_logs/r32_gpu4_all_done.txt 2>/dev/null || echo 0)
  if [ "$done" -ge 1 ]; then break; fi
  sleep 180
done
bash research_logs/run_r33_layer_noise.sh 4 > research_logs/r33_launcher.out 2>&1
# LAGCL-inspired tail-boost (gamma<0) follow-up: opposite hypothesis to R28 head-boost,
# runs right after the layer-wise scan on the same GPU4.
bash research_logs/run_r31b_gpu4_tailboost.sh 4 > research_logs/r33b_launcher.out 2>&1
echo "R33_ALLDONE $(date)" > research_logs/r33_gpu4_all_done.txt

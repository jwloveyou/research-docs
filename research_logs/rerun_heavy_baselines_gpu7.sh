#!/usr/bin/env bash
# =============================================================================
# R80 · rerun_heavy_baselines_gpu7.sh
# 官方超参重跑的【重型】部分（GPU7）：HCCF 已由旧 master 跑完 gowalla/yelp、
# amazon 在途；本脚本等 hccf_ALLDONE + GPU7 空闲后，继续跑 dccf→…→ultragcn。
# lightgcn/sgl/simgcl/xsimgcl 已改由 masmath GPU0 并行跑（run_baselines_light_masmath.sh），
# 故从本列表移除，避免 GPU7 在 deadline 前重复跑轻量 baseline。
# =============================================================================
set -u
cd ~/SGL-Torch
L=research_logs/r78_official_baselines_progress.txt
M=research_logs/r78_heavy_rerun_master.txt

wait_gpu7_idle () {
  local idle=0
  while [ $idle -lt 2 ]; do
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null | tr -d ' ')
    if [ -n "$mem" ] && [ "$mem" -lt 1000 ]; then
      idle=$((idle+1))
    else
      idle=0
    fi
    sleep 120
  done
}

echo "BEGIN $(date)" > "$M"
echo "wait for hccf_ALLDONE ..." >> "$M"
# 等当前 hccf 三集跑完（amazon 在途）
while ! grep -q "hccf_ALLDONE" "$L" 2>/dev/null; do
  sleep 120
done
echo "hccf_ALLDONE seen $(date)" >> "$M"
wait_gpu7_idle
echo "GPU7 idle, launch heavy list $(date)" >> "$M"

for m in dccf adagcl ngcf lightgcl dgcf ncl directau ultragcn; do
  echo "=== launch $m $(date) ===" >> "$M"
  bash research_logs/run_baselines_official.sh 7 "$m" >> "$M" 2>&1
  echo "=== done $m rc=$? $(date) ===" >> "$M"
done
echo "R80_HEAVY_BASELINES_DONE $(date)" >> "$M"

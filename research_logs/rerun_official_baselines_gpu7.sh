#!/usr/bin/env bash
# R78 (task #13/#14 audit fix): 用官方超参顺序重跑全部 13 个 baseline（GPU7）。
# 顺序：先跑 No.1/经典 baseline（HCCF/DCCF/AdaGCL/NGCF/LightGCL/DGCF/NCL/DirectAU/UltraGCN），
#       再跑 SSL 骨干 baseline（LightGCN/SGL/SimGCL/XSimGCL）。
# 每个 baseline 内部顺序 gowalla -> yelp2018 -> amazon-book。
set -u
cd "$(dirname "$0")/.."
L=research_logs/r78_official_rerun_master.txt
echo "BEGIN $(date)" > "$L"

for m in hccf dccf adagcl ngcf lightgcl dgcf ncl directau ultragcn lightgcn sgl simgcl xsimgcl; do
  echo "=== launch $m $(date) ===" >> "$L"
  bash research_logs/run_baselines_official.sh 7 "$m" >> "$L" 2>&1
  echo "=== done $m rc=$? $(date) ===" >> "$L"
done
echo "R78_ALL_OFFICIAL_BASELINES_DONE $(date)" >> "$L"

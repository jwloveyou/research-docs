#!/usr/bin/env bash
# =============================================================================
# R119 · run_r119_static_candidates.sh —— 三个「范数=流行度」轴的静态第 4 组件
#   候选（R113 线性中心化 init_pop_gamma / R115 忠实 PRISM init_prism_alpha /
#   R119 后训练 norm_calib 的 β sweep）串行跑完，写 R119_STATIC_CANDIDATES_ALLDONE。
#
# 顺序与预算：
#   1) run_r113_initpop.sh     （4 run：ctrl/pop025/pop050/pop100，各 60ep gowalla）
#   2) run_r115_prism.sh       （4 run：ctrl/prism050/prism100/prism200，各 60ep）
#   3) run_r119_normcalib_sweep.sh（1 run，eval 期 β sweep {0,.25,.5,1.0}，60ep）
#   共 9 run，~7h（单卡串行）。判定阈值：任一候选 best R@20 > ctrl(0.1878/0.1597)
#   + 0.001 即升级 3-seed + yelp/amazon 确认；否则核心三集「3 通用组件」维持定稿。
#
# 串行、单卡、不共驻（R82 教训）。由 wait_r119_static_candidates.sh 在 baseline
# 队列空 + GPU0 空闲后自动拉起，勿与 auto_scheduler.sh 争卡。
# =============================================================================
set -u
cd ~/SGL-Torch || exit 1
L=research_logs/r119_static_candidates_progress.txt
echo "BEGIN $(date) gpu=0 [static candidates R113+R115+R119]" > "$L"

bash research_logs/run_r113_initpop.sh
echo "R113_INITPOP_DONE $(date) (see r113_initpop_progress.txt)" >> "$L"

bash research_logs/run_r115_prism.sh
echo "R115_PRISM_DONE $(date) (see r115_prism_progress.txt)" >> "$L"

bash research_logs/run_r119_normcalib_sweep.sh
echo "R119_NORMCALIB_DONE $(date) (see r119_normcalib_progress.txt)" >> "$L"

echo "R119_STATIC_CANDIDATES_ALLDONE $(date)" >> "$L"

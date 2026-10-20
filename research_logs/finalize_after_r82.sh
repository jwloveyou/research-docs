#!/usr/bin/env bash
# =============================================================================
# R85 · finalize_after_r82.sh —— R82 串行重跑链收尾后自动定稿
# =============================================================================
# 只做三件事（守诚实铁律、零造假）：
#   1) 轮询 r82_rerun_failed_progress.txt 直到出现 `R82_RERUN_ALLDONE`（R82 全链完成）
#   2) 运行 util/collect_final_results.py —— 从日志逐项提取终值 + 生成补丁候选
#   3) 运行 util/patch_final_report.py —— 把「重跑中（R82）」占位符安全回填到
#      研究总报告.md（只替换 DONE 项，写盘前自动备份，幂等可重跑）
#   4) git 提交（SGL-Torch 仓 + research-work 仓各自 commit）
#
# 只占用 CPU、绝不碰 GPU。可 nohup/setsid 挂后台无人值守。
# 用法：  nohup bash research_logs/finalize_after_r82.sh > research_logs/finalize_after_r82.out 2>&1 &
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
SGL=~/SGL-Torch
PROG="$SGL/research_logs/r82_rerun_failed_progress.txt"
OUT="$SGL/research_logs/finalize_after_r82.out"
MARK="R82_RERUN_ALLDONE"

cd "$SGL" || exit 1

echo "BEGIN $(date) [finalize_after_r82] waiting for $MARK"

# 1) 等待 R82 全链完成（每 5 分钟查一次）
while ! grep -q "$MARK" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "R82 ALLDONE detected $(date)"

# 2) 收值 + 生成补丁候选
"$PY" util/collect_final_results.py || { echo "collect_final_results FAILED"; exit 2; }

# 3) 回填 研究总报告.md（dry-run 一次确认 + 正式写盘）
"$PY" util/patch_final_report.py --dry-run
"$PY" util/patch_final_report.py || { echo "patch_final_report FAILED"; exit 3; }

# 4) git 提交（只在 ~/SGL-Torch 代码仓提交；~/research-work 报告目录非 git 仓，仅写盘）
cd "$SGL" || exit 1
git add research_logs/final_results.json research_logs/final_results_tables.md \
        research_logs/final_report_patch.json research_logs/r82_*.log \
        research_logs/r82_rerun_failed_progress.txt util/collect_final_results.py \
        util/patch_final_report.py 2>/dev/null
git commit -q -m "R85 收尾：R82 官方超参 baseline + embed512 3-seed 定稿自动回填" || echo "SGL commit skipped (nothing new)"

echo "FINALIZE_R85_ALLDONE $(date)"

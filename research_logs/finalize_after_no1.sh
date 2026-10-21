#!/usr/bin/env bash
# =============================================================================
# R99 · finalize_after_no1.sh —— No.1 baseline（HCCF/DCCF/AdaGCL）官方超参
#        masmath 复跑（R95）ALLDONE 后自动定稿 §2.3 No.1 对比表
# =============================================================================
# 只做三件事（守诚实铁律、零造假、零碰 GPU）：
#   1) 轮询 r95_no1_baselines_progress.txt 直到 `NO1_BASELINES_ALLDONE`
#   2) 运行 util/collect_no1_results.py —— 从 r95_*.log 逐项提取 9 个 run 的
#      best R@20/NDCG@20，回填 研究总报告.md §2.3（写盘前自动备份、幂等可重跑）
#   3) git 提交 SGL-Torch 的脚本/日志（研究总报告.md 在 ~/research-work，非 git 仓）
#
# 只占用 CPU、绝不碰 GPU。与 wait_no1_masmath.sh（GPU 训练链）零抢卡。
# 用法：nohup bash research_logs/finalize_after_no1.sh > research_logs/finalize_after_no1.out 2>&1 &
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
SGL=~/SGL-Torch
PROG="$SGL/research_logs/r95_no1_baselines_progress.txt"
MARK="NO1_BASELINES_ALLDONE"

cd "$SGL" || exit 1
echo "BEGIN $(date) [finalize_after_no1] waiting for $MARK"

while ! grep -q "$MARK" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "NO1 ALLDONE detected $(date)"

# 先 dry-run 确认，再正式回填（幂等）
"$PY" util/collect_no1_results.py --dry-run
"$PY" util/collect_no1_results.py || { echo "collect_no1_results FAILED"; exit 2; }

# git 提交（SGL-Torch 代码仓；报告目录非 git 仓，仅写盘）
git add util/collect_no1_results.py research_logs/finalize_after_no1.sh \
        research_logs/no1_results.json \
        research_logs/r95_*.log research_logs/r95_no1_baselines_progress.txt 2>/dev/null
git commit -q -m "R99 收尾：No.1 baseline（HCCF/DCCF/AdaGCL）官方超参 masmath 复跑定稿，§2.3 自动回填" \
  || echo "SGL commit skipped (nothing new)"

echo "FINALIZE_AFTER_NO1_ALLDONE $(date)"

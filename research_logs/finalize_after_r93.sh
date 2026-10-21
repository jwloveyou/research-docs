#!/usr/bin/env bash
# =============================================================================
# R98 · finalize_after_r93.sh —— R93 丢失候选复跑 ALLDONE 后自动定稿 §4 负结果清单
# =============================================================================
# 只做三件事（守诚实铁律、零造假、零碰 GPU）：
#   1) 轮询 r93_lost_candidates_progress.txt 直到 `R93_LOST_CANDIDATES_ALLDONE`
#   2) 运行 util/analyze_r93_candidates.py —— 逐项提取 8 候选 + ctrl 的 best 值、
#      按 > ctrl+0.001 阈值判定、把 §4 负结果清单「8 候选未出值」占位段安全回填
#      （写盘前自动备份、幂等可重跑）
#   3) git 提交 SGL-Torch 的脚本/日志（研究总报告.md 在 ~/research-work，非 git 仓）
#
# 只占用 CPU、绝不碰 GPU。与 wait_no1_masmath.sh（GPU 训练链）零抢卡。
# 用法：nohup bash research_logs/finalize_after_r93.sh > research_logs/finalize_after_r93.out 2>&1 &
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
SGL=~/SGL-Torch
PROG="$SGL/research_logs/r93_lost_candidates_progress.txt"
MARK="R93_LOST_CANDIDATES_ALLDONE"

cd "$SGL" || exit 1
echo "BEGIN $(date) [finalize_after_r93] waiting for $MARK"

while ! grep -q "$MARK" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "R93 ALLDONE detected $(date)"

# 先 dry-run 确认，再正式回填（幂等）
"$PY" util/analyze_r93_candidates.py --dry-run
"$PY" util/analyze_r93_candidates.py || { echo "analyze_r93_candidates FAILED"; exit 2; }

# git 提交（SGL-Torch 代码仓；报告目录非 git 仓，仅写盘）
git add util/analyze_r93_candidates.py research_logs/finalize_after_r93.sh \
        research_logs/r93_candidates_analysis.json \
        research_logs/r93_gowalla_*.log research_logs/r93_lost_candidates_progress.txt 2>/dev/null
git commit -q -m "R98 收尾：R93 丢失候选 60ep 决定性复跑定稿，§4 负结果清单自动回填" \
  || echo "SGL commit skipped (nothing new)"

echo "FINALIZE_AFTER_R93_ALLDONE $(date)"

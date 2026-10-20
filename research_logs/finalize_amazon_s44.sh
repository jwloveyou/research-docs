#!/usr/bin/env bash
# =============================================================================
# R89 · finalize_amazon_s44.sh —— amazon 256→512 显著性 2-seed→3-seed 无人值守升级
# =============================================================================
# 只做一件事（守诚实铁律、零造假、零碰 GPU）：
#   1) 轮询 r82_rerun_failed_progress.txt 直到 amazon-book embed512 s44 完成
#      （标记 `END   amazon-book_emb512_s44`；若整链先 ALLDONE 也触发）
#   2) 运行 util/finalize_amazon_s44.py —— 从日志逐项读 s44 + 算 3-seed 显著性，
#      把 §2.4 表两行 2-seed→3-seed + 结论行回填（写盘前自动备份、幂等）
#   3) git 提交 SGL-Torch 的脚本/日志（研究总报告.md 在 ~/research-work，非 git 仓）
#
# 与 finalize_after_r82.sh（改 §2.3/§2.1/§3.2）零重叠、可并行。只占 CPU。
# 用法：nohup bash research_logs/finalize_amazon_s44.sh > research_logs/finalize_amazon_s44.out 2>&1 &
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
SGL=~/SGL-Torch
PROG="$SGL/research_logs/r82_rerun_failed_progress.txt"
OUT="$SGL/research_logs/finalize_amazon_s44.out"
MARK_S44="END   amazon-book_emb512_s44"
MARK_ALL="R82_RERUN_ALLDONE"

cd "$SGL" || exit 1
echo "BEGIN $(date) [finalize_amazon_s44] waiting for amazon-book_emb512_s44 (or R82 ALLDONE)"

# 等 s44 完成（每 3 分钟查一次）；s44 在 amazon-lightgcn 之前，故通常早于 ALLDONE
while ! grep -qF "$MARK_S44" "$PROG" 2>/dev/null && ! grep -qF "$MARK_ALL" "$PROG" 2>/dev/null; do
  sleep 180
done
echo "amazon-book_emb512_s44 marker detected $(date)"

# 跑 3-seed 升级脚本（幂等；s44 未出值时会干净退出）
"$PY" util/finalize_amazon_s44.py || { echo "finalize_amazon_s44 FAILED"; exit 2; }

# git 提交（只提交脚本 + 相关日志；报告目录非 git 仓）
git add util/finalize_amazon_s44.py research_logs/finalize_amazon_s44.sh \
        research_logs/r82_amazon-book_emb512_s44.log 2>/dev/null
git commit -q -m "R89: amazon 256→512 显著性自动升级 2-seed→3-seed（s44 收齐后自动回填 §2.4）" \
  || echo "SGL commit skipped (nothing new)"

echo "FINALIZE_AMAZON_S44_ALLDONE $(date)"

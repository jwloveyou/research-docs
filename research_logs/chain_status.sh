#!/usr/bin/env bash
# =============================================================================
# chain_status.sh —— scnumath 被封后的 masmath 单卡「收尾管线」一眼快照。
#
# 用途：10-22 死线前，一个命令看清「R93 丢失候选复跑 → No.1 baseline 官方超参
#       复跑 → 四个 finalizer」这条关键路径的健康状态，避免逐个读 progress 文件。
#
# 纯 CPU、不碰 GPU、不抢卡、不启动任何训练。用法：bash research_logs/chain_status.sh
# =============================================================================
cd ~/SGL-Torch/research_logs 2>/dev/null || exit 1

echo "========== $(date '+%F %T') masmath 单卡收尾管线快照 =========="
echo ""
echo "[GPU0]"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0 2>/dev/null \
  | awk '{printf "  used=%s MiB  total=%s MiB  util=%s\n", $1, $3, $5}' || echo "  nvidia-smi 不可用"

echo ""
echo "[R93 丢失候选复跑]  (目标: 8 候选 + ctrl，各 60ep，之后写 R93_LOST_CANDIDATES_ALLDONE)"
if [ -f r93_lost_candidates_progress.txt ]; then
  echo "  完成: $(grep -c '^END ' r93_lost_candidates_progress.txt 2>/dev/null)/9"
  grep '^END ' r93_lost_candidates_progress.txt 2>/dev/null | tail -2 | sed 's/^/  /'
  grep -q 'R93_LOST_CANDIDATES_ALLDONE' r93_lost_candidates_progress.txt && echo "  >> R93 已 ALLDONE"
else
  echo "  (未开始)"
fi

echo ""
echo "[No.1 baseline 官方超参复跑]  (HCCF→DCCF→AdaGCL × 3 集，之后写 NO1_BASELINES_ALLDONE)"
if [ -f r95_no1_baselines_progress.txt ] && [ -s r95_no1_baselines_progress.txt ]; then
  echo "  完成: $(grep -c '^END ' r95_no1_baselines_progress.txt 2>/dev/null)/9"
  grep '^END ' r95_no1_baselines_progress.txt 2>/dev/null | tail -2 | sed 's/^/  /'
  grep -q 'NO1_BASELINES_ALLDONE' r95_no1_baselines_progress.txt && echo "  >> No.1 已 ALLDONE"
else
  echo "  (等待 R93 完成)"
fi

echo ""
echo "[finalizer 值守进程]"
for spec in \
  "5667 run_r93_lost_candidates.sh(R93链本体)" \
  "78210 finalize_after_r93.sh(等R93→回填§4)" \
  "56849 wait_no1_masmath.sh(等R93→启动No.1)" \
  "87771 finalize_after_no1.sh(等No.1→回填§2.3)"; do
  pid=${spec%% *}; name=${spec#* }
  if ps -p "$pid" >/dev/null 2>&1; then
    printf "  [ALIVE] PID %-6s %s\n" "$pid" "$name"
  else
    printf "  [DONE ] PID %-6s %s\n" "$pid" "$name"
  fi
done

echo ""
echo "[下一事件预估]"
if grep -q 'R93_LOST_CANDIDATES_ALLDONE' r93_lost_candidates_progress.txt 2>/dev/null; then
  grep -q 'NO1_BASELINES_ALLDONE' r95_no1_baselines_progress.txt 2>/dev/null \
    && echo "  全链已 ALLDONE，等 finalize_after_no1 回填 §2.3" \
    || echo "  No.1 baseline 训练中（HCCF 先出值）"
else
  echo "  R93 收尾（dropedge09→init2）→ finalize_after_r93 回填 §4 + wait_no1 启动 No.1"
fi
echo ""
echo "================================================================"

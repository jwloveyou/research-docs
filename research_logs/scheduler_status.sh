#!/usr/bin/env bash
# =============================================================================
# scheduler_status.sh —— 跨机 GPU 自动调度器（R108）健康一眼快照（R111）。
#
# 背景：chain_status.sh（R100）面向「R93 丢失候选复跑 → No.1 baseline」这条
#   masmath 单卡链，现已过时（R93 ALLDONE 01:31、No.1 ALLDONE 10-23，均已收口）。
#   当前唯一关键路径是 auto_scheduler.sh 跨机调度 15 个经典 baseline
#   （DGCF/NCL/DirectAU/RecDCL/UltraGCN × 3 集）+ finalize_after_classic.sh 值守回填。
#   本脚本面向这条新路径，只读不写、纯 CPU、不碰 GPU、不启动任何训练。
#
# 用法：bash research_logs/scheduler_status.sh
# =============================================================================
set -u
cd ~/SGL-Torch || exit 1
SCN="zhangxu@222.201.80.152"
SSHOPT="-o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "========== $(date '+%F %T') 跨机调度器 + 经典 baseline 收尾管线快照 =========="

echo ""
echo "[调度器进程]"
if pgrep -f 'auto_scheduler.sh' >/dev/null 2>&1; then
  echo "  [ALIVE] auto_scheduler.sh PID=$(pgrep -f 'auto_scheduler.sh' | head -1)"
else
  echo "  [DEAD ] auto_scheduler.sh 未运行（调度器挂了，需重启）"
fi

echo ""
echo "[在跑任务 state]"
STATE="research_logs/auto_scheduler_state.txt"
if [ -s "$STATE" ]; then
  n=$(grep -c '|' "$STATE")
  echo "  state 记录 $n 个在跑任务："
  while IFS='|' read -r key card _; do
    [ -n "${key:-}" ] || continue
    printf "    %-20s @ %s\n" "$key" "$card"
  done < "$STATE"
  # 核对每个 state 任务是否真有 python 在跑
  echo "  本地 python 进程核对："
  for key in $(cut -d'|' -f1 "$STATE"); do
    if pgrep -f "dataset=${key%_*}.*${key#*_}=1" >/dev/null 2>&1 || \
       pgrep -f "dataset=$(echo "$key" | sed 's/_[a-z0-9]*$//')" >/dev/null 2>&1; then
      echo "    [RUN ] $key"
    else
      echo "    [????] $key（本地未见 python，可能跑在 scnumath）"
    fi
  done
else
  echo "  state 为空（无在跑任务或调度器刚重启）"
fi

echo ""
echo "[队列]"
QUEUE="research_logs/baseline_queue.txt"
if [ -s "$QUEUE" ]; then
  n=$(grep -cvE '^[[:space:]]*(#|$)' "$QUEUE")
  echo "  剩余 $n 个任务，队头：$(grep -vE '^[[:space:]]*(#|$)' "$QUEUE" | head -1 | awk '{print $1"/"$2}')"
else
  echo "  队列已空"
fi

echo ""
echo "[经典 baseline 进度]（DGCF/NCL/DirectAU/RecDCL/UltraGCN × 3 集，共 15 任务由调度器管）"
PROG="research_logs/r102_classic_baselines_progress.txt"
if [ -f "$PROG" ]; then
  echo "  START 计数: $(grep -c 'START ' "$PROG")  END 计数: $(grep -c '^END ' "$PROG")"
  echo "  已收 best（近 5 条）："
  grep -B0 -A0 'best_result' "$PROG" 2>/dev/null | grep -E 'best_result|0\.' | tail -5 | sed 's/^/    /'
  grep -q 'CLASSIC_BASELINES_ALLDONE' "$PROG" && echo "  >> CLASSIC_BASELINES_ALLDONE 已写" || echo "  >> 尚未 ALLDONE"
else
  echo "  (progress 文件不存在)"
fi

echo ""
echo "[完成标记 scheduler_done]"
echo "  masmath: $(ls research_logs/scheduler_done/*.DONE 2>/dev/null | wc -l) 个"
echo "  scnumath: $(ssh $SSHOPT "$SCN" 'ls ~/SGL-Torch/research_logs/scheduler_done/*.DONE 2>/dev/null | wc -l' 2>/dev/null || echo '?') 个"

echo ""
echo "[finalize 值守进程]"
for spec in "78025 finalize_after_classic.sh(等CLASSIC_ALLDONE→collect→回填§2.3→commit)"; do
  pid=${spec%% *}; name=${spec#* }
  if ps -p "$pid" >/dev/null 2>&1; then
    printf "  [ALIVE] PID %-6s %s\n" "$pid" "$name"
  else
    printf "  [DONE ] PID %-6s %s\n" "$pid" "$name"
  fi
done

echo ""
echo "[GPU 占用]"
echo "  masmath GPU0:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0 2>/dev/null \
  | awk '{printf "    used=%s MiB  total=%s MiB  util=%s\n", $1, $3, $5}'
echo "  scnumath GPU4/5/6/7:"
ssh $SSHOPT "$SCN" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 4,5,6,7" 2>/dev/null \
  | awk -F',' '{gsub(/^ +/,"",$2); gsub(/^ +/,"",$3); printf "    GPU%s used=%s util=%s\n", $1, $2, $3}' || echo "    (ssh 不可达)"

echo ""
echo "[下一事件预估]"
if grep -q 'CLASSIC_BASELINES_ALLDONE' "$PROG" 2>/dev/null; then
  echo "  调度器已 ALLDONE，finalize_after_classic.sh 应已触发回填 §2.3"
else
  echo "  等 gowalla_dgcf(200ep)/yelp2018_dgcf(100ep) 完成 → 调度器 harvest → 续跑队列头"
fi
echo ""
echo "====================================================================="

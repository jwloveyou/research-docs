#!/usr/bin/env bash
# =============================================================================
# R95 · wait_no1_masmath.sh —— R93 丢失候选复跑 ALLDONE 后，自动在 GPU0 排队
#       启动 No.1 baseline（HCCF/DCCF/AdaGCL）官方超参复跑（task#8 收口）。
#
# 双门守卫（守 R52/R34 教训，防抢卡 + 防等不存在标记）：
#   1) R93 完成标记：research_logs/r93_lost_candidates_progress.txt 含
#      `R93_LOST_CANDIDATES_ALLDONE`（run_r93_lost_candidates.sh 末尾写）；
#   2) GPU0 真空闲：连续两次（间隔 120s）显存 <1000MiB（真空闲 ~0-500MiB，
#      最小实跑 ~1442MiB，1000 可安全分隔，绝不复现 R48 <3000 误判抢卡）。
#
# 串行、不共驻（R82 教训）。只占 CPU，启动后由 run_no1_baselines_masmath.sh
# 自己串行跑 9 个 run 并写 NO1_BASELINES_ALLDONE 标记。
# 用法：nohup bash research_logs/wait_no1_masmath.sh > research_logs/wait_no1_masmath.out 2>&1 &
# =============================================================================
set -u
cd ~/SGL-Torch || exit 1

PROG="research_logs/r93_lost_candidates_progress.txt"
GPU=0

echo "BEGIN $(date) [wait_no1_masmath] waiting for R93_LOST_CANDIDATES_ALLDONE + GPU0 idle (<1000MiB x2)"

# 门 1：等 R93 丢失候选复跑完成（每 5 分钟查一次）
while ! grep -q "R93_LOST_CANDIDATES_ALLDONE" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "R93_LOST_CANDIDATES_ALLDONE detected $(date)"

# 门 2：等 GPU0 真空闲（连续两次 <1000MiB，间隔 120s）
while true; do
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
  sleep 120
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null)
  if [ "${m1:-99999}" -lt 1000 ] && [ "${m2:-99999}" -lt 1000 ]; then
    echo "GPU0 idle (${m1}MiB -> ${m2}MiB) $(date)"
    break
  fi
  sleep 180
done

echo "launching run_no1_baselines_masmath.sh $(date)"
bash research_logs/run_no1_baselines_masmath.sh > research_logs/r95_no1_launcher.out 2>&1

echo "WAIT_NO1_MASMATH_DONE $(date)"

#!/usr/bin/env bash
# =============================================================================
# R102 · wait_classic_masmath.sh —— No.1 baseline（HCCF/DCCF/AdaGCL）官方超参
#        masmath 复跑（R95）ALLDONE 后，自动在 GPU0 排队启动 7 个经典 baseline
#       （LightGCL/NGCF/UltraGCN/DGCF/NCL/DirectAU/RecDCL）官方超参复跑（task#10 收口）。
#
# 双门守卫（守 R52/R34 教训，防抢卡 + 防等不存在标记）：
#   1) No.1 完成标记：research_logs/r95_no1_baselines_progress.txt 含
#      `NO1_BASELINES_ALLDONE`（run_no1_baselines_masmath.sh 末尾写）；
#   2) GPU0 真空闲：连续两次（间隔 120s）显存 <1000MiB（真空闲 ~0-500MiB，
#      最小实跑 ~1442MiB，1000 可安全分隔，绝不复现 R48 <3000 误判抢卡）。
#
# 串行、不共驻（R82 教训）。只占 CPU，启动后由 run_classic_baselines_masmath.sh
# 自己串行跑 21 个 run 并写 CLASSIC_BASELINES_ALLDONE 标记。
# 用法：nohup bash research_logs/wait_classic_masmath.sh > research_logs/wait_classic_masmath.out 2>&1 &
# =============================================================================
set -u
cd ~/SGL-Torch || exit 1

PROG="research_logs/r95_no1_baselines_progress.txt"
GPU=0

echo "BEGIN $(date) [wait_classic_masmath] waiting for NO1_BASELINES_ALLDONE + GPU0 idle (<1000MiB x2)"

# 门 1：等 No.1 baseline 复跑完成（每 5 分钟查一次）
while ! grep -q "NO1_BASELINES_ALLDONE" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "NO1_BASELINES_ALLDONE detected $(date)"

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

echo "launching run_classic_baselines_masmath.sh $(date)"
bash research_logs/run_classic_baselines_masmath.sh > research_logs/r102_classic_launcher.out 2>&1

echo "WAIT_CLASSIC_MASMATH_DONE $(date)"

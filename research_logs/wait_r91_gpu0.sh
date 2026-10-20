#!/usr/bin/env bash
# =============================================================================
# R92 · wait_r91_gpu0.sh —— R82 官方超参 baseline 重跑链收尾后，自动在 GPU0 排队
#       启动 R91 用户侧冻结语义消融（第 5 组件候选·语义轴双侧补全）。
#
# 双门守卫（守 R52/R34 教训，防抢卡 + 防等不存在标记）：
#   1) R82 全链完成标记：research_logs/r82_rerun_failed_progress.txt 含
#      `R82_RERUN_ALLDONE`（R82 链最后一个任务 amazon-lightgcn 结束后才写）；
#   2) GPU0 真空闲：连续两次（间隔 120s）显存 <1000MiB（真空闲 ~0-500MiB，
#      最小实跑 ~1442MiB，1000 可安全分隔，绝不复现 R48 <3000 误判抢卡）。
#
# 串行、不与其他任务共驻（R82 教训：并行共驻曾致 CUDA OOM + 磁盘配额失败）。
# 只占 CPU（轮询 + 拉起），启动后由 run_r91_user_sem.sh 自己串行跑 7 个 run 并
# 写 R91_USER_SEM_ALLDONE 标记。
# 用法：nohup bash research_logs/wait_r91_gpu0.sh > research_logs/wait_r91_gpu0.out 2>&1 &
# =============================================================================
set -u
cd ~/SGL-Torch || exit 1

PROG="research_logs/r82_rerun_failed_progress.txt"
GPU=0

echo "BEGIN $(date) [wait_r91_gpu0] waiting for R82_RERUN_ALLDONE + GPU0 idle (<1000MiB x2)"

# 门 1：等 R82 全链完成（每 5 分钟查一次）
while ! grep -q "R82_RERUN_ALLDONE" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "R82_RERUN_ALLDONE detected $(date)"

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

echo "launching run_r91_user_sem.sh $(date)"
bash research_logs/run_r91_user_sem.sh > research_logs/r91_launcher.out 2>&1

echo "WAIT_R91_GPU0_DONE $(date)"

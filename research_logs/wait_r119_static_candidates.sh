#!/usr/bin/env bash
# =============================================================================
# R119 · wait_r119_static_candidates.sh —— baseline 队列空后，自动在 masmath GPU0
#   跑完三个「范数=流行度」静态第 4 组件候选（R113/R115/R119，共 9 run ~7h）。
#
# 双门守卫（守 R52/R34 教训，防抢卡 + 防等不存在标记）：
#   1) 经典 baseline 全完成：`CLASSIC_BASELINES_ALLDONE` 标记（跨机调度器
#      auto_scheduler.sh 队列空 + state 空时写入，写后调度器 exit 0）；
#   2) GPU0 真空闲：连续两次（间隔 120s）显存 <1000MiB（真空闲 ~0-500MiB，
#      最小实跑 ~1442MiB，1000 可安全分隔，绝不复现 R48 <3000 误判抢卡）。
#
# 串行、不与其他任务共驻（R82 教训：并行共驻曾致 CUDA OOM + 磁盘配额失败）。
# 只占 CPU（轮询 + 拉起），启动后由 run_r119_static_candidates.sh 自己串行跑
# 9 个 run 并写 R119_STATIC_CANDIDATES_ALLDONE。
# 用法：nohup bash research_logs/wait_r119_static_candidates.sh \
#         > research_logs/wait_r119_static_candidates.out 2>&1 &
# =============================================================================
set -u
cd ~/SGL-Torch || exit 1

# 门 1 标记：与 finalize_after_classic.sh 共用同一个「baseline 全完成」信号。
# 优先从 scheduler 的 classic progress 里找；兼容 finalize 后仍保留该标记。
PROG="research_logs/r102_classic_baselines_progress.txt"
GPU=0

echo "BEGIN $(date) [wait_r119_static_candidates] waiting for CLASSIC_BASELINES_ALLDONE + GPU0 idle"

# 门 1：等经典 baseline 全完成（每 5 分钟查一次；标记可能写进 progress 或 scheduler 日志）
while ! grep -q "CLASSIC_BASELINES_ALLDONE" "$PROG" 2>/dev/null; do
  sleep 300
done
echo "CLASSIC_BASELINES_ALLDONE detected $(date)"

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

echo "launching run_r119_static_candidates.sh $(date)"
bash research_logs/run_r119_static_candidates.sh > research_logs/r119_static_launcher.out 2>&1

echo "WAIT_R119_STATIC_CANDIDATES_DONE $(date)"

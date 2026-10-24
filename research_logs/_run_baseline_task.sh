#!/usr/bin/env bash
# =============================================================================
# _run_baseline_task.sh —— 单个 baseline 任务的运行器（masmath / scnumath 通用）
#
# 用法: _run_baseline_task.sh <gpu_id> <tag> <ds> <eps> <extra args...>
#
# 职责（守「训练→收值→回填」无人值守管线 + 跨机自动调度 R108）：
#   1) 在指定 GPU 上跑 main.py（官方超参经 <extra args...> 传入）
#   2) 训练日志写 research_logs/r102_<ds>_<tag>.log（finalizer 的 collect_classic_results.py 读这个名）
#   3) 结束后写 DONE 标记 research_logs/scheduler_done/<ds>_<tag>.DONE
#      （含 END rc + best_result 摘要 + ALLDONE 行），供调度器采集。
#
# 双机通用：PY 按本机路径自动探测（masmath=/home/Yehaikun, scnumath=/home/zhangxu）。
# main.py 内部已 os.environ['CUDA_VISIBLE_DEVICES']=gpu_id，故此处只需 --gpu_id，
# 不再额外设 CUDA_VISIBLE_DEVICES（避免 double-remap 到错误的设备号）。
# =============================================================================
set -u

GPU=$1; TAG=$2; DS=$3; EPS=$4; shift 4

if [ -x /home/Yehaikun/miniconda3/envs/sgl/bin/python ]; then
  PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
elif [ -x /home/zhangxu/miniconda3/envs/sgl/bin/python ]; then
  PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
else
  PY=python
fi

cd ~/SGL-Torch || exit 1
LOG="research_logs/r102_${DS}_${TAG}.log"
DONEDIR="research_logs/scheduler_done"
mkdir -p "$DONEDIR"
DONE="$DONEDIR/${DS}_${TAG}.DONE"

# 清理旧的 DONE 标记，避免调度器误判（重跑时覆盖）
rm -f "$DONE"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" main.py \
  --recommender=NoiseSGL --dataset="$DS" --seed=42 --epochs="$EPS" --verbose=1 \
  --stop_cnt=1000 --gpu_id="$GPU" --use_mlc=0 --ssl_anneal=0 --ssl_anneal_epochs=0 \
  --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 "$@" > "$LOG" 2>&1
rc=$?

{
  echo "END ${DS}_${TAG} rc=$rc $(date)"
  echo "--- ${DS}_${TAG} best ---"
  grep "best_result" -A3 "$LOG" | tail -4
  echo "ALLDONE ${DS}_${TAG} rc=$rc $(date)"
} > "$DONE"

exit 0

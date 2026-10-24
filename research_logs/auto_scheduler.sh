#!/usr/bin/env bash
# =============================================================================
# auto_scheduler.sh —— 跨机 GPU 自动调度器（R108，用户 2026-10-25 指令）
#
# 同时监控 masmath GPU0 + scnumath GPU4/5/6/7（共 5 张卡），一有卡空闲
# （<1000MiB，守 R52 教训，绝不复现 R48 <3000 误判抢卡）就自动取队列头的
# baseline 任务在空闲卡上开跑，跑完自动记 ALLDONE + 继续下一个，直到
# 剩余 baseline（DGCF/NCL/DirectAU/RecDCL）全部复现完。
#
# 调度策略：
#   - 优先用 scnumath 的 4 卡并行（快 4 倍）；masmath GPU0 作为第 5 个 worker
#     （顺序在 scnumath 之后，避免占用 masmath 跑融合/消融实验的单卡）。
#   - 代码同步：启动前 + 每个 remote 任务收尾时用 rsync（排除 dataset/ 与
#     大日志，见 sync_code / harvest 里的 rsync）。
#
# 状态/标记（守「训练→收值→回填」无人值守管线）：
#   - 队列:   research_logs/baseline_queue.txt（每行一个任务，调度时取队头）
#   - 运行态: research_logs/auto_scheduler_state.txt（key|host|gpu 每行一个在跑任务）
#   - 完成标记: research_logs/scheduler_done/<ds>_<tag>.DONE（runner 写完即完成）
#   - 进度:   research_logs/r102_classic_baselines_progress.txt（追加 START/END/best，
#             末尾写 CLASSIC_BASELINES_ALLDONE，触发既有 finalize_after_classic.sh）
#   - 调度日志: research_logs/auto_scheduler.log
#
# 用法（masmath 上，setsid 后台）：
#   setsid nohup bash research_logs/auto_scheduler.sh \
#     > research_logs/auto_scheduler.out 2>&1 &
# =============================================================================
set -u

cd ~/SGL-Torch || exit 1
SCN="zhangxu@222.201.80.152"
QUEUE="research_logs/baseline_queue.txt"
STATE="research_logs/auto_scheduler_state.txt"
PROG="research_logs/r102_classic_baselines_progress.txt"
ALOG="research_logs/auto_scheduler.log"
DONEDIR="research_logs/scheduler_done"
SSHOPT="-o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes"

mkdir -p "$DONEDIR"

# 调度优先级：scnumath 4 卡并行在前，masmath GPU0 兜底
CARDS="scnumath:4 scnumath:5 scnumath:6 scnumath:7 local:0"
POLL=150   # 秒，每轮监控间隔

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$ALOG"; }
echo "===== auto_scheduler BEGIN $(date) =====" >> "$ALOG"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
host_of() { echo "${1%%:*}"; }   # card spec "host:gpu" -> host
gpu_of()  { echo "${1##*:}"; }   # card spec "host:gpu" -> gpu

# 判空闲：<1000MiB（真空闲 ~0-500MiB；最小实跑 ~1442MiB，1000 安全分隔）
is_free() {
  local card=$1; local host; local gpu; local mem
  host=$(host_of "$card"); gpu=$(gpu_of "$card")
  if [ "$host" = "local" ]; then
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
  else
    mem=$(ssh $SSHOPT "$SCN" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu" 2>/dev/null)
  fi
  mem=${mem:-99999}
  [ "$mem" -lt 1000 ] 2>/dev/null
}

# 判某任务是否已完成（本机或 scnumath 上存在 DONE 标记）
is_done() {
  local key=$1
  [ -f "$DONEDIR/${key}.DONE" ] && return 0
  ssh $SSHOPT "$SCN" "test -f ~/SGL-Torch/research_logs/scheduler_done/${key}.DONE" 2>/dev/null
}

# 卡是否正被本调度器占用（在 state 里）
card_busy() {
  local card=$1
  grep -q "^[^|]*|${card}$" "$STATE" 2>/dev/null
}

# ---------------------------------------------------------------------------
# 同步代码（排除 dataset/ 与大日志；.git 也排除）
# ---------------------------------------------------------------------------
sync_code() {
  log "rsync code -> scnumath (exclude dataset/research_logs/.git)"
  rsync -av --exclude='dataset/' --exclude='research_logs/' --exclude='.git/' \
    ~/SGL-Torch/ "$SCN":~/SGL-Torch/ >> "$ALOG" 2>&1 || log "WARN rsync code failed"
}

# ---------------------------------------------------------------------------
# 派发一个任务到一张卡
#   入参: $1=card spec, $2=tag, $3=ds, $4=eps, 其余=extra args
#   成功返回 0 并写 state；失败返回非 0（不写 state、不删队列）
# ---------------------------------------------------------------------------
dispatch() {
  local card=$1 tag=$2 ds=$3 eps=$4; shift 4
  local host gpu key
  host=$(host_of "$card"); gpu=$(gpu_of "$card")
  key="${ds}_${tag}"

  # 派发前清掉可能残留的 DONE 标记（防旧标记误判）
  rm -f "$DONEDIR/${key}.DONE"
  ssh $SSHOPT "$SCN" "rm -f ~/SGL-Torch/research_logs/scheduler_done/${key}.DONE" 2>/dev/null

  if [ "$host" = "local" ]; then
    nohup bash research_logs/_run_baseline_task.sh "$gpu" "$tag" "$ds" "$eps" "$@" \
      >> "$ALOG" 2>&1 &
    local pid=$!
    # 稍等确认进程没立刻崩
    sleep 5
    if kill -0 "$pid" 2>/dev/null || ps -p "$pid" >/dev/null 2>&1; then
      echo "${key}|${card}" >> "$STATE"
      echo "START ${ds}_${tag} gpu=$gpu ep=$eps [scheduler:local] $(date)" >> "$PROG"
      log "dispatch ${key} -> ${card} (pid=$pid)"
      return 0
    else
      log "WARN dispatch local ${key} failed (pid=$pid 已退出)"
      return 1
    fi
  else
    # remote：ssh 里 nohup 后台启动，返回后不阻塞
    local cmd
    # cd/mkdir 的 stderr 重定向到 /dev/null（否则后台 job 持有 ssh 通道、ssh 卡住不返回），
    # nohup 的 stdout/stderr 重定向到 launch.out、stdin 重定向 /dev/null（完全脱离 ssh 会话）
    cmd="cd ~/SGL-Torch 2>/dev/null && mkdir -p research_logs/scheduler_done 2>/dev/null && nohup bash research_logs/_run_baseline_task.sh $gpu $tag $ds $eps $@ > research_logs/scheduler_done/${key}.launch.out 2>&1 < /dev/null &"
    if ssh $SSHOPT "$SCN" "$cmd" 2>/dev/null; then
      echo "${key}|${card}" >> "$STATE"
      echo "START ${ds}_${tag} gpu=$gpu ep=$eps [scheduler:scnumath] $(date)" >> "$PROG"
      log "dispatch ${key} -> ${card} (scnumath)"
      return 0
    else
      log "WARN dispatch scnumath ${key} failed"
      return 1
    fi
  fi
}

# ---------------------------------------------------------------------------
# 采集已完成任务：把 remote 日志 rsync 回 masmath，END/best 追加进进度文件，
# 并从 state 移除。
# ---------------------------------------------------------------------------
harvest() {
  local key=$1 card=$2
  local host gpu ds tag
  host=$(host_of "$card"); gpu=$(gpu_of "$card")
  ds="${key%_*}"; tag="${key#*_}"

  if [ "$host" != "local" ]; then
    # 拉回训练日志 + DONE 标记内容
    rsync -av "$SCN":~/SGL-Torch/research_logs/r102_${ds}_${tag}.log \
      ~/SGL-Torch/research_logs/ >> "$ALOG" 2>&1 || log "WARN rsync log ${key} failed"
  fi

  # 从 DONE 标记（本机）取 END/best 追加进进度文件
  if [ -f "$DONEDIR/${key}.DONE" ]; then
    cat "$DONEDIR/${key}.DONE" >> "$PROG"
  else
    # remote 场景：ssh cat
    ssh $SSHOPT "$SCN" "cat ~/SGL-Torch/research_logs/scheduler_done/${key}.DONE" >> "$PROG" 2>/dev/null \
      || echo "END ${ds}_${tag} rc=? $(date)" >> "$PROG"
  fi
  log "harvest ${key} done from ${card}"

  # 从 state 移除该任务
  local tmp
  tmp=$(mktemp)
  grep -v "^${key}|" "$STATE" > "$tmp" 2>/dev/null
  mv "$tmp" "$STATE"
}

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
# 初始代码同步（用户指令：代码同步用 rsync，排除 dataset/research_logs）
sync_code

# 若 state 文件不存在则建空
[ -f "$STATE" ] || : > "$STATE"

# 建 DONE 目录（本机 + scnumath）。⚠️ 不清理残留 DONE 标记——R120 修复：
# 原「rm -f *.DONE」会在调度器重启时清掉「停机期间已完成任务」的 DONE 标记，
# 使该任务永远不被 harvest（state 非空 → ALLDONE 永不触发 → 死锁）。
# 正确做法：DONE 标记由 runner 自清理（启动前 rm -f 自己的 DONE）+ dispatch 派发
# 前按 key 清理；停机期间完成的 state 任务，由主循环第 1 步（harvest）在重启后
# 首个迭代自动采集，无需清标记。
mkdir -p "$DONEDIR"
ssh $SSHOPT "$SCN" "mkdir -p ~/SGL-Torch/research_logs/scheduler_done" 2>/dev/null

log "queue head: $(head -1 "$QUEUE" | awk '{print $1"/"$2}')"

# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
loop=0
while true; do
  loop=$((loop+1))
  # 周期性代码同步（每 20 轮≈50min 重同步；防「调度器启动后修 bug/改代码不达 scnumath」——
  # R109/R113/R115 曾靠手动 rsync 兜底，本修复让后续改动自动传播，零抢卡）
  if [ $((loop % 20)) -eq 0 ]; then
    sync_code
  fi

  # 1) 采集已完成任务（释放卡）
  if [ -s "$STATE" ]; then
    while IFS='|' read -r key card _rest; do
      [ -n "${key:-}" ] || continue
      if is_done "$key"; then
        harvest "$key" "$card"
      fi
    done < "$STATE"
  fi

  # 2) 派发新任务到空闲卡
  for card in $CARDS; do
    # 队列空则停止派发
    [ -s "$QUEUE" ] || break
    # 该卡正跑本调度器的任务则跳过
    card_busy "$card" && continue
    # 卡空闲才派发
    if is_free "$card"; then
      line=$(grep -vE '^[[:space:]]*(#|$)' "$QUEUE" | head -1)
      [ -n "$line" ] || continue
      # 解析行: tag ds eps rest...
      read -r tag ds eps rest <<< "$line"
      if dispatch "$card" "$tag" "$ds" "$eps" $rest; then
        # 派发成功：队头出队（保留注释行）
        sed -i '1d' "$QUEUE"
      else
        # 派发失败：不删队头，下轮重试
        log "dispatch fail, retry later: ${tag}/${ds}"
      fi
    fi
  done

  # 3) 全部完成判定
  if [ ! -s "$QUEUE" ] && [ ! -s "$STATE" ]; then
    echo "CLASSIC_BASELINES_ALLDONE $(date)" >> "$PROG"
    echo "===== auto_scheduler ALLDONE $(date) =====" >> "$ALOG"
    log "ALL 12 remaining baselines done; wrote CLASSIC_BASELINES_ALLDONE"
    exit 0
  fi

  sleep "$POLL"
done

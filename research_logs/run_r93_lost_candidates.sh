#!/usr/bin/env bash
# =============================================================================
# R93 · scnumath 丢失的 8 个静态第4组件候选 —— masmath 单卡决定性复跑
#   （honesty gap 收口：这些候选在 R31b/R37/R38/R39/R40/R41/R49/R60 已实现并
#    排队于 scnumath GPU4 链，但随 10-19 scnumath 封禁一并丢失，从未被评估。
#    此前进度报告把它们与「已测并判负」的候选一起计入「关闭」，属诚实性缺口。）
#
# 协议（与 r34/r35/r49/r58/r60 的 4th-component 测试台逐项一致）：
#   gowalla / embed256 / win30 余弦退火(ssl_anneal=2, ssl_anneal_epochs=30,
#   floor=0) / view_mix=nn / noise_eps=0.2 / ssl_reg=0.1 / seed=42 /
#   best-over-epochs / 全量排序 + NDCG@20 / stop_cnt=1000 / 60ep（直抵 gowalla
#   真峰 ep61 附近，与 r34/r35 的 60ep 定案协议一致）。
#
# 8 个候选（对应总报告 §4 记录的 scnumath 丢失 8 候选，每轴 1 个决定性取值，
# faithful 到原轮次脚本）：
#   1 ctrl          （无开关，重立 masmath 同机基线）
#   2 tailboost     (R31b, LAGCL tail 增强)       --noise_deg_gamma=-0.5
#   3 lr_anneal     (R37, cos win100 floor0.1)   --lr_anneal=2 --lr_anneal_epochs=100 --lr_floor=0.1
#   4 weight_decay  (R38, Adam 解耦衰减 1e-4)     --weight_decay=1e-4
#   5 agg_weights   (R39, 浅层偏向池化)           --agg_weights=shallow
#   6 swa_start     (R40, BPR 精调相权重平均)     --swa_start=30
#   7 bpr_margin    (R41, 排序 margin 0.1)        --bpr_margin=0.1
#   8 dropedge_keep (R49, 主分支边丢弃 keep0.9)   --dropedge_keep=0.9
#   9 init_scale    (R60, 嵌入初始化缩放 2.0)     --init_scale=2.0
#
# 串行、单卡、不共驻（R82 教训：并行共驻曾致 CUDA OOM + 磁盘配额失败）。
# 只由 wait_r93_gpu0.sh 在 R91 用户侧语义消融 ALLDONE 后拉起。
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L=research_logs/r93_lost_candidates_progress.txt
echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  tag=$1; extra=$2
  log="research_logs/r93_gowalla_${tag}.log"
  echo "START ${tag} gpu=$GPU ep=60 ${extra} $(date)" >> "$L"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=60 --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --embed_size=256 --view_mix=nn --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0.0 \
    --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 --noise_item_boost=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
    --weight_decay=0.0 --sem_emb=0 --user_sem_emb=0 \
    $extra > "$log" 2>&1
  rc=$?
  echo "END   ${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

run ctrl         ""
run tailboost05  "--noise_deg_gamma=-0.5"
run lr_cos100    "--lr_anneal=2 --lr_anneal_epochs=100 --lr_floor=0.1"
run wd4          "--weight_decay=1e-4"
run agg_shallow  "--agg_weights=shallow"
run swa30        "--swa_start=30"
run margin01     "--bpr_margin=0.1"
run dropedge09   "--dropedge_keep=0.9"
run init2        "--init_scale=2.0"

echo "R93_LOST_CANDIDATES_ALLDONE $(date)" >> "$L"

# 简易汇总：把每个候选的 best R@20 与 ctrl 对比，自动标记 >= ctrl+0.001 的候选
echo "=== summary (best R@20 / NDCG@20) ===" >> "$L"
for f in research_logs/r93_gowalla_*.log; do
  tag=$(basename "$f" .log | sed 's/r93_gowalla_//')
  vals=$(grep "best_result" -A3 "$f" | tail -3 | awk '{print $2, $3}' | paste -sd' ' -)
  echo "$tag  $vals" >> "$L"
done
echo "(对照：历史 ctrl=0.1877/0.1596@ep60, gowalla seed噪声 ±0.0008)" >> "$L"

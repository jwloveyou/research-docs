#!/usr/bin/env bash
# =============================================================================
# R113 · PRISM 式流行度感知嵌入范数初始化（第 21 个静态组件候选，初始化轴）
#   （KDD'25 arXiv:2505.11318「weight decay 的主函数=把流行度编码进嵌入范数」，
#     PRISM 预编码该流行度；本 repo R23 L2 归一化消融已独立证实「范数=流行度」——
#     抹掉范数即抹掉流行度信号，-11%~-28%。这是初始化轴里唯一未实测的
#     「per-node 流行度感知」子轴（R60 init_scale 是均匀缩放、已判中性）。）
#
# 协议（与 r34/r35/r49/r58/r60/r93 的 4th-component 测试台逐项一致）：
#   gowalla / embed256 / win30 余弦退火(ssl_anneal=2, ssl_anneal_epochs=30,
#   floor=0) / view_mix=nn / noise_eps=0.2 / ssl_reg=0.1 / seed=42 /
#   best-over-epochs / 全量排序 + NDCG@20 / stop_cnt=1000 / 60ep（直抵 gowalla
#   真峰 ep61 附近）。
#
# 扫描（线性中心化公式，max scale ≈ 1+γ 有界，守 R28「勿用度幂律」教训）：
#   1 ctrl          --init_pop_gamma=0.0   （=历史 ctrl 0.1878/0.1597@ep60）
#   2 pop025        --init_pop_gamma=0.25
#   3 pop050        --init_pop_gamma=0.5
#   4 pop100        --init_pop_gamma=1.0
#
# 串行、单卡、不共驻（R82 教训）。只应在 GPU 空闲时手动运行（勿与 baseline
# 调度器 auto_scheduler.sh 争卡），结果写 r113_initpop_progress.txt。
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L=research_logs/r113_initpop_progress.txt
echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  tag=$1; gamma=$2
  log="research_logs/r113_gowalla_${tag}.log"
  echo "START ${tag} gpu=$GPU ep=60 gamma=${gamma} $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=60 --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --embed_size=256 --view_mix=nn --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0.0 \
    --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 --noise_item_boost=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
    --weight_decay=0.0 --sem_emb=0 --user_sem_emb=0 --init_scale=1.0 \
    --init_pop_gamma=${gamma} > "$log" 2>&1
  rc=$?
  echo "END   ${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

run ctrl   0.0
run pop025 0.25
run pop050 0.5
run pop100 1.0

echo "R113_INITPOP_ALLDONE $(date)" >> "$L"

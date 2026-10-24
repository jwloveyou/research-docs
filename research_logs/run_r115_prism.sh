#!/usr/bin/env bash
# =============================================================================
# R115 · 忠实 PRISM 初始化（KDD'25 arXiv:2505.11318，Eq. 4 逐式实现）
#   （R113 的 init_pop_gamma 是「PRISM-style 线性中心化」近似（scale=1+γ(c−mean)，
#     c=deg/max(deg)，非论文公式）；本轮补上**忠实** Eq. 4：
#       row 先单位化  i_init/||i_init||，再乘  alpha*log(deg+2)+(1-alpha)。
#     alpha=0 => 纯单位向量；alpha=1 => 纯流行度 log 幅值；log(d+2) 天然有界
#     （R28「勿用度幂律」的 ~50× 崩坏不适用）。
#
# 协议（与 r34/r35/r49/r58/r60/r93/r113 的 4th-component 测试台逐项一致）：
#   gowalla / embed256 / win30 余弦退火(ssl_anneal=2, ssl_anneal_epochs=30,
#   floor=0) / view_mix=nn / noise_eps=0.2 / ssl_reg=0.1 / seed=42 /
#   best-over-epochs / 全量排序 + NDCG@20 / stop_cnt=1000 / 60ep。
#
# 扫描（忠实 PRISM，alpha 是流行度编码强度插值）：
#   1 ctrl          --init_prism_alpha=-1.0（=标准 xavier，历史 ctrl 0.1878/0.1597@ep60）
#   2 prism050      --init_prism_alpha=0.5
#   3 prism100      --init_prism_alpha=1.0（PRISM 论文 α=1 即「消除 weight decay」）
#   4 prism200      --init_prism_alpha=2.0（外推，检验更强流行度幅值是否有利）
#
# 串行、单卡、不共驻（R82 教训）。只应在 GPU 空闲时手动运行（勿与 baseline
# 调度器 auto_scheduler.sh 争卡），结果写 r115_prism_progress.txt。
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L=research_logs/r115_prism_progress.txt
echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  tag=$1; alpha=$2
  log="research_logs/r115_gowalla_${tag}.log"
  echo "START ${tag} gpu=$GPU ep=60 alpha=${alpha} $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=60 --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --embed_size=256 --view_mix=nn --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0.0 \
    --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 --noise_item_boost=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
    --weight_decay=0.0 --sem_emb=0 --user_sem_emb=0 --init_scale=1.0 \
    --init_pop_gamma=0.0 --init_prism_alpha=${alpha} > "$log" 2>&1
  rc=$?
  echo "END   ${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

run ctrl      -1.0
run prism050  0.5
run prism100  1.0
run prism200  2.0

echo "R115_PRISM_ALLDONE $(date)" >> "$L"

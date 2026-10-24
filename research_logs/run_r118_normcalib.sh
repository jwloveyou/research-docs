#!/usr/bin/env bash
# =============================================================================
# R118 · 后训练（eval-time）物品嵌入范数校准（degree-prior norm calibration）
#   第 23 个静态组件候选 · 「范数=流行度」轴的后训练（post-training）子轴。
#
# 动机（三路文献独立佐证）：
#   ① RecSys'24《Post-Training Embedding Enhancement for Long-Tail Recommendation》
#      (DOI 10.1145/3627673.3679978)：tail 物品的嵌入「范数 less reflective of
#      actual popularity」且「方向 less effective」——范数估计对 tail 物品不可靠；
#   ② KDD'25 PRISM（arXiv:2505.11318）：weight decay 主函数=把流行度编码进嵌入范数；
#   ③ KDD'26 分析向量分解（DOI 10.1145/3770854.3780295）：BPR 把物品嵌入沿「流行度
#      方向」组织、范数 ∝ 流行度。
#   本 repo 自身证据：R23 L2 归一化（余弦评分）三集强负 −11%~−28%（范数携带流行度
#   信号、不能抹掉）；R113/R115 在「初始化」时预编码范数（待 GPU）。
#   => 本候选是【后训练】时把物品范数**重标定**（denoise）到 degree prior，而非
#      R23 的「抹掉范数」、也非 R113/R115 的「初始化时预编码」—— 全新正交子轴。
#
# 实现（eval-time，静态零参数）：
#   target_i = log(deg_i+2) 相对先验（mean 归一化）
#   cur_i    = ||e_i|| 相对当前范数（mean 归一化）
#   scale_i  = 1 + beta * (target_i / cur_i - 1)
#   e_i      = e_i * scale_i          （只在 eval() 前向、训练不动）
#   beta=0 严格 no-op（向后兼容）；beta=1 完全 snap 到 log(d+2) 相对先验。
#
# 协议（与 r34/r35/r49/r58/r60/r93/r113/r115 的 4th-component 测试台逐项一致）：
#   gowalla / embed256 / win30 余弦退火 / view_mix=nn / noise_eps=0.2 / ssl_reg=0.1
#   / seed=42 / best-over-epochs / 全量排序+NDCG@20 / stop_cnt=1000 / 60ep。
#
# 扫描（beta ∈ {0, 0.25, 0.5, 1.0}）：
#   1 ctrl     beta=0.0（历史 ctrl 0.1878/0.1597@ep60）
#   2 calib025 beta=0.25
#   3 calib050 beta=0.5
#   4 calib100 beta=1.0
#
# 串行、单卡、不共驻（R82 教训）。只应在 GPU 空闲时手动运行（勿与 baseline
# 调度器 auto_scheduler.sh 争卡），结果写 r118_normcalib_progress.txt。
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L=research_logs/r118_normcalib_progress.txt
echo "BEGIN $(date) gpu=$GPU" > "$L"

run () {
  tag=$1; beta=$2
  log="research_logs/r118_gowalla_${tag}.log"
  echo "START ${tag} gpu=$GPU ep=60 beta=${beta} $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=60 --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --embed_size=256 --view_mix=nn --use_mlc=0 \
    --noise_eps=0.2 --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0.0 \
    --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 --noise_item_boost=0.0 \
    --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
    --weight_decay=0.0 --sem_emb=0 --user_sem_emb=0 --init_scale=1.0 \
    --init_pop_gamma=0.0 --init_prism_alpha=-1.0 \
    --norm_calib_beta=${beta} > "$log" 2>&1
  rc=$?
  echo "END   ${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

run ctrl     0.0
run calib025 0.25
run calib050 0.5
run calib100 1.0

echo "R118_NORMCALIB_ALLDONE $(date)" >> "$L"

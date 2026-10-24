#!/usr/bin/env bash
# =============================================================================
# R119 · 后训练（eval-time）物品范数校准的 β 扫描（单次训练 + eval 期 β sweep）
#   （R118 的 4× 加速版：norm_calib 只在 eval() 前向改变物品范数、训练完全不受
#     beta 影响 → 单次训练 + 每 epoch 对 beta∈{0,.25,.5,1.0} 各评估一次，与
#     「4 次独立训练」**逐位等价**，但 ~4× 更省 GPU。）
#
# 实现：NoiseSGL 的 `norm_calib_sweep` 开关（本轮新增，默认空=严格向后兼容）。
#   evaluate_model() 里对每个 beta 调 lightgcn.set_norm_calib(beta) 后单独评估、
#   各自 best-over-epochs，train_model 末尾打印 `[norm-calib sweep] beta -> ...`。
#
# 协议（与 r34/r35/r49/r58/r60/r93/r113/r115 的 4th-component 测试台逐项一致）：
#   gowalla / embed256 / win30 余弦退火 / view_mix=nn / noise_eps=0.2 / ssl_reg=0.1
#   / seed=42 / best-over-epochs / 全量排序+NDCG@20 / stop_cnt=1000 / 60ep。
#
# 单 run，结果写 r119_normcalib_progress.txt / r119_gowalla_sweep.log。
# 只应在 GPU 空闲时运行（勿与 baseline 调度器争卡），通常由
# wait_r119_static_candidates.sh 在 baseline 队列空后自动拉起。
# =============================================================================
set -u
PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
cd ~/SGL-Torch || exit 1
GPU=0
L=research_logs/r119_normcalib_progress.txt
log="research_logs/r119_gowalla_sweep.log"
echo "BEGIN $(date) gpu=$GPU" > "$L"

echo "START normcalib_sweep gpu=$GPU ep=60 beta=0.0,0.25,0.5,1.0 $(date)" >> "$L"
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
  --recommender=NoiseSGL --dataset=gowalla --seed=42 --epochs=60 --verbose=1 \
  --stop_cnt=1000 --gpu_id=$GPU --embed_size=256 --view_mix=nn --use_mlc=0 \
  --noise_eps=0.2 --ssl_reg=0.1 --ssl_anneal=2 --ssl_anneal_epochs=30 \
  --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0.0 \
  --noise_deg_gamma=0.0 --noise_layer_gamma=0.0 --noise_item_boost=0.0 \
  --res_alpha=0.0 --spec_bal=0.0 --lr_anneal=0 --ssl_temp_anneal=0 \
  --weight_decay=0.0 --sem_emb=0 --user_sem_emb=0 --init_scale=1.0 \
  --init_pop_gamma=0.0 --init_prism_alpha=-1.0 \
  --norm_calib_beta=0.0 --norm_calib_sweep=0.0,0.25,0.5,1.0 > "$log" 2>&1
rc=$?
echo "END   normcalib_sweep rc=$rc $(date)" >> "$L"
echo "--- normcalib_sweep best (primary beta=0) ---" >> "$L"
grep "best_result" -A3 "$log" | tail -4 >> "$L"
echo "--- normcalib_sweep per-beta ---" >> "$L"
grep "norm-calib sweep" "$log" >> "$L"

echo "R119_NORMCALIB_SWEEP_ALLDONE $(date)" >> "$L"

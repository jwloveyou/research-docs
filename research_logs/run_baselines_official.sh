#!/usr/bin/env bash
# =============================================================================
# run_baselines_official.sh  (task #13/#14 audit fix, R78)
# -----------------------------------------------------------------------------
# 每个 baseline 用【它自己的官方超参】跑，禁止"容量匹配 embed256"扭曲。
# 融合模型仍用 embed256（组件③），baseline 各用官方维度/层数/batch/lr/reg。
#
# 官方超参出处（已逐项核官方仓库 parser/README）：
#   LightGCN  SIGIR'20  gusye1234/LightGCN-PyTorch  recdim=64 layer=3 bpr_batch=2048
#                        lr=1e-3 decay=1e-4 topk=[20]
#   SGL       SIGIR'21  CCIIPLab/SGL                 embed=64 layer=3 batch=2048 lr=1e-3
#                        decay=1e-4 ssl_reg=0.1 ssl_ratio=0.1 ssl_temp=0.2 aug=ED both_side
#   SimGCL    SIGIR'23  CCF-CR/SimGCL                embed=64 layer=3 batch=2048 lr=1e-3
#                        decay=1e-4 cl_rate=0.1 eps=0.1 tau=0.2
#   XSimGCL   SIGIR'23  (SimGCL 变体)                embed=64 layer=3 eps=0.2 cl=0.1 tau=0.2
#   LightGCL  ICLR'23   HKUDS/LightGCL               d=64 gnn_layer=2 batch=256 lr=1e-3
#                        lambda1=0.2 q=5 temp=0.2 lambda2=1e-7
#   HCCF      SIGIR'22  akaxlh/HCCF                  latdim=32 gnn_layer=2 batch=256
#                        sslbatch=4096 lr=1e-3 reg=1e-5 hyperNum=128 keepRate=0.5 leaky=0.5
#                        yelp: temp=1 ssl_reg=1e-4 ; amazon: temp=0.1 ssl_reg=1e-7 reg=1e-2
#   DCCF      SIGIR'23  HKUDS/DCCF                   embed=32 n_layers=2 lr=1e-3
#                        n_intents=128 temp=1 emb_reg=2.5e-5 cen_reg=5e-3 ssl_reg=1e-1
#                        (无全局 weight decay；官方用 train_num 采样，本框架全量 BPR 每 epoch)
#   AdaGCL    KDD'23    HKUDS/AdaGCL                 latdim=32 gnn_layer=2 batch=4096 lr=1e-3
#                        reg=1e-5 ssl_reg=0.1 ib_reg=0.1 temp=0.5 lambda0=1e-4 gamma=-0.45
#                        zeta=1.05 init_temp=2.0 temp_decay=0.98 eps=1e-3
#   NGCF      SIGIR'19  xiangwang1223/NGCF           embed=64 layer=[64,64,64] batch=1024
#                        gowalla lr=1e-4 amazon lr=5e-4 regs=1e-5 node_drop=0.1 mess_drop=0.1
#   UltraGCN  CIKM'21   (论文实验设置)               embed=64 R=300 lr=1e-4 init=N(0,1e-4)
#   DGCF      SIGIR'20  xiangwang1223/DGCF           embed=64 layer=3 K=4 batch=1024 lr=1e-3
#   NCL       WWW'22    RUCAIBox/NCL                 embed=64 layer=3 batch=4096
#                        ssl_reg=1e-6 proto_reg=5e-8/1e-7 temp=0.05 alpha/clusters per-dataset
#                        lr: gowalla=2e-3, else 1e-3
#   DirectAU  KDD'22    (论文/官方)                  embed=64 gamma=1 t=2 lr=1e-3 batch=1024
#
# 评测协议（与融合模型完全一致，task#14 已核）：
#   - 数据切分 = LightGCN/SGL/SimGCL/LightGCL 官方同一份 train/test（.train/.test 与
#     train.txt/test.txt 逐对等价，见 R78 审计）
#   - R@20/NDCG@20 = 全量物品排序（非采样）、训练项置 -inf、fold-out、best-over-epochs
#     （NDCG@20 最佳 epoch），与官方 evaluator 公式一致
#
# 用法: bash run_baselines_official.sh <gpu_id> <baseline>
#   baseline ∈ {lightgcn,sgl,simgcl,xsimgcl,lightgcl,hccf,dccf,adagcl,ngcf,
#               ultragcn,dgcf,ncl,directau,all}
# =============================================================================
set -u
# portable python: masmath=/home/Yehaikun, scnumath=/home/zhangxu
if [ -x /home/Yehaikun/miniconda3/envs/sgl/bin/python ]; then
  PY=/home/Yehaikun/miniconda3/envs/sgl/bin/python
else
  PY=/home/zhangxu/miniconda3/envs/sgl/bin/python
fi
cd "$(dirname "$0")/.."
GPU="${1:-7}"
B="${2:-all}"
L="research_logs/r78_official_baselines_progress.txt"
[ -f "$L" ] || echo "BEGIN $(date)" > "$L"

run () {
  # run <tag> <ds> <eps> <extra...>
  tag=$1; ds=$2; eps=$3; shift 3
  log="research_logs/r78_${ds}_${tag}.log"
  echo "START ${ds}_${tag} gpu=$GPU ep=$eps $(date)" >> "$L"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY main.py \
    --recommender=NoiseSGL --dataset=$ds --seed=42 --epochs=$eps --verbose=1 \
    --stop_cnt=1000 --gpu_id=$GPU --use_mlc=0 --ssl_anneal=0 --ssl_anneal_epochs=0 \
    --ssl_anneal_floor=0.0 --l2_norm=0 --dropout=0 --noise_deg_gamma=0.0 "$@" > "$log" 2>&1
  rc=$?
  echo "END   ${ds}_${tag} rc=$rc $(date)" >> "$L"
  echo "--- ${ds}_${tag} best ---" >> "$L"
  grep "best_result" -A3 "$log" | tail -4 >> "$L"
}

# ---- LightGCN (官方 embed64/3层/batch2048/lr1e-3/decay1e-4) ----
lightgcn () {
  run lightgcn gowalla     300 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0
  run lightgcn yelp2018    100 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0
  run lightgcn amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --ssl_reg=0
}

# ---- SGL (官方 embed64/3层/batch2048/ED/ssl_reg0.1/temp0.2/ratio0.1) ----
sgl () {
  run sgl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=ee --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --noise_eps=0.2
  run sgl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=ee --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --noise_eps=0.2
  run sgl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=ee --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2 --noise_eps=0.2
}

# ---- SimGCL (官方 embed64/3层/batch2048/cl_rate0.1/eps0.1/tau0.2) ----
simgcl () {
  run simgcl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
  run simgcl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
  run simgcl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=0 --noise_eps=0.1 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
}

# ---- XSimGCL (官方 embed64/3层/eps0.2/cl0.1/tau0.2) ----
xsimgcl () {
  run xsimgcl gowalla     150 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
  run xsimgcl yelp2018     50 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
  run xsimgcl amazon-book  30 --embed_size=64 --n_layers=3 --batch_size=2048 --lr=0.001 --reg=1e-4 --view_mix=nn --noise_mode=1 --noise_eps=0.2 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
}

# ---- LightGCL (官方 d=64/gnn_layer2/batch256/lambda1=0.2/q=5/temp0.2/lambda2=1e-7) ----
lightgcl () {
  run lightgcl gowalla     150 --embed_size=64 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-7 --lightgcl=1 --lightgcl_perlayer=1 --gl_rank=5 --ssl_reg=0.2 --ssl_temp=0.2 --ssl_ratio=0.1
  run lightgcl yelp2018     50 --embed_size=64 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-7 --lightgcl=1 --lightgcl_perlayer=1 --gl_rank=5 --ssl_reg=0.2 --ssl_temp=0.2 --ssl_ratio=0.1
  run lightgcl amazon-book  30 --embed_size=64 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-7 --lightgcl=1 --lightgcl_perlayer=1 --gl_rank=5 --ssl_reg=0.2 --ssl_temp=0.2 --ssl_ratio=0.1
}

# ---- HCCF (官方 latdim32/gnn_layer2/batch256/reg1e-5/hyperNum128/keepRate0.5) ----
# yelp/gowalla: temp=1 ssl_reg=1e-4 ; amazon: temp=0.1 ssl_reg=1e-7 reg=1e-2
hccf () {
  run hccf gowalla     100 --embed_size=32 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-5 --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-4 --hccf_temp=1.0
  run hccf yelp2018    100 --embed_size=32 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-5 --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-4 --hccf_temp=1.0
  run hccf amazon-book 100 --embed_size=32 --n_layers=2 --batch_size=256 --lr=0.001 --reg=1e-2 --hccf=1 --hccf_hyper_num=128 --hccf_keep_rate=0.5 --hccf_ssl_reg=1e-7 --hccf_temp=0.1
}

# ---- DCCF (官方 embed32/n_layers2/lr1e-3/intents128/temp1/emb2.5e-5/cen5e-3/ssl1e-1) ----
dccf () {
  run dccf gowalla     150 --embed_size=32 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=0.0 --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1
  run dccf yelp2018    100 --embed_size=32 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=0.0 --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1
  run dccf amazon-book 100 --embed_size=32 --n_layers=2 --batch_size=1024 --lr=0.001 --reg=0.0 --dccf=1 --dccf_n_intents=128 --dccf_temp=1.0 --dccf_emb_reg=2.5e-5 --dccf_cen_reg=5e-3 --dccf_ssl_reg=0.1
}

# ---- AdaGCL (官方 latdim32/gnn_layer2/batch4096/lr1e-3/reg1e-5) ----
adagcl () {
  run adagcl gowalla     200 --embed_size=32 --n_layers=2 --batch_size=4096 --lr=0.001 --reg=1e-5 --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3
  run adagcl yelp2018    100 --embed_size=32 --n_layers=2 --batch_size=4096 --lr=0.001 --reg=1e-5 --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3
  run adagcl amazon-book 100 --embed_size=32 --n_layers=2 --batch_size=4096 --lr=0.001 --reg=1e-5 --adagcl=1 --adagcl_ssl_reg=0.1 --adagcl_ib_reg=0.1 --adagcl_reg=1e-5 --adagcl_temp=0.5 --adagcl_beta=0.1 --adagcl_lambda0=1e-4 --adagcl_gamma=-0.45 --adagcl_zeta=1.05 --adagcl_init_temp=2.0 --adagcl_temp_decay=0.98 --adagcl_eps=1e-3
}

# ---- NGCF (官方 embed64/3层/batch1024/reg1e-5/drop0.1；gowalla·yelp lr=1e-4, amazon 5e-4) ----
ngcf () {
  run ngcf gowalla     400 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.0001 --reg=1e-5 --ngcf=1 --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1
  run ngcf yelp2018    400 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.0001 --reg=1e-5 --ngcf=1 --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1
  run ngcf amazon-book 200 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.0005 --reg=1e-5 --ngcf=1 --ngcf_node_dropout=0.1 --ngcf_message_dropout=0.1
}

# ---- UltraGCN (官方 embed64/R=300/lr1e-4/init N(0,1e-4)) ----
ultragcn () {
  run ultragcn gowalla     150 --embed_size=64 --lr=0.0001 --reg=1e-4 --param_init=normal --stddev=0.01 --ultragcn=1 --ultragcn_lambda=1.0 --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --batch_size=1024
  run ultragcn yelp2018    100 --embed_size=64 --lr=0.0001 --reg=1e-4 --param_init=normal --stddev=0.01 --ultragcn=1 --ultragcn_lambda=1.0 --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --batch_size=1024
  run ultragcn amazon-book 100 --embed_size=64 --lr=0.0001 --reg=1e-4 --param_init=normal --stddev=0.01 --ultragcn=1 --ultragcn_lambda=1.0 --ultragcn_gamma=1.0 --ultragcn_ii_neighbor=10 --num_negatives=300 --batch_size=1024
}

# ---- DGCF (官方 embed64/3层/K=4/batch1024/lr1e-3) ----
dgcf () {
  run dgcf gowalla     200 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.001 --reg=1e-4 --dgcf=1 --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01
  run dgcf yelp2018    100 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.001 --reg=1e-4 --dgcf=1 --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01
  run dgcf amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=1024 --lr=0.001 --reg=1e-4 --dgcf=1 --dgcf_n_factors=4 --dgcf_temp=1.0 --dgcf_ind_reg=0.01
}

# ---- NCL (官方 embed64/3层/batch4096/ssl_reg1e-6/proto_reg5e-8~1e-7/temp0.05) ----
ncl () {
  run ncl gowalla     200 --embed_size=64 --n_layers=3 --batch_size=4096 --lr=0.002 --reg=1e-4 --ncl=1 --ncl_ssl_reg=1e-6 --ncl_proto_reg=5e-8 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=0.5 --ncl_num_clusters=10 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 --ncl_kmeans_seed=0
  run ncl yelp2018    100 --embed_size=64 --n_layers=3 --batch_size=4096 --lr=0.001 --reg=1e-4 --ncl=1 --ncl_ssl_reg=1e-6 --ncl_proto_reg=1e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=1.5 --ncl_num_clusters=2000 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 --ncl_kmeans_seed=0
  run ncl amazon-book 100 --embed_size=64 --n_layers=3 --batch_size=4096 --lr=0.001 --reg=1e-6 --ncl=1 --ncl_ssl_reg=1e-6 --ncl_proto_reg=1e-7 --ncl_temp=0.05 --ncl_hyper_layers=1 --ncl_alpha=0.8 --ncl_num_clusters=2000 --ncl_m_step=1 --ncl_warm_up=20 --ncl_kmeans_iters=5 --ncl_kmeans_seed=0
}

# ---- DirectAU (官方 embed64/gamma1/t2/lr1e-3/batch1024，无 GCN) ----
directau () {
  run directau gowalla     300 --embed_size=64 --lr=0.001 --reg=1e-4 --batch_size=1024 --directau=1 --directau_gamma=1.0 --directau_t=2.0
  run directau yelp2018    100 --embed_size=64 --lr=0.001 --reg=1e-4 --batch_size=1024 --directau=1 --directau_gamma=1.0 --directau_t=2.0
  run directau amazon-book 100 --embed_size=64 --lr=0.001 --reg=1e-4 --batch_size=1024 --directau=1 --directau_gamma=1.0 --directau_t=2.0
}

case "$B" in
  lightgcn|sgl|simgcl|xsimgcl|lightgcl|hccf|dccf|adagcl|ngcf|ultragcn|dgcf|ncl|directau)
    echo "=== $B ===" >> "$L"
    $B
    echo "${B}_ALLDONE $(date)" >> "$L"
    ;;
  all)
    for m in lightgcn sgl simgcl xsimgcl lightgcl hccf dccf adagcl ngcf ultragcn dgcf ncl directau; do
      echo "=== $m ===" >> "$L"
      $m
      echo "${m}_ALLDONE $(date)" >> "$L"
    done
    echo "ALL_BASELINES_OFFICIAL_DONE $(date)" >> "$L"
    ;;
  *)
    echo "unknown baseline $B" >> "$L"; exit 1
    ;;
esac

"""
Paper: Self-supervised Graph Learning for Recommendation (SGL backbone)
        + SimGCL (SIGIR'22) / XSimGCL (TOIS'23) embedding-space noise augmentation.

NoiseSGL: replaces SGL's random edge-drop views with two independent embedding-space
noise views (uniform noise on the unit sphere, per propagation layer). Graph
perturbation -> embedding perturbation, following SimGCL/XSimGCL findings.
"""

__author__ = "Jiancan Wu"
__email__ = "wujcan@gmail.com"

__all__ = ["NoiseSGL"]

import torch
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice


class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None
        # Round-3: SimGCF frequency-signal-scaler layer weights (None -> uniform mean).
        self.freq_alphas = None
        self.freq_flip = False

        # # weight initialization
        # self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items, eps=0.0):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        # NoiseSGL: two independent noisy views over the SAME full graph (ignore sub-graphs).
        user_embeddings1, item_embeddings1 = self._forward_gcn(self.norm_adj, perturb=True, eps=eps)
        user_embeddings2, item_embeddings2 = self._forward_gcn(self.norm_adj, perturb=True, eps=eps)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)    # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)    # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1, 
                                        torch.transpose(user_embeddings2, 0, 1))        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1, 
                                        torch.transpose(item_embeddings2, 0, 1))        # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  # [batch_size, num_users]

        return sup_logits, ssl_logits_user, ssl_logits_item

    def _forward_gcn(self, norm_adj, return_all=False, perturb=False, eps=0.0):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            if perturb and eps > 0:
                # SimGCL-style embedding-space perturbation: uniform noise on the unit
                # sphere, signed by the embedding, scaled by eps. Applied per propagation
                # layer to inject structural-independent stochasticity (no edge dropping).
                random_noise = torch.rand_like(ego_embeddings)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise, dim=1) * eps
            all_embeddings += [ego_embeddings]

        if return_all:
            layer_list = []
            for emb in all_embeddings:
                u, i = torch.split(emb, [self.num_users, self.num_items], dim=0)
                layer_list.append((u, i))
            return layer_list

        if self.freq_alphas is not None:
            # SimGCF frequency signal scaler: weighted (not uniform) layer aggregation,
            # weights = fitted monomial coefficients of g(lambda)*f(lambda), normalized.
            w = self.freq_alphas.to(ego_embeddings.device).view(1, -1, 1)
            all_embeddings = (torch.stack(all_embeddings, dim=1) * w).sum(dim=1)
        else:
            all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        if self.freq_flip:
            # "space flip" of the high-pass SimGCF variant
            user_embeddings = -user_embeddings

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class NoiseSGL(AbstractRecommender):
    def __init__(self, config):
        super(NoiseSGL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd','ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]
        self.ssl_warmup_epochs = config["ssl_warmup_epochs"] if "ssl_warmup_epochs" in config else 0
        self.mlc_start_layer = config["mlc_start_layer"] if "mlc_start_layer" in config else 0
        self.mlc_layer_weight = config["mlc_layer_weight"] if "mlc_layer_weight" in config else "uniform"
        self.ssl_degree_alpha = config["ssl_degree_alpha"] if "ssl_degree_alpha" in config else 0.5
        self.noise_eps = config["noise_eps"] if "noise_eps" in config else 0.1
        self.use_mlc = config["use_mlc"] if "use_mlc" in config else 1
        # Round-2: hardness-calibrated (semi-hard) negative sampling.
        self.neg_mode = config["neg_mode"] if "neg_mode" in config else "uniform"
        assert self.neg_mode in ["uniform", "semihard"]
        self.neg_K = int(config["neg_K"]) if "neg_K" in config else 5
        # Round-3: SimGCF frequency signal scaler (arXiv:2512.15744)
        self.freq_scaler = int(config["freq_scaler"]) if "freq_scaler" in config else 0
        self.freq_type = config["freq_type"] if "freq_type" in config else "low"
        assert self.freq_type in ["low", "high"]
        self.freq_alpha = float(config["freq_alpha"]) if "freq_alpha" in config else -3.0
        self.freq_beta = float(config["freq_beta"]) if "freq_beta" in config else 1.0
        self.freq_mu = float(config["freq_mu"]) if "freq_mu" in config else 1.0
        self.freq_a = float(config["freq_a"]) if "freq_a" in config else 2.0
        self.freq_b = float(config["freq_b"]) if "freq_b" in config else 1.1
        self.freq_norm = int(config["freq_norm"]) if "freq_norm" in config else 1
        self.freq_mix = float(config["freq_mix"]) if "freq_mix" in config else 1.0

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_mode,
            self.ssl_temp,
            self.ssl_reg
        )
        self.model_str += '/warmup=%d-mlc_start=%d-mlc_weight=%s' % (
            self.ssl_warmup_epochs,
            self.mlc_start_layer,
            self.mlc_layer_weight
        )
        self.model_str += '/deg_alpha=%.2f' % self.ssl_degree_alpha
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, 
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        user_degree = np.array(self.dataset.train_csr_mat.sum(1)).reshape(-1).astype(np.float32)
        item_degree = np.array(self.dataset.train_csr_mat.sum(0)).reshape(-1).astype(np.float32)
        self.user_ssl_weights = np.power(user_degree + 1.0, -float(self.ssl_degree_alpha))
        self.item_ssl_weights = np.power(item_degree + 1.0, -float(self.ssl_degree_alpha))
        self.user_ssl_weights = self.user_ssl_weights / self.user_ssl_weights.mean()
        self.item_ssl_weights = self.item_ssl_weights / self.item_ssl_weights.mean()
        self.user_ssl_weights = torch.from_numpy(self.user_ssl_weights).float().to(self.device)
        self.item_ssl_weights = torch.from_numpy(self.item_ssl_weights).float().to(self.device)
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.freq_scaler:
            alphas = self._fit_freq_alphas()
            if self.freq_norm:
                alphas = alphas / np.abs(alphas).mean()
            w = alphas / (self.n_layers + 1)
            if self.freq_mix < 1.0:
                # blend the scaler waveform with the uniform (all-pass) LightGCN filter
                uniform = np.ones_like(w) / (self.n_layers + 1)
                w = (1.0 - self.freq_mix) * uniform + self.freq_mix * w
            if self.freq_type == "high":
                w = np.array([((-1) ** (i + 1)) * a for i, a in enumerate(w)])
                self.lightgcn.freq_flip = True
            # fold the 1/(L+1) mean into the weights so all-ones == uniform LightGCN
            self.lightgcn.freq_alphas = torch.tensor(w, dtype=torch.float32)
            self.logger.info("[freq-scaler] type=%s alphas=%s" % (
                self.freq_type, np.round(self.lightgcn.freq_alphas.cpu().numpy(), 4).tolist()))
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)), 
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    @staticmethod
    def _jacobi_conv(n, x, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
        """Jacobi polynomial basis filter (ported from SimGCF PolyConv.py)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x)
        x1 = np.ones_like(x)
        y += alphas[0] * x1
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef2 = (a + b + 2) / (r - l)
        x2 = coef1 + coef2 * x
        y += alphas[1] * x2
        X = [x1, x2]
        for i in range(2, n):
            coef_l = 2 * i * (i + a + b) * (2 * i - 2 + a + b)
            c1 = (2 * i + a + b - 1) * (2 * i + a + b) * (2 * i + a + b - 2)
            c2 = (2 * i + a + b - 1) * (a ** 2 - b ** 2)
            c3 = 2 * (i - 1 + a) * (i - 1 + b) * (2 * i + a + b)
            t1, t2, t3 = c1 / coef_l, c2 / coef_l, c3 / coef_l
            t1_2 = t1 * (2 / (r - l))
            t2_2 = t1 * ((r + l) / (r - l)) + t2
            nx = t1_2 * (x * X[-1]) + t2_2 * X[-1] - t3 * X[-2]
            X.append(nx)
            y += alphas[i] * nx
        return y / n

    def _fit_freq_alphas(self):
        """Fit monomial coefficients a_i s.t. sum_i a_i*lambda^i ~= g(lambda)*Jacobi(lambda),
        where g is the SimGCF sigmoid frequency signal scaler and the Jacobi filter is the
        backbone (all-ones coefficients). Solved exactly by least squares over lambda in [-1,1]
        (equivalent to the Adam fit in SimGCF's fit_filter.py, but closed form).
        """
        n = self.n_layers + 1
        x = np.arange(-1.0, 1.0, 0.0001)
        jac = self._jacobi_conv(n, x, [1.0] * n, self.freq_a, self.freq_b)
        g = self.freq_mu / (1.0 + np.exp(self.freq_alpha * (x + self.freq_beta)))
        target = g * jac
        A = np.vstack([x ** i for i in range(n)]).T
        alphas, *_ = np.linalg.lstsq(A, target, rcond=None)
        return alphas

    @staticmethod
    def _pick_semihard(sc, pos_sc, valid):
        """Semi-hard selection: hardest candidate still below the positive score;
        fall back to the easiest valid candidate (avoids pushing false negatives)."""
        neg_inf = torch.tensor(float("-inf"), device=sc.device)
        pos_inf = torch.tensor(float("inf"), device=sc.device)
        below = valid & (sc < pos_sc[:, None])
        sc_below = torch.where(below, sc, neg_inf)
        sc_valid = torch.where(valid, sc, pos_inf)
        has = below.any(dim=1)
        idx_best_below = sc_below.argmax(dim=1)
        idx_min_valid = sc_valid.argmin(dim=1)
        pick = torch.where(has, idx_best_below, idx_min_valid)
        return pick

    def _semihard_sampler(self):
        """One epoch of pairwise batches with hardness-calibrated negatives.

        Candidate pools are mined once per epoch from the current L-layer
        embeddings (two-stage paradigm), then the semi-hard candidate is chosen.
        Negative-key set is used to drop candidates that are true positives.
        """
        pairs = self.dataset.train_data.to_user_item_pairs()
        N = len(pairs)
        if not hasattr(self, "_pos_keys_sorted"):
            uk = pairs[:, 0].astype(np.int64)
            ik = pairs[:, 1].astype(np.int64)
            self._pos_keys_sorted = np.sort(uk * np.int64(self.num_items) + ik)
        pos_keys = self._pos_keys_sorted
        hard_neg = np.empty(N, dtype=np.int64)
        with torch.no_grad():
            ue, ie = self.lightgcn._forward_gcn(self.lightgcn.norm_adj)
            chunk = 65536
            for s0 in range(0, N, chunk):
                e0 = min(s0 + chunk, N)
                bu_np = pairs[s0:e0, 0].astype(np.int64)
                bp_np = pairs[s0:e0, 1].astype(np.int64)
                n = e0 - s0
                bu = torch.from_numpy(bu_np).long().to(self.device)
                bp = torch.from_numpy(bp_np).long().to(self.device)
                cand = torch.randint(0, self.num_items, (n, self.neg_K), device=self.device)
                ue_b = ue[bu]                              # [n, d]
                ie_c = ie[cand]                            # [n, K, d]
                sc = (ie_c * ue_b[:, None, :]).sum(-1)     # [n, K]
                pos_sc = (ie[bp] * ue_b).sum(-1)           # [n]
                # true-positive mask (false-negative avoidance)
                ck = (bu_np[:, None] * np.int64(self.num_items) + cand.cpu().numpy()).ravel()
                is_pos = np.isin(ck, pos_keys).reshape(n, self.neg_K)
                valid = torch.from_numpy(~is_pos).to(self.device)
                pick = self._pick_semihard(sc, pos_sc, valid)
                hard_neg[s0:e0] = cand.gather(1, pick[:, None]).squeeze(1).cpu().numpy()
        order = np.random.permutation(N)
        for s0 in range(0, N, self.batch_size):
            idx = order[s0:s0 + self.batch_size]
            yield (pairs[idx, 0].astype(np.int64),
                   pairs[idx, 1].astype(np.int64),
                   hard_neg[idx])

    def _pairwise_iter(self):
        if self.neg_mode == "semihard":
            return self._semihard_sampler()
        return PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                 batch_size=self.batch_size, shuffle=True)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            # NoiseSGL: views come from embedding-space perturbation, no sub-graphs needed.
            sub_graph1, sub_graph2 = None, None
            data_iter = self._pairwise_iter()
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items, eps=self.noise_eps)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)
                # DMLC: multi-layer contrast (disable with use_mlc=0 for pure SimGCL-style).
                if self.use_mlc:
                    layers2_v1 = self.lightgcn._forward_gcn(self.lightgcn.norm_adj, return_all=True,
                                                            perturb=True, eps=self.noise_eps)
                    layers2_v2 = self.lightgcn._forward_gcn(self.lightgcn.norm_adj, return_all=True,
                                                            perturb=True, eps=self.noise_eps)

                    mlc_loss = 0.0
                    n_layers = self.lightgcn.n_layers + 1
                    start_layer = min(max(int(self.mlc_start_layer), 0), n_layers - 1)
                    active_layers = list(range(start_layer, n_layers))
                    if self.mlc_layer_weight == "linear":
                        layer_weights = np.arange(1, len(active_layers) + 1, dtype=np.float32)
                    elif self.mlc_layer_weight == "deep":
                        layer_weights = np.array(active_layers, dtype=np.float32) + 1.0
                    else:
                        layer_weights = np.ones(len(active_layers), dtype=np.float32)
                    layer_weights = layer_weights / layer_weights.sum()

                    for weight, li in zip(layer_weights, active_layers):
                        u1, i1 = layers2_v1[li]
                        u2, i2 = layers2_v2[li]
                        u1n, i1n = F.normalize(u1, dim=1), F.normalize(i1, dim=1)
                        u2n, i2n = F.normalize(u2, dim=1), F.normalize(i2, dim=1)
                        bu1 = F.embedding(bat_users, u1n)
                        bi1 = F.embedding(bat_pos_items, i1n)
                        bu2 = F.embedding(bat_users, u2n)
                        bi2 = F.embedding(bat_pos_items, i2n)
                        pu = inner_product(bu1, bu2)
                        pi = inner_product(bi1, bi2)
                        tu = torch.matmul(bu1, u2n.T)
                        ti = torch.matmul(bi1, i2n.T)
                        su = tu - pu[:, None]
                        si = ti - pi[:, None]
                        cu = torch.logsumexp(su / self.ssl_temp, dim=1)
                        ci = torch.logsumexp(si / self.ssl_temp, dim=1)
                        u_weight = torch.index_select(self.user_ssl_weights, 0, bat_users)
                        i_weight = torch.index_select(self.item_ssl_weights, 0, bat_pos_items)
                        mlc_loss += float(weight) * torch.sum((cu * u_weight) + (ci * i_weight))
                    infonce_loss = infonce_loss + mlc_loss

                if self.ssl_warmup_epochs > 0:
                    ssl_weight = min(1.0, float(epoch) / float(self.ssl_warmup_epochs))
                else:
                    ssl_weight = 1.0
                loss = bpr_loss + self.ssl_reg * ssl_weight * infonce_loss + self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch, 
                total_loss/self.num_ratings,
                total_bpr_loss / self.num_ratings,
                (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time()-training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.cpu().detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.cpu().detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()

"""
DGW-LightGCL: DGW + LightGCL correct architecture
"""
__author__ = "Yehaikun"
__all__ = ["DGW_LightGCL"]

import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss, get_initializer, sp_mat_to_sp_tensor
from data import PairwiseSamplerV2
import numpy as np
from time import time
import scipy.sparse as sp
from reckit import randint_choice


class DGW_LightGCL(AbstractRecommender):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.lr = config['lr']
        self.param_init = config["param_init"]
        self.n_layers = config['n_layers']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_temp = config["ssl_temp"]
        self.svd_q = int(config["svd_q"]) if "svd_q" in config else 5
        self.ssl_degree_alpha = config["ssl_degree_alpha"] if "ssl_degree_alpha" in config else 0.5
        self.ssl_warmup_epochs = config["ssl_warmup_epochs"] if "ssl_warmup_epochs" in config else 3
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.num_ratings = self.dataset.num_train_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # --- Build adjacency on GPU ---
        adj = self._create_adj()
        self.norm_adj = sp_mat_to_sp_tensor(adj).to(self.device)

        # --- Build interaction matrix R on GPU, SVD on R (M x N) ---
        users_items = self.dataset.train_data.to_user_item_pairs()
        R_dense = torch.zeros(self.num_users, self.num_items, device=self.device)
        R_dense[torch.from_numpy(users_items[:,0]).long(), torch.from_numpy(users_items[:,1]).long()] = 1.0
        svd_u, svd_s, svd_v = torch.svd_lowrank(R_dense, q=self.svd_q)
        self.u_mul_s = svd_u * svd_s  # (M, q)
        self.vt = svd_v.T            # (q, N)
        self.ut = svd_u.T            # (q, M)
        self.v_mul_s = svd_v * svd_s  # (N, q)
        del R_dense; torch.cuda.empty_cache()

        # --- DGW degree weights ---
        user_deg = np.array(self.dataset.train_csr_mat.sum(1)).reshape(-1).astype(np.float32)
        item_deg = np.array(self.dataset.train_csr_mat.sum(0)).reshape(-1).astype(np.float32)
        self.user_w = np.power(user_deg + 1.0, -self.ssl_degree_alpha)
        self.item_w = np.power(item_deg + 1.0, -self.ssl_degree_alpha)
        self.user_w = torch.from_numpy(self.user_w / self.user_w.mean()).float().to(self.device)
        self.item_w = torch.from_numpy(self.item_w / self.item_w.mean()).float().to(self.device)

        # --- Embeddings (single table, on GPU) ---
        self.E_u_0 = nn.Parameter(torch.empty(self.num_users, self.emb_size, device=self.device))
        self.E_i_0 = nn.Parameter(torch.empty(self.num_items, self.emb_size, device=self.device))
        init_fn = get_initializer(self.param_init)
        init_fn(self.E_u_0); init_fn(self.E_i_0)
        self.optimizer = torch.optim.Adam([self.E_u_0, self.E_i_0], lr=self.lr)

        # Cache for eval
        self.E_u_final = None; self.E_i_final = None

    def _create_adj(self):
        n = self.num_users + self.num_items
        ui = self.dataset.train_data.to_user_item_pairs()
        un, inn = ui[:, 0], ui[:, 1]
        r = np.ones_like(un, dtype=np.float32)
        a = sp.csr_matrix((r, (un, inn + self.num_users)), shape=(n, n))
        a = a + a.T
        rs = np.array(a.sum(1)).flatten()
        d = np.power(rs, -0.5).flatten()
        d[np.isinf(d)] = 0.
        return sp.diags(d).dot(a).dot(sp.diags(d))

    def _propagate(self):
        """E chain (standard GCN) + G chain (SVD). Returns per-layer lists."""
        Eu = [self.E_u_0]; Ei = [self.E_i_0]
        Gu = [self.E_u_0]; Gi = [self.E_i_0]
        for _ in range(self.n_layers):
            # E: standard GCN (concat -> sparse mm -> split)
            ego = torch.cat([Eu[-1], Ei[-1]], dim=0)
            ego = torch_sp.mm(self.norm_adj, ego)
            u, i = torch.split(ego, [self.num_users, self.num_items], dim=0)
            Eu.append(u); Ei.append(i)
            # G: SVD low-rank propagation (correct shape: M,q; N,q; q,N; q,M)
            Gu.append(self.u_mul_s @ (self.vt @ Ei[-2]))       # (M,q)@((q,N)@(N,d))=(M,d)
            Gi.append(self.v_mul_s @ (self.ut @ Eu[-2]))       # (N,q)@((q,M)@(M,d))=(N,d)
        return Eu, Ei, Gu, Gi

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss = total_bpr = total_reg = 0.0
            t0 = time()
            for bu, bp, bn in data_iter:
                bu = torch.from_numpy(bu).long().to(self.device); bp = torch.from_numpy(bp).long().to(self.device); bn = torch.from_numpy(bn).long().to(self.device)
                # Propagate per-batch for gradient tracking
                Eu, Ei, Gu, Gi = self._propagate()
                E_u = sum(Eu); E_i = sum(Ei)
                # BPR
                bpr = -torch.sum(F.logsigmoid(inner_product(E_u[bu], E_i[bp]) - inner_product(E_u[bu], E_i[bn])))
                reg = l2_loss(E_u[bu], E_i[bp], E_i[bn])
                # MLC+DGW: per-layer InfoNCE(E, G)
                ssl = 0.0
                nl = self.n_layers + 1
                for li in range(nl):
                    ul = F.normalize(Eu[li], dim=1); il = F.normalize(Ei[li], dim=1)
                    gl = F.normalize(Gu[li], dim=1); hl = F.normalize(Gi[li], dim=1)
                    # User side
                    pu = inner_product(ul[bu], gl[bu])
                    au = ul[bu] @ gl.T
                    cu = torch.logsumexp((au - pu.unsqueeze(1)) / self.ssl_temp, dim=1)
                    # Item side
                    pi = inner_product(il[bp], hl[bp])
                    ai = il[bp] @ hl.T
                    ci = torch.logsumexp((ai - pi.unsqueeze(1)) / self.ssl_temp, dim=1)
                    # DGW weights
                    ssl += torch.sum(self.user_w[bu] * cu) + torch.sum(self.item_w[bp] * ci)
                ssl = ssl / nl
                sw = min(1.0, float(epoch) / float(self.ssl_warmup_epochs)) if self.ssl_warmup_epochs > 0 else 1.0
                loss = bpr + self.ssl_reg * sw * ssl + self.reg * reg
                total_loss += loss; total_bpr += bpr; total_reg += self.reg * reg
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch, total_loss / self.num_ratings, total_bpr / self.num_ratings,
                (total_loss - total_bpr - total_reg) / self.num_ratings, total_reg / self.num_ratings, time() - t0))
            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                res, flag = self._evaluate()
                self.logger.info("epoch %d:\t%s" % (epoch, res))
                if flag: self.best_epoch = epoch; stopping_step = 0; self.logger.info("Find a better model.")
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping at epoch: %d" % epoch); break
        self.logger.info("best_result@epoch %d:\n\t\t%s" % (
            self.best_epoch, '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])))

    def _evaluate(self):
        cur, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < cur[1]: self.best_result = cur; return buf, True
        return buf, False

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return torch.matmul(self.E_u_final[users], self.E_i_final.T).cpu().detach().numpy()

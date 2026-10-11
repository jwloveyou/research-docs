"""
NTCF-SGL: Self-supervised Graph Learning with a curvature-aware,
node-specific propagation depth backbone.

Idea borrowed from:
  "Neural Tree Collaborative Filtering: Rethinking Graph Collaborative
   Filtering as Tree Collaborative Filtering with Curvature-Aware
   Propagation Depth" (CIKM'26, arXiv:2608.10297)

Reference baseline: SGL (SIGIR'21, https://github.com/wujcan/SGL-Torch)

The only change w.r.t. SGL is the propagation / aggregation:
  * ntcf_enable = 0 -> plain LightGCN mean aggregation over L layers (baseline)
  * ntcf_enable = 1 -> NTCF: each node v gets a node-specific effective depth
        dbar_N(v) = mean degree of v's neighbours
        kappa(v)  = (dbar_N(v) - deg(v)) / (dbar_N(v) + deg(v) + eps)
        Delta(v)  = +1 if kappa(v) < -tau   (local hub -> deeper)
                    -1 if kappa(v) > +tau   (peripheral -> shallower)
                     0 otherwise
        L_v       = max(L + Delta(v), 1)
    During propagation a node whose depth budget is exhausted is reset to its
    layer-0 embedding (mask m_v^(l) = 1[L_v > l]); masked nodes still feed that
    value to their still-active neighbours. Lmax = max_v L_v layers are run and
    the final representation is the mean over all Lmax+1 layer states.
"""

__author__ = "Yehaikun"
__all__ = ["NTCF_SGL"]

import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from data import PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice


class _LightGCN_NTCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers,
                 ntcf_enable=0, node_depth=None):
        super(_LightGCN_NTCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.ntcf_enable = bool(ntcf_enable)
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        if self.ntcf_enable:
            assert node_depth is not None
            # Lmax = max_v L_v  (adaptive: equals L when no node is adjusted)
            self.node_depth = node_depth.long()
            self.lmax = int(self.node_depth.max().item())
        else:
            self.node_depth = None
            self.lmax = n_layers

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

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

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
                                        torch.transpose(user_embeddings2, 0, 1))
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]

        return sup_logits, ssl_logits_user, ssl_logits_item

    def _forward_gcn(self, norm_adj, return_all=False):
        ego_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        if self.ntcf_enable:
            # curvature-driven node mask, cached at init
            # keep_v^(l) = 1[L_v > l]
            depth = self.node_depth.to(ego_embeddings.device)
            cur = ego_embeddings
            for l in range(1, self.lmax + 1):
                agg = torch_sp.mm(norm_adj, cur)
                keep = (depth > (l - 1)).float().unsqueeze(1)   # [N,1]
                cur = keep * agg + (1.0 - keep) * ego_embeddings
                all_embeddings.append(cur)
        else:
            cur = ego_embeddings
            for k in range(self.n_layers):
                cur = torch_sp.mm(norm_adj, cur)
                all_embeddings.append(cur)

        if return_all:
            layer_list = []
            for emb in all_embeddings:
                u, i = torch.split(emb, [self.num_users, self.num_items], dim=0)
                layer_list.append((u, i))
            return layer_list

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.num_users, self.num_items], dim=0)
        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN_NTCF, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = \
            self._forward_gcn(self.norm_adj)


class NTCF_SGL(AbstractRecommender):
    def __init__(self, config):
        super(NTCF_SGL, self).__init__(config)

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

        # Hyper-parameters for SSL (canonical SGL)
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Hyper-parameters for NTCF
        self.ntcf_enable = int(config["ntcf_enable"]) if "ntcf_enable" in config else 0
        self.ntcf_tau = float(config["ntcf_tau"]) if "ntcf_tau" in config else 0.1
        # ntcf_center: subtract the mean of kappa (per-side) before thresholding.
        # Needed on bipartite CF graphs, where the asymmetric user/item degree
        # distributions make raw kappa systematically positive.
        self.ntcf_center = int(config["ntcf_center"]) if "ntcf_center" in config else 0

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (self.n_layers, self.reg)
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio, self.ssl_mode, self.ssl_temp, self.ssl_reg)
        self.model_str += '/ntcf=%d-tau=%.2f-center=%d' % (
            self.ntcf_enable, self.ntcf_tau, self.ntcf_center)
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, self.model_name, self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, self.model_name, self.n_layers)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.num_ratings = self.dataset.num_train_ratings
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # ---------- NTCF: closed-form local degree-imbalance curvature ----------
        node_depth = None
        if self.ntcf_enable:
            user_deg = np.asarray(self.dataset.train_csr_mat.sum(1)).reshape(-1).astype(np.float64) + 1e-8
            item_deg = np.asarray(self.dataset.train_csr_mat.sum(0)).reshape(-1).astype(np.float64) + 1e-8
            R = self.dataset.train_csr_mat.tocsr()
            # mean degree of the neighbours of each user / item, O(|E|)
            user_nbr_deg = np.asarray(R.dot(item_deg)).reshape(-1) / np.maximum(
                np.asarray(R.sum(1)).reshape(-1), 1.0)
            item_nbr_deg = np.asarray(R.T.dot(user_deg)).reshape(-1) / np.maximum(
                np.asarray(R.sum(0)).reshape(-1), 1.0)
            eps = 1e-8
            kappa_user = (user_nbr_deg - user_deg) / (user_nbr_deg + user_deg + eps)
            kappa_item = (item_nbr_deg - item_deg) / (item_nbr_deg + item_deg + eps)
            if self.ntcf_center:
                kappa_user = kappa_user - kappa_user.mean()
                kappa_item = kappa_item - kappa_item.mean()
            kappa = np.concatenate([kappa_user, kappa_item])
            delta = np.zeros_like(kappa)
            delta[kappa < -self.ntcf_tau] = 1.0
            delta[kappa > self.ntcf_tau] = -1.0
            depth = np.maximum(self.n_layers + delta, 1.0)
            node_depth = torch.from_numpy(depth).float().to(self.device)
            n_shallow = int((depth < self.n_layers).sum())
            n_deep = int((depth > self.n_layers).sum())
            n_flat = int((depth == self.n_layers).sum())
            self.logger.info(
                "[NTCF] tau=%.2f  L=%d  Lmax=%d  shallow(<=L-1)=%d  flat(L)=%d  deep(L+1)=%d" % (
                    self.ntcf_tau, self.n_layers, int(depth.max()),
                    n_shallow, n_flat, n_deep))

        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN_NTCF(
            self.num_users, self.num_items, self.emb_size, adj_matrix,
            self.n_layers, ntcf_enable=self.ntcf_enable,
            node_depth=node_depth).to(self.device)
        self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=int(self.num_users * self.ssl_ratio), replace=False)
                drop_item_idx = randint_choice(self.num_items, size=int(self.num_items * self.ssl_ratio), replace=False)
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
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                      batch_size=self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)

                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss (canonical SGL)
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)

                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch,
                total_loss / self.num_ratings,
                total_bpr_loss / self.num_ratings,
                (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time() - training_start_time))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)
        return self.best_result

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

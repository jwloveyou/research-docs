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
import math
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
        # Round-25: main-branch embedding (input/node) dropout as an overfitting
        # regulariser for the BPR fine-tune phase. Standard GNN regularisation
        # (Srivastava et al. 2014; NGCF message-dropout; SGL node-dropout aug_type=nd)
        # applied ONLY to the main (prediction) branch input embeddings, so the SSL
        # noise views stay identical (respects R4/R6/R7: don't touch the propagation
        # operator and keep main/view filters identical). Static, parameter-free change.
        # Default rate 0 = off (backward compatible); nn.Dropout no-ops in eval() mode
        # so the eval-time embeddings stay deterministic.
        self.dropout_rate = 0.0
        self.dropout = nn.Dropout(self.dropout_rate)
        self._user_embeddings_final = None
        self._item_embeddings_final = None
        # Round-23: L2-normalised (cosine) scoring. When on, the final main-branch
        # user/item embeddings are unit-normalised before the BPR ranking score and
        # before eval-time prediction, so the inner product becomes cosine similarity.
        # Static, parameter-free change (respects the R1/R15/R18 'static wins' law);
        # removes embedding-magnitude (popularity/degree) bias from the ranking signal,
        # aligning with DirectAU's uniformity goal but applied at the scoring level
        # rather than as an extra training loss (avoids the R9 slow-convergence pitfall).
        # Default 0 = off (backward compatible).
        self.l2_norm = False
        # Round-3: SimGCF frequency-signal-scaler layer weights (None -> uniform mean).
        self.freq_alphas = None
        self.freq_flip = False
        # Round-4: where the scaler is applied. main = prediction branch (also used at
        # eval); views = the two contrastive SSL branches. Enables isolating WHICH branch
        # caused Round-3's slow convergence (arXiv:2512.15744 drop-in on NoiseSGL).
        self.freq_on_main = True
        self.freq_on_views = True
        # Round-6: view composition (see NoiseSGL.__init__). Set on the backbone module.
        self.view_mix = "nn"
        self.ssl_temp = 0.2
        # Round-16: noise injection mode. 0 = SimGCL (perturb after every propagation
        # layer, default/backward-compatible); 1 = XSimGCL (perturb the layer-0 embedding
        # once, then propagate cleanly). Both keep the SAME LightGCN operator (R4/R6).
        self.noise_mode = 0
        # Round-28: density-aware (degree-scaled) noise magnitude. None = uniform
        # noise_eps (backward compatible); otherwise a per-node multiplier of the
        # per-layer noise magnitude (mean-normalised to 1).
        self.noise_deg_scale = None
        # Round-33: depth-aware (layer-wise) noise schedule. noise_layer_gamma=0 keeps
        # a uniform per-layer noise magnitude (backward compatible). gamma>0 gives
        # deeper layers a stronger perturbation (over-smoothing grows with depth, cf.
        # the E11 mechanism: later layers collapse toward the principal eigenvector and
        # need a larger re-diversifying noise); gamma<0 gives the reverse. Static,
        # parameter-free, and orthogonal to the node axis (noise_deg_gamma) and the
        # epoch axis (noise_anneal). scale_k = eps * (1 + gamma * k/(L-1)).
        self.noise_layer_gamma = 0.0
        # Round-19: adaptive per-node per-layer gating (AdaGate-style, GRU update-gate
        # analogue). gate = sigmoid(W_k . h_k + b_k) computed from the pre-propagation
        # embedding; h_{k+1} = gate*(A h_k) + (1-gate)*h_k. Adds per-node adaptive
        # propagation (expressivity / anti-oversmoothing) WITHOUT any new view or global
        # signal (respects R4/R6/R7/R11). Default 0 = off (backward compatible).
        self.gate_mode = 0
        self.gate_proj = None
        # Round-7: LightGCL-style *global* collaborative view, built from a truncated
        # SVD of the normalised adjacency (A ~= U_q diag(S_q) V_q^T). Stored in low-rank
        # factors because densifying an n x n matrix would need tens of GBs.
        self.gl_U = None    # [n, q]
        self.gl_S = None    # [q]
        self.gl_Vt = None   # [q, n]
        # Round-17: original LightGCL (arXiv:2302.08191) local-vs-global contrast.
        # The global view is the *bipartite* (I x J) truncated SVD operator applied to the
        # local (main-branch) embeddings: g^(u) = U_q S_q V_q^T . e^(v), g^(v) = V_q S_q
        # U_q^T . e^(u). This is a deterministic denoising signal (no stochastic noise),
        # directly contrasted against the clean local view in InfoNCE. Default off.
        self.lightgcl = False
        self.lightgcl_perlayer = 1   # 1 = per-layer InfoNCE (paper eq.6); 0 = final emb only
        self.gl_UqS = None    # [I, q] = U_q * S_q  (user side, bipartite SVD)
        self.gl_VqS = None    # [J, q] = V_q * S_q  (item side, bipartite SVD)
        # Round-8: RecDCL-style feature-wise contrastive learning (FCL). Representation
        # layer only -> does NOT touch the propagation operator (respects R4/R6). All
        # coefficients default to 0 (module off, backward compatible).
        self.fcl_poly = 0.0      # polynomial-kernel feature-uniformity term
        self.fcl_bt = 0.0        # BarlowTwins-style user-item cross-correlation term
        self.fcl_off = 0.0       # direct off-diagonal decorrelation of user & item Grams
        self.fcl_mom = 0.0       # momentum cross-modal alignment term
        self.fcl_a = 1.0
        self.fcl_degree = 4
        self.fcl_polyc = 1e-7
        self.fcl_bt_off = 0.01
        self.fcl_momentum = 0.1
        # Round-9: DirectAU-style alignment & uniformity (arXiv:2206.12811, KDD'22).
        # Representation-layer only -> does NOT touch the propagation operator (respects
        # R4/R6). Default 0 = off (backward compatible). directau_align weights the
        # positive-pair alignment (||u-i||^2 on the unit sphere); directau_unif weights the
        # global uniformity (log E exp(-t*d)) over sampled nodes.
        self.directau_align = 0.0
        self.directau_unif = 0.0
        self.directau_t = 2.0
        self.directau_k = 512
        # Round-11: NCL-style (WWW'22, arXiv:2202.06200) prototype (cluster) interest
        # injection. k-means clusters the *main-branch* node embeddings into K
        # prototypes; each node's final embedding is convex-blended toward its nearest
        # prototype centre by proto_lam. This injects a coarse global "interest group"
        # signal into the representation WITHOUT touching the propagation operator
        # (respects R4/R6) and WITHOUT any contrastive loss on prototypes (the most
        # conservative variant, per R10, to avoid the R7 global-view pitfall). Default
        # proto_K=0 -> off, backward compatible.
        self.proto_K = 0
        self.proto_lam = 0.0
        self.proto_centers = None   # [K, d]
        self.proto_assign_u = None  # [num_users] long
        self.proto_assign_i = None  # [num_items] long
        d = self.embed_dim
        # projector (MLP d->d->d, no bias, BN+ReLU) as in RecDCL; only used by FCL.
        self.fcl_projector = nn.Sequential(
            nn.Linear(d, d, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
            nn.Linear(d, d, bias=False))
        self.fcl_predictor = nn.Linear(d, d)
        self.fcl_bn = nn.BatchNorm1d(d, affine=False)
        self.u_target_his = None
        self.i_target_his = None

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
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj, use_freq=self.freq_on_main, apply_proto=True, apply_dropout=True)

        if self.l2_norm:
            # Round-23: unit-normalise the fused main-branch embeddings used for the BPR
            # ranking score (the contrastive views are already normalised separately).
            user_embeddings = F.normalize(user_embeddings, dim=1)
            item_embeddings = F.normalize(item_embeddings, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        if neg_item_embs.dim() == 3:
            # Round-21: K>1 negatives per positive -> [batch_size, K] margin scores
            sup_neg_ratings = (user_embs.unsqueeze(1) * neg_item_embs).sum(-1)  # [batch_size, K]
            sup_logits = sup_pos_ratings.unsqueeze(1) - sup_neg_ratings          # [batch_size, K]
        else:
            sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
            sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        if self.lightgcl:
            # Round-17: original LightGCL local-vs-global contrastive loss replaces the
            # noise-view InfoNCE. No stochastic views are built (the clean local view is
            # the main branch itself, the global view is the deterministic SVD operator).
            infonce = self._lightgcl_infonce(users, items)
        else:
            # Round-6/7: build a list of contrastive views. Each view uses the SAME LightGCN
            # operator family; only the injected stochasticity / operator differs:
            #   n = embedding-space noise (SimGCL/XSimGCL), e = edge-dropout structure (SGL),
            #   g = truncated-SVD global collaborative view (LightGCL, Round-7).
            # String order is respected; "nn"/"ne"/"nee" reproduce the Round-6 numerics.
            views = []  # each entry: (normalised user-emb, normalised item-emb)
            n_ed = 0
            for ch in self.view_mix:
                if ch == "n":
                    ue, ie = self._forward_gcn(self.norm_adj, perturb=True, eps=eps,
                                               use_freq=self.freq_on_views)
                elif ch == "e":
                    adj = sub_graph1 if n_ed == 0 else sub_graph2
                    n_ed += 1
                    ue, ie = self._forward_gcn(adj, perturb=False, use_freq=self.freq_on_views)
                elif ch == "g":
                    ue, ie = self._forward_gcn(None, use_gl=True, use_freq=self.freq_on_views)
                else:
                    raise ValueError("unknown view char %r" % ch)
                views.append((F.normalize(ue, dim=1), F.normalize(ie, dim=1)))

            # Pairwise InfoNCE over all view pairs, averaged (for view_mix="nn" this is
            # numerically identical to the original NoiseSGL two-view InfoNCE).
            infonce = None
            npairs = 0
            for a in range(len(views)):
                ue_a, ie_a = views[a]
                for b in range(a + 1, len(views)):
                    ue_b, ie_b = views[b]
                    bu = F.embedding(users, ue_a)
                    bi = F.embedding(items, ie_a)
                    pos_u = inner_product(bu, F.embedding(users, ue_b))
                    pos_i = inner_product(bi, F.embedding(items, ie_b))
                    tot_u = torch.matmul(bu, torch.transpose(ue_b, 0, 1))
                    tot_i = torch.matmul(bi, torch.transpose(ie_b, 0, 1))
                    cu = torch.logsumexp((tot_u - pos_u[:, None]) / self.ssl_temp, dim=1)
                    ci = torch.logsumexp((tot_i - pos_i[:, None]) / self.ssl_temp, dim=1)
                    pair_loss = torch.sum(cu + ci)
                    infonce = pair_loss if infonce is None else infonce + pair_loss
                    npairs += 1
            infonce = infonce / max(npairs, 1)

        fcl_loss = None
        if (self.fcl_poly > 0) or (self.fcl_bt > 0) or (self.fcl_off > 0) or (self.fcl_mom > 0):
            fcl_loss = self._feature_cl_loss(users, items, user_embs, item_embs)

        dau_loss = None
        if (self.directau_align > 0) or (self.directau_unif > 0):
            dau_loss = self._directau_loss(users, items, user_embeddings, item_embeddings)

        return sup_logits, infonce, fcl_loss, dau_loss

    @staticmethod
    def _off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _poly_term(self, c):
        # RecDCL polynomial-kernel uniformity: log mean((a*C + c0)^degree). The
        # diagonal of C is ~1 (BatchNorm-standardised dims), so minimising this drives
        # the off-diagonal (inter-dimension) correlations to 0 -> orthogonal features.
        v = (self.fcl_a * c + self.fcl_polyc).clamp(min=1e-12) ** self.fcl_degree
        return v.mean().clamp(min=1e-12).log()

    def _feature_cl_loss(self, users, items, user_embs, item_embs):
        """Feature-wise contrastive (FCL) loss, applied to the *batch* of fused main
        embeddings (representation layer only). Mirrors RecDCL (arXiv:2401.15635)."""
        b = user_embs.size(0)
        zu = self.fcl_bn(self.fcl_projector(F.normalize(user_embs, dim=-1)))
        zi = self.fcl_bn(self.fcl_projector(F.normalize(item_embs, dim=-1)))
        loss = 0.0
        if self.fcl_poly > 0:
            cu = zu.t() @ zu / b
            ci = zi.t() @ zi / b
            loss = loss + self.fcl_poly * (0.5 * self._poly_term(cu) + 0.5 * self._poly_term(ci))
        if self.fcl_bt > 0:
            c = zu.t() @ zi / b
            on_diag = (torch.diagonal(c) - 1.0).pow(2).mean()
            off_diag = self._off_diagonal(c).pow(2).mean()
            loss = loss + self.fcl_bt * (on_diag + self.fcl_bt_off * off_diag)
        if self.fcl_off > 0:
            cu = zu.t() @ zu / b
            ci = zi.t() @ zi / b
            loss = loss + self.fcl_off * (self._off_diagonal(cu).pow(2).mean()
                                          + self._off_diagonal(ci).pow(2).mean())
        if self.fcl_mom > 0:
            if self.u_target_his is None:
                self.u_target_his = torch.randn((self.num_users, self.embed_dim),
                                                device=user_embs.device)
                self.i_target_his = torch.randn((self.num_items, self.embed_dim),
                                                device=item_embs.device)
            with torch.no_grad():
                u_target = self.u_target_his[users].detach() * self.fcl_momentum \
                    + user_embs.detach() * (1.0 - self.fcl_momentum)
                i_target = self.i_target_his[items].detach() * self.fcl_momentum \
                    + item_embs.detach() * (1.0 - self.fcl_momentum)
                self.u_target_his[users] = user_embs.detach()
                self.i_target_his[items] = item_embs.detach()
            pu = self.fcl_predictor(user_embs)
            pi = self.fcl_predictor(item_embs)
            mom = 0.5 * (-F.cosine_similarity(pu, i_target.detach(), dim=-1).mean())
            mom = mom + 0.5 * (-F.cosine_similarity(pi, u_target.detach(), dim=-1).mean())
            loss = loss + self.fcl_mom * mom
        return loss

    def _directau_loss(self, users, items, user_embeddings, item_embeddings):
        """DirectAU (arXiv:2206.12811) alignment & uniformity on the unit hypersphere.

        alignment: minimise ||u - i||^2 over positive (user,item) pairs (sum form, so the
        weight is on the same scale as the summed BPR/InfoNCE losses).
        uniformity: minimise log E[exp(-t * d)] over pairs of sampled nodes (users and
        items separately), pushing the embedding distribution toward a uniform spread on
        the hypersphere. Only the representation layer is touched; the propagation
        operator is untouched (respects the R4/R6 constraint).
        """
        u = F.normalize(user_embeddings[users], dim=1)
        i = F.normalize(item_embeddings[items], dim=1)
        loss = 0.0
        if self.directau_align > 0:
            loss = loss + self.directau_align * ((u - i) ** 2).sum()
        if self.directau_unif > 0:
            k = min(self.directau_k, self.num_users, self.num_items)
            uid = torch.randint(0, self.num_users, (k,), device=u.device)
            iid = torch.randint(0, self.num_items, (k,), device=u.device)
            uu = F.normalize(user_embeddings[uid], dim=1)
            ii = F.normalize(item_embeddings[iid], dim=1)
            d_u = 2.0 - 2.0 * (uu @ uu.t())          # squared L2 on unit sphere = 2-2cos
            d_i = 2.0 - 2.0 * (ii @ ii.t())
            unif_u = torch.log(torch.exp(-self.directau_t * d_u).mean() + 1e-12)
            unif_i = torch.log(torch.exp(-self.directau_t * d_i).mean() + 1e-12)
            loss = loss + self.directau_unif * (unif_u + unif_i)
        return loss

    def _apply_global(self, x):
        """Global-view propagation with the rank-q SVD approximation of the normalised
        adjacency: (U_q diag(S_q) V_q^T) x, evaluated in low-rank form (O(n*q*d)) so the
        dense n x n reconstruction is never materialised."""
        return self.gl_U @ (self.gl_S.unsqueeze(1) * (self.gl_Vt @ x))

    def _lightgcl_infonce(self, users, items):
        """Original LightGCL (ICLR'23, arXiv:2302.08191) local-global contrastive loss.

        Local view = the clean LightGCN main-branch embeddings; global view = the SAME
        local embeddings propagated through the rank-q SVD-reconstructed *bipartite*
        adjacency A^ = U_q S_q V_q^T:
            g_l^(u) = U_q S_q V_q^T . E_{l-1}^(v),   g_l^(v) = V_q S_q U_q^T . E_{l-1}^(u)
        InfoNCE contrasts z_l (local) vs g_l (global), user- and item-side. With
        lightgcl_perlayer=1 the loss is summed over layers l=0..L (paper eq.6); with 0
        only the final mean-pooled local embedding is contrasted against its global view
        (cheaper single-step variant). This is NOT the Round-7 noise-vs-global mixed view.
        """
        layers = self._forward_gcn(self.norm_adj, return_all=True,
                                   use_freq=self.freq_on_main)
        UqS = self.gl_UqS
        VqS = self.gl_VqS
        if self.lightgcl_perlayer:
            n_steps = self.n_layers + 1
        else:
            # single-step: only the final (mean-pooled) local embeddings
            u_f = torch.stack([u for u, _ in layers], dim=1).mean(dim=1)
            i_f = torch.stack([i for _, i in layers], dim=1).mean(dim=1)
            layers = [(u_f, i_f)]
            n_steps = 1
        loss = 0.0
        for step in range(n_steps):
            u_z, i_z = layers[step]
            if self.lightgcl_perlayer:
                if step == 0:
                    u_prev, i_prev = layers[0]
                else:
                    u_prev, i_prev = layers[step - 1]
            else:
                u_prev, i_prev = u_z, i_z
            g_u = UqS @ (VqS.t() @ i_prev)    # [I, q] @ [q, J] @ [J, d]
            g_i = VqS @ (UqS.t() @ u_prev)    # [J, q] @ [q, I] @ [I, d]
            u_z = F.normalize(u_z, dim=1)
            i_z = F.normalize(i_z, dim=1)
            g_u = F.normalize(g_u, dim=1)
            g_i = F.normalize(g_i, dim=1)
            bu = F.embedding(users, u_z)
            bi = F.embedding(items, i_z)
            pos_u = inner_product(bu, F.embedding(users, g_u))
            pos_i = inner_product(bi, F.embedding(items, g_i))
            tot_u = torch.matmul(bu, g_u.t())
            tot_i = torch.matmul(bi, g_i.t())
            cu = torch.logsumexp((tot_u - pos_u[:, None]) / self.ssl_temp, dim=1)
            ci = torch.logsumexp((tot_i - pos_i[:, None]) / self.ssl_temp, dim=1)
            loss = loss + torch.sum(cu) + torch.sum(ci)
        return loss

    def _forward_gcn(self, norm_adj, return_all=False, perturb=False, eps=0.0, use_freq=None,
                     use_gl=False, apply_proto=False, apply_dropout=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        if apply_dropout and self.dropout_rate > 0:
            # Round-25: input/node dropout on the MAIN branch only (BPR overfitting
            # regulariser). The contrastive noise views never pass apply_dropout=True, so
            # the SSL self-consistency signal is unaffected. nn.Dropout is a no-op when
            # self.training is False, hence eval()/predict stay deterministic.
            ego_embeddings = self.dropout(ego_embeddings)
        # Round-16: noise_mode=1 (XSimGCL) perturbs the layer-0 embedding ONCE and then
        # propagates it cleanly; noise_mode=0 (SimGCL, default) perturbs after every layer.
        if perturb and eps > 0 and self.noise_mode == 1:
            random_noise = torch.rand_like(ego_embeddings)
            scale = eps if self.noise_deg_scale is None else eps * self.noise_deg_scale.view(-1, 1)
            ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise, dim=1) * scale
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            prev = ego_embeddings
            if use_gl:
                ego_embeddings = self._apply_global(ego_embeddings)
            elif isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            if self.gate_mode == 1 and self.gate_proj is not None:
                # adaptive blend of propagated signal and identity (anti-oversmoothing);
                # gate computed from the pre-propagation embedding, shared across main /
                # noise-view / eval calls so the operator stays identical (R4/R6).
                gate = torch.sigmoid(self.gate_proj[k](prev))
                ego_embeddings = gate * ego_embeddings + (1.0 - gate) * prev
            if perturb and eps > 0 and self.noise_mode == 0:
                # SimGCL-style embedding-space perturbation: uniform noise on the unit
                # sphere, signed by the embedding, scaled by eps. Applied per propagation
                # layer to inject structural-independent stochasticity (no edge dropping).
                # Round-28: when noise_deg_scale is set, the magnitude becomes density-
                # aware (high-degree nodes over-smooth more -> stronger perturbation).
                # Round-33: when noise_layer_gamma!=0 the magnitude also becomes depth-
                # aware (scale_k = eps * (1 + gamma*k/(L-1))), so later layers receive
                # a stronger re-diversifying perturbation to counter depth-accumulated
                # over-smoothing (cf. E11).
                random_noise = torch.rand_like(ego_embeddings)
                scale = eps if self.noise_deg_scale is None else eps * self.noise_deg_scale.view(-1, 1)
                if self.noise_layer_gamma != 0.0 and self.n_layers > 1:
                    scale = scale * (1.0 + self.noise_layer_gamma * (float(k) / float(self.n_layers - 1)))
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise, dim=1) * scale
            all_embeddings += [ego_embeddings]

        if return_all:
            layer_list = []
            for emb in all_embeddings:
                u, i = torch.split(emb, [self.num_users, self.num_items], dim=0)
                layer_list.append((u, i))
            return layer_list

        if self.freq_alphas is not None and (use_freq is None or use_freq):
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
        if apply_proto and self.proto_K > 0 and self.proto_lam > 0 and self.proto_centers is not None:
            cu = self.proto_centers[self.proto_assign_u]
            ci = self.proto_centers[self.proto_assign_i]
            user_embeddings = (1.0 - self.proto_lam) * user_embeddings + self.proto_lam * cu
            item_embeddings = (1.0 - self.proto_lam) * item_embeddings + self.proto_lam * ci

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
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(
            self.norm_adj, use_freq=self.freq_on_main, apply_proto=True)
        if self.l2_norm:
            self._user_embeddings_final = F.normalize(self._user_embeddings_final, dim=1)
            self._item_embeddings_final = F.normalize(self._item_embeddings_final, dim=1)


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
        # Round-30: InfoNCE temperature schedule (MM-TS WACV'26 / DySTreSS / RL temperature
        # curriculum). ssl_temp_anneal=0 keeps the fixed ssl_temp (backward compatible);
        # 1 = linear, 2 = cosine interpolation of the InfoNCE temperature from
        # ssl_temp_start to ssl_temp_end over ssl_temp_anneal_epochs epochs (0 => full
        # training budget), then holds. Operates on the *shape* of the InfoNCE (which
        # negatives the gradient concentrates on) rather than its overall weight
        # (R13/R15 ssl_anneal), so it is orthogonal to all 3 fusion components. Static
        # schedule, no learnable parameters (respects the R1/R15/R18 'static wins' law).
        self.ssl_temp_anneal = int(config["ssl_temp_anneal"]) if "ssl_temp_anneal" in config else 0
        assert self.ssl_temp_anneal in (0, 1, 2)
        self.ssl_temp_anneal_epochs = int(config["ssl_temp_anneal_epochs"]) if "ssl_temp_anneal_epochs" in config else 0
        self.ssl_temp_start = float(config["ssl_temp_start"]) if "ssl_temp_start" in config else self.ssl_temp
        self.ssl_temp_end = float(config["ssl_temp_end"]) if "ssl_temp_end" in config else self.ssl_temp
        self.ssl_warmup_epochs = config["ssl_warmup_epochs"] if "ssl_warmup_epochs" in config else 0
        self.ssl_anneal = int(config["ssl_anneal"]) if "ssl_anneal" in config else 0
        self.ssl_anneal_floor = float(config["ssl_anneal_floor"]) if "ssl_anneal_floor" in config else 0.0
        # Round-15: fixed anneal window (ssl_anneal_epochs>0) decays lambda to floor over
        # exactly W epochs and then holds floor (pure-BPR fine-tune) for the rest; 0 keeps
        # the R13 behaviour (anneal over the full self.epochs budget).
        self.ssl_anneal_epochs = int(config["ssl_anneal_epochs"]) if "ssl_anneal_epochs" in config else 0
        self.mlc_start_layer = config["mlc_start_layer"] if "mlc_start_layer" in config else 0
        self.mlc_layer_weight = config["mlc_layer_weight"] if "mlc_layer_weight" in config else "uniform"
        self.ssl_degree_alpha = config["ssl_degree_alpha"] if "ssl_degree_alpha" in config else 0.5
        self.noise_eps = config["noise_eps"] if "noise_eps" in config else 0.1
        # Round-16: 0 = SimGCL per-layer noise (default); 1 = XSimGCL single-shot input noise.
        self.noise_mode = int(config["noise_mode"]) if "noise_mode" in config else 0
        assert self.noise_mode in (0, 1)
        # Round-20: noise curriculum (epsilon schedule). noise_anneal=0 keeps the fixed
        # noise_eps (backward compatible); 1 = linear, 2 = cosine decay of the per-layer
        # embedding-noise magnitude from noise_eps down to noise_eps_floor over the first
        # noise_anneal_epochs epochs (0 => full training budget). Purely a static schedule
        # (no learnable parameters), consistent with the R1/R15/R18 "static wins" law.
        self.noise_anneal = int(config["noise_anneal"]) if "noise_anneal" in config else 0
        assert self.noise_anneal in (0, 1, 2)
        self.noise_anneal_epochs = int(config["noise_anneal_epochs"]) if "noise_anneal_epochs" in config else 0
        self.noise_eps_floor = float(config["noise_eps_floor"]) if "noise_eps_floor" in config else 0.0
        # Round-28: density-aware noise strength. noise_deg_gamma=0 keeps uniform
        # noise_eps (backward compatible). gamma>0 scales the per-node noise magnitude
        # linearly with degree centrality: scale_i = 1 + gamma*(c_i - mean(c)),
        # c_i = deg_i / max(deg). High-degree nodes (which aggregate more neighbours
        # and thus over-smooth more, cf. E11) get stronger perturbation, low-degree
        # nodes slightly weaker; mean scale stays ~1 so the average noise magnitude
        # is unchanged. Static, parameter-free (respects the R1/R15/R18 'static wins').
        self.noise_deg_gamma = float(config["noise_deg_gamma"]) if "noise_deg_gamma" in config else 0.0
        # Round-33: depth-aware (layer-wise) noise schedule. 0 = uniform per-layer
        # magnitude (backward compatible). gamma>0 -> deeper layers get a stronger
        # perturbation (depth-accumulated over-smoothing, cf. E11); gamma<0 -> reverse.
        # Static and orthogonal to the node axis (noise_deg_gamma) and epoch axis
        # (noise_anneal); respects the R1/R15/R18 'static wins' law.
        self.noise_layer_gamma = float(config["noise_layer_gamma"]) if "noise_layer_gamma" in config else 0.0
        # Round-19: adaptive per-node per-layer gating (AdaGate-style). gate_mode=1 turns
        # on the gated residual in _forward_gcn; gate_init sets the initial sigmoid bias
        # (positive => gate starts near 1, i.e. close to plain LightGCN, avoiding the
        # R4 slow-convergence pitfall of a main-branch structural change).
        self.gate_mode = int(config["gate_mode"]) if "gate_mode" in config else 0
        assert self.gate_mode in (0, 1)
        self.gate_init = float(config["gate_init"]) if "gate_init" in config else 2.0
        self.use_mlc = config["use_mlc"] if "use_mlc" in config else 1
        # Round-2: hardness-calibrated (semi-hard) negative sampling.
        self.neg_mode = config["neg_mode"] if "neg_mode" in config else "uniform"
        assert self.neg_mode in ["uniform", "semihard"]
        self.neg_K = int(config["neg_K"]) if "neg_K" in config else 5
        # Round-21: multi-negative BPR (sampled-softmax-style ranking margin). K>1 draws
        # K independent negatives per positive (lower-variance ranking gradient). Static
        # sampling change only -> no learnable parameters (consistent with the R1/R15/R18
        # "static wins" law). Default 1 = backward compatible.
        self.num_negatives = int(config["num_negatives"]) if "num_negatives" in config else 1
        assert self.num_negatives >= 1
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
        # Round-4: scope of the scaler (1/1 = Round-3 behaviour).
        self.freq_on_main = int(config["freq_on_main"]) if "freq_on_main" in config else 1
        self.freq_on_views = int(config["freq_on_views"]) if "freq_on_views" in config else 1
        # Round-5: NT-SSM (arXiv:2605.24015) type-aware neighbour-pair reweighting,
        # operationalised (simplified) as a per-pair structural-type weight on the BPR
        # gradient. Type = popularity stratum of the (user,item) positive pair; the
        # propagation filter is deliberately NOT touched (respects the Round-4 finding
        # that main/view filters must stay identical). Default off -> backward compatible.
        self.nt_ssm = int(config["nt_ssm"]) if "nt_ssm" in config else 0
        self.nt_ssm_alpha = float(config["nt_ssm_alpha"]) if "nt_ssm_alpha" in config else 0.5
        # Round-6: view-composition ablation (structural ED view vs embedding-noise view).
        #   "nn"  = two embedding-noise views (current NoiseSGL -> default, backward compatible)
        #   "ne"  = one noise view + one edge-dropout structural view (SGL-style)
        #   "nee" = two noise views + one edge-dropout view (3 views, averaged pairwise InfoNCE)
        #   "ee"  = two edge-dropout views
        # Round-7: view_mix generalised to a string over {n: embedding-noise, e: edge-
        # dropout structural, g: SVD global}. "nn" (default) = backward compatible.
        # Round-8: RecDCL feature-wise CL coefficients (all default 0 = off).
        self.fcl_poly = float(config["fcl_poly"]) if "fcl_poly" in config else 0.0
        self.fcl_bt = float(config["fcl_bt"]) if "fcl_bt" in config else 0.0
        self.fcl_off = float(config["fcl_off"]) if "fcl_off" in config else 0.0
        self.fcl_mom = float(config["fcl_mom"]) if "fcl_mom" in config else 0.0
        self.fcl_degree = int(config["fcl_degree"]) if "fcl_degree" in config else 4
        self.fcl_momentum = float(config["fcl_momentum"]) if "fcl_momentum" in config else 0.1
        # Round-9: DirectAU alignment & uniformity (all default 0 = off).
        self.directau_align = float(config["directau_align"]) if "directau_align" in config else 0.0
        self.directau_unif = float(config["directau_unif"]) if "directau_unif" in config else 0.0
        self.directau_t = float(config["directau_t"]) if "directau_t" in config else 2.0
        self.directau_k = int(config["directau_k"]) if "directau_k" in config else 512
        # Round-23: L2-normalised (cosine) scoring (4th-component candidate). Static,
        # parameter-free; normalises the final fused embeddings before BPR/eval scoring.
        self.l2_norm = int(config["l2_norm"]) if "l2_norm" in config else 0
        assert self.l2_norm in (0, 1)
        # Round-25: main-branch embedding (input/node) dropout rate. Static, parameter-
        # free regulariser for the BPR fine-tune phase (the R12c/R15 peak-then-overfit
        # problem). 0 = off (backward compatible).
        self.dropout = float(config["dropout"]) if "dropout" in config else 0.0
        assert 0.0 <= self.dropout < 1.0
        # Round-11: NCL-style prototype (cluster) interest injection.
        self.proto_K = int(config["proto_K"]) if "proto_K" in config else 0
        self.proto_lam = float(config["proto_lam"]) if "proto_lam" in config else 0.0
        self.proto_interval = int(config["proto_interval"]) if "proto_interval" in config else 1
        self.proto_iters = int(config["proto_iters"]) if "proto_iters" in config else 10
        self.proto_seed = int(config["proto_seed"]) if "proto_seed" in config else 0
        self.view_mix = str(config["view_mix"]) if "view_mix" in config else "nn"
        assert len(self.view_mix) >= 1 and all(c in "neg" for c in self.view_mix), \
            "view_mix must be a non-empty string over {n,e,g}, got %r" % self.view_mix
        assert self.view_mix.count("e") <= 2, "at most 2 edge-dropout views supported"
        self.use_gl = "g" in self.view_mix
        # Rank of the truncated SVD used for the global view (LightGCL uses q=5).
        self.gl_rank = int(config["gl_rank"]) if "gl_rank" in config else 5
        # Round-17: original LightGCL local-vs-global contrast (arXiv:2302.08191).
        # lightgcl=1 replaces the noise-view InfoNCE with a clean-local-vs-SVD-global
        # contrast. lightgcl_perlayer=1 sums InfoNCE over layers 0..L (paper eq.6),
        # 0 uses only the final mean-pooled embedding (cheaper single-step variant).
        self.lightgcl = int(config["lightgcl"]) if "lightgcl" in config else 0
        self.lightgcl_perlayer = int(config["lightgcl_perlayer"]) if "lightgcl_perlayer" in config else 1

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
        self.nt_ssm_w = None
        user_degree = np.array(self.dataset.train_csr_mat.sum(1)).reshape(-1).astype(np.float32)
        item_degree = np.array(self.dataset.train_csr_mat.sum(0)).reshape(-1).astype(np.float32)
        self.user_ssl_weights = np.power(user_degree + 1.0, -float(self.ssl_degree_alpha))
        self.item_ssl_weights = np.power(item_degree + 1.0, -float(self.ssl_degree_alpha))
        self.user_ssl_weights = self.user_ssl_weights / self.user_ssl_weights.mean()
        self.item_ssl_weights = self.item_ssl_weights / self.item_ssl_weights.mean()
        self.user_ssl_weights = torch.from_numpy(self.user_ssl_weights).float().to(self.device)
        self.item_ssl_weights = torch.from_numpy(self.item_ssl_weights).float().to(self.device)
        self._noise_deg_scale = None
        if self.noise_deg_gamma != 0.0:
            deg = np.concatenate([user_degree, item_degree], axis=0).astype(np.float32)
            c = deg / (float(deg.max()) + 1e-12)          # degree centrality in [0,1]
            c_mean = float(c.mean())
            scale = 1.0 + self.noise_deg_gamma * (c - c_mean)  # mean ~1, max ~1+gamma
            scale = np.clip(scale, 0.1, 10.0)
            self._noise_deg_scale = torch.from_numpy(scale).float().to(self.device)
            self.logger.info("[noise-deg] gamma=%.2f scale[min/mean/max]=%.3f/%.3f/%.3f" % (
                self.noise_deg_gamma, float(scale.min()), float(scale.mean()), float(scale.max())))
        if self.nt_ssm:
            # Precompute one weight per training positive pair: w = (deg_u+1)^-a * (deg_i+1)^-a,
            # normalised to mean 1 so the overall BPR scale is unchanged. Low-degree (tail)
            # neighbour pairs receive a larger share of the gradient (the "type-aware" part).
            npairs = self.dataset.train_data.to_user_item_pairs()
            pu = npairs[:, 0].astype(np.int64)
            pi = npairs[:, 1].astype(np.int64)
            w = np.power(user_degree[pu] + 1.0, -self.nt_ssm_alpha) * \
                np.power(item_degree[pi] + 1.0, -self.nt_ssm_alpha)
            w = (w / w.mean()).astype(np.float32)
            ni = int(self.num_items)
            self.nt_ssm_w = {int(pu[k]) * ni + int(pi[k]): float(w[k]) for k in range(len(pu))}
            self.logger.info("[nt-ssm] alpha=%.2f pairs=%d w[min/mean/max]=%.3f/%.3f/%.3f" % (
                self.nt_ssm_alpha, len(pu), float(w.min()), float(w.mean()), float(w.max())))
        adj_matrix = self.create_adj_mat()
        if self.use_gl:
            self._init_global_view(adj_matrix)
        if self.lightgcl:
            self._init_lightgcl_global()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.use_gl:
            self.lightgcn.gl_U = torch.from_numpy(self.gl_U).float().to(self.device)
            self.lightgcn.gl_S = torch.from_numpy(self.gl_S).float().to(self.device)
            self.lightgcn.gl_Vt = torch.from_numpy(self.gl_Vt).float().to(self.device)
        if self.lightgcl:
            self.lightgcn.gl_UqS = torch.from_numpy(self.gl_UqS).float().to(self.device)
            self.lightgcn.gl_VqS = torch.from_numpy(self.gl_VqS).float().to(self.device)
        self.lightgcn.freq_on_main = bool(self.freq_on_main)
        self.lightgcn.freq_on_views = bool(self.freq_on_views)
        self.lightgcn.view_mix = self.view_mix
        self.lightgcn.ssl_temp = self.ssl_temp
        self.lightgcn.noise_mode = self.noise_mode
        self.lightgcn.noise_deg_scale = self._noise_deg_scale
        self.lightgcn.noise_layer_gamma = self.noise_layer_gamma
        self.lightgcn.l2_norm = bool(self.l2_norm)
        self.lightgcn.dropout_rate = self.dropout
        self.lightgcn.dropout = nn.Dropout(self.dropout)
        self.lightgcn.gate_mode = self.gate_mode
        if self.gate_mode == 1:
            self.lightgcn.gate_proj = nn.ModuleList(
                [nn.Linear(self.emb_size, 1, bias=True) for _ in range(self.n_layers)]
            ).to(self.device)
            for proj in self.lightgcn.gate_proj:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, self.gate_init)
            self.logger.info("[gate] gate_mode=1 gate_init=%.2f n_layers=%d" % (
                self.gate_init, self.n_layers))
        self.lightgcn.lightgcl = bool(self.lightgcl)
        self.lightgcn.lightgcl_perlayer = int(self.lightgcl_perlayer)
        self.lightgcn.fcl_poly = self.fcl_poly
        self.lightgcn.fcl_bt = self.fcl_bt
        self.lightgcn.fcl_off = self.fcl_off
        self.lightgcn.fcl_mom = self.fcl_mom
        self.lightgcn.fcl_degree = self.fcl_degree
        self.lightgcn.fcl_momentum = self.fcl_momentum
        self.lightgcn.directau_align = self.directau_align
        self.lightgcn.directau_unif = self.directau_unif
        self.lightgcn.directau_t = self.directau_t
        self.lightgcn.directau_k = self.directau_k
        self.lightgcn.proto_K = self.proto_K
        self.lightgcn.proto_lam = self.proto_lam
        if self.proto_K > 0:
            self.logger.info("[proto] K=%d lam=%.3f interval=%d iters=%d seed=%d" % (
                self.proto_K, self.proto_lam, self.proto_interval, self.proto_iters, self.proto_seed))
        if self.fcl_poly or self.fcl_bt or self.fcl_off or self.fcl_mom:
            self.logger.info("[fcl] poly=%.3f bt=%.3f off=%.3f mom=%.3f degree=%d momentum=%.2f" % (
                self.fcl_poly, self.fcl_bt, self.fcl_off, self.fcl_mom,
                self.fcl_degree, self.fcl_momentum))
        if self.directau_align or self.directau_unif:
            self.logger.info("[directau] align=%.3f unif=%.3f t=%.2f k=%d" % (
                self.directau_align, self.directau_unif, self.directau_t, self.directau_k))
        self.logger.info("[view-mix] view_mix=%s (n=noise/e=edge-dropout/g=svd-global)" % self.view_mix)
        if self.noise_anneal:
            self.logger.info("[noise-curriculum] anneal=%d epochs=%d eps=%.3f->%.3f" % (
                self.noise_anneal, self.noise_anneal_epochs, self.noise_eps, self.noise_eps_floor))
        if self.ssl_temp_anneal:
            self.logger.info("[temp-sched] anneal=%d epochs=%d tau=%.3f->%.3f" % (
                self.ssl_temp_anneal, self.ssl_temp_anneal_epochs, self.ssl_temp_start, self.ssl_temp_end))
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
            self.logger.info("[freq-scaler] type=%s on_main=%d on_views=%d alphas=%s" % (
                self.freq_type, self.freq_on_main, self.freq_on_views,
                np.round(self.lightgcn.freq_alphas.cpu().numpy(), 4).tolist()))
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    def _init_global_view(self, adj):
        """Truncated SVD of the (symmetric) normalised adjacency: A ~= U_q diag(S_q) V_q^T.
        This is the LightGCL-style *denoised global* collaborative-relation operator.
        Kept as low-rank factors; the dense n x n matrix is never formed."""
        from scipy.sparse.linalg import svds
        t0 = time()
        q = int(min(self.gl_rank, min(adj.shape) - 1))
        u, s, vt = svds(adj, k=q)
        order = np.argsort(s)[::-1]
        self.gl_U = np.ascontiguousarray(u[:, order], dtype=np.float32)
        self.gl_S = np.ascontiguousarray(s[order], dtype=np.float32)
        self.gl_Vt = np.ascontiguousarray(vt[order, :], dtype=np.float32)
        # fraction of ||A||_F^2 captured by the top-q singular values (global-view fidelity)
        fro2 = float(adj.multiply(adj).sum())
        energy = float(np.sum(self.gl_S ** 2)) / max(fro2, 1e-12)
        self.logger.info("[gl-view] rank=%d svd_time=%.1fs energy_frac=%.4f s_top=%s" % (
            q, time() - t0, energy, np.round(self.gl_S[:5], 3).tolist()))

    def _init_lightgcl_global(self):
        """Round-17: truncated SVD of the *bipartite* normalised interaction matrix
        R_norm = D_u^{-1/2} R D_i^{-1/2} (I x J), giving the original LightGCL global-view
        operator A^ = U_q S_q V_q^T. Stored as U_q S_q ([I, q]) and V_q S_q ([J, q]); the
        dense I x J reconstruction is never materialised."""
        from scipy.sparse.linalg import svds
        t0 = time()
        R = self.dataset.train_csr_mat.astype(np.float32).tocsr()
        u_deg = np.asarray(R.sum(1)).ravel() + 1e-12
        i_deg = np.asarray(R.sum(0)).ravel() + 1e-12
        R_norm = (sp.diags(1.0 / np.sqrt(u_deg)) @ R @ sp.diags(1.0 / np.sqrt(i_deg))).tocsr()
        q = int(min(self.gl_rank, min(R_norm.shape) - 1))
        u, s, vt = svds(R_norm, k=q)
        order = np.argsort(s)[::-1]
        u = u[:, order]
        s = s[order]
        vt = vt[order, :]
        self.gl_UqS = np.ascontiguousarray(u * s[None, :], dtype=np.float32)   # [I, q]
        self.gl_VqS = np.ascontiguousarray(vt.T * s[None, :], dtype=np.float32)  # [J, q]
        fro2 = float(R_norm.multiply(R_norm).sum())
        energy = float(np.sum(s ** 2)) / max(fro2, 1e-12)
        self.logger.info("[lightgcl] rank=%d svd_time=%.1fs energy_frac=%.4f s_top=%s perlayer=%d" % (
            q, time() - t0, energy, np.round(s[:5], 3).tolist(), self.lightgcl_perlayer))

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
        if self.num_negatives > 1:
            return self._multi_neg_sampler()
        return PairwiseSamplerV2(self.dataset.train_data, num_neg=1,
                                 batch_size=self.batch_size, shuffle=True)

    def _multi_neg_sampler(self):
        """Round-21: pairwise sampler with K>1 negative items per positive (sampled
        softmax-style ranking margin). Mirrors PairwiseSamplerV2's uniform (user, positive)
        sampling exactly (same number of samples and the same user distribution); the only
        change is that each positive is paired with K independently-drawn negatives, so the
        BPR gradient is averaged over K candidates (lower variance). Yields neg_items of
        shape [batch, K]; num_negatives=1 keeps the original V2 sampler untouched.
        """
        from collections import defaultdict
        K = int(self.num_negatives)
        user_pos_dict = self.dataset.train_data.to_user_dict()
        num_item = int(self.num_items)
        user_arr = np.array(list(user_pos_dict.keys()), dtype=np.int64)
        num_samples = sum(len(v) for v in user_pos_dict.values())
        # One pass per epoch (the training loop re-creates this generator every epoch,
        # mirroring PairwiseSamplerV2). No while-True: the generator must terminate so
        # each epoch yields exactly num_samples (user,pos) pairs.
        users = user_arr[randint_choice(len(user_arr), size=num_samples, replace=True)]
        users = np.asarray(users, dtype=np.int64)
        counts = defaultdict(int)
        for u in users:
            counts[u] += 1
        pos_by_user = {}
        neg_by_user = {}
        for u, c in counts.items():
            ui = int(u)
            pos_items = user_pos_dict[ui]
            pi = randint_choice(len(pos_items), size=c, replace=True)
            pi = pi if isinstance(pi, (list, np.ndarray)) else [pi]
            pos_by_user[u] = np.asarray(pos_items[np.asarray(pi, dtype=np.int64)], dtype=np.int64)
            neg = randint_choice(num_item, size=c * K, replace=True, exclusion=user_pos_dict[ui])
            neg_by_user[u] = np.asarray(neg, dtype=np.int64).reshape(c, K)
        cursor = defaultdict(int)
        pos_arr = np.empty(num_samples, dtype=np.int64)
        neg_arr = np.empty((num_samples, K), dtype=np.int64)
        for i, u in enumerate(users):
            ci = cursor[u]
            pos_arr[i] = pos_by_user[u][ci]
            neg_arr[i] = neg_by_user[u][ci]
            cursor[u] = ci + 1
        order = np.random.permutation(num_samples)
        for s0 in range(0, num_samples, self.batch_size):
            idx = order[s0:s0 + self.batch_size]
            yield (users[idx].astype(np.int64), pos_arr[idx], neg_arr[idx])

    def _cluster_prototypes(self):
        """Round-11: k-means over the *main-branch* node embeddings (users+items) into K
        prototypes. Runs on CPU via scipy kmeans2 (minit='points', deterministic seed);
        the per-batch injection in _forward_gcn then only gathers the centres (O(n*d)).
        Clustering is on the pre-injection embeddings (apply_proto=False) to avoid a
        self-reinforcing feedback loop.
        """
        from scipy.cluster.vq import kmeans2
        with torch.no_grad():
            ue, ie = self.lightgcn._forward_gcn(self.lightgcn.norm_adj,
                                                use_freq=self.lightgcn.freq_on_main)
            Z = torch.cat([ue, ie], dim=0)
            Z = Z.cpu().numpy().astype(np.float64)
        n = Z.shape[0]
        K = int(min(self.proto_K, n - 1))
        t0 = time()
        centers, labels = kmeans2(Z, K, iter=self.proto_iters, minit="points",
                                  seed=self.proto_seed)
        centers = torch.from_numpy(centers.astype(np.float32)).to(self.device)
        assign = torch.from_numpy(labels.astype(np.int64)).to(self.device)
        self.lightgcn.proto_centers = centers
        self.lightgcn.proto_assign_u = assign[:self.num_users]
        self.lightgcn.proto_assign_i = assign[self.num_users:]
        self.logger.info("[proto] re-cluster K=%d time=%.1fs" % (K, time() - t0))

    def _ssl_temp(self, epoch):
        """Effective InfoNCE temperature at a given epoch (Round-30: temperature schedule).

        ssl_temp_anneal=0 keeps the fixed ssl_temp (backward compatible). Otherwise the
        InfoNCE temperature is interpolated from ssl_temp_start to ssl_temp_end over
        ssl_temp_anneal_epochs epochs (linear or cosine) and then held. Unlike the
        SSL-weight annealing (R13/R15, which scales the whole InfoNCE magnitude), this
        changes the *sharpness* of the contrastive distribution -- which negatives the
        gradient concentrates on -- so it is orthogonal to all existing fusion modules.
        Motivated by MM-TS (temperature schedules for long-tail contrastive learning,
        arXiv:2603.08202) and DySTreSS (dynamic temperature scaling).
        """
        if not self.ssl_temp_anneal or self.ssl_temp_start == self.ssl_temp_end or self.epochs <= 0:
            return self.ssl_temp_start
        total = float(self.ssl_temp_anneal_epochs) if self.ssl_temp_anneal_epochs > 0 else float(self.epochs)
        t = min(max(float(epoch - 1) / float(max(total - 1, 1.0)), 0.0), 1.0)
        a, b = self.ssl_temp_start, self.ssl_temp_end
        if self.ssl_temp_anneal == 1:          # linear interpolation
            sched = a + (b - a) * t
        elif self.ssl_temp_anneal == 2:        # cosine interpolation
            sched = b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            sched = a
        return max(sched, 1e-4)

    def _noise_eps(self, epoch):
        """Effective per-layer noise magnitude at a given epoch (Round-20: noise curriculum).

        noise_anneal=0 keeps the fixed noise_eps (backward compatible). Otherwise the
        noise is decayed from noise_eps to noise_eps_floor over noise_anneal_epochs
        epochs (linear or cosine), mirroring the SSL-weight annealing (R13/R15) but on
        the augmentation strength itself. Rationale: a stronger augmentation early
        builds a robust basin fast (R1/R12), while a weaker/zero augmentation later
        lets the same shared LightGCN operator produce cleaner views for the InfoNCE
        ramp-down, i.e. an augmentation curriculum (task B cross-domain idea).
        """
        if not self.noise_anneal or self.epochs <= 0:
            return self.noise_eps
        total = float(self.noise_anneal_epochs) if self.noise_anneal_epochs > 0 else float(self.epochs)
        t = min(max(float(epoch) / max(total, 1.0), 0.0), 1.0)
        start = self.noise_eps
        floor = self.noise_eps_floor
        if self.noise_anneal == 1:          # linear decay to floor
            sched = start + (floor - start) * t
        elif self.noise_anneal == 2:        # cosine decay to floor
            sched = floor + (start - floor) * 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            sched = start
        return sched

    def _ssl_weight(self, epoch):
        """Multiplier on the InfoNCE loss at a given epoch (R13: SSL annealing).

        Warmup ramp-up reproduces the existing ssl_warmup_epochs behaviour. The new
        annealing ramp-down (ssl_anneal>0) then decays lambda over the remaining
        epochs so the BPR loss can fine-tune; motivated by R12c (gowalla peaks at
        ep14 then monotone overfits to ep49, because the noise-view InfoNCE keeps
        pulling the representation). Default ssl_anneal=0 -> backward compatible.
        """
        if self.ssl_warmup_epochs > 0:
            w = min(1.0, float(epoch) / float(self.ssl_warmup_epochs))
        else:
            w = 1.0
        if not self.ssl_anneal or self.epochs <= 0:
            return w
        warm = self.ssl_warmup_epochs if self.ssl_warmup_epochs > 0 else 0
        if epoch <= warm:
            return w
        total = float(self.ssl_anneal_epochs) if self.ssl_anneal_epochs > 0 else float(self.epochs)
        t = min(max(float(epoch - warm) / float(max(total - warm, 1)), 0.0), 1.0)
        floor = self.ssl_anneal_floor
        if self.ssl_anneal == 1:       # linear decay to floor
            sched = 1.0 - t * (1.0 - floor)
        elif self.ssl_anneal == 2:     # cosine decay to floor
            sched = floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            sched = 1.0
        return w * sched

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            if self.proto_K > 0 and ((epoch - 1) % self.proto_interval == 0):
                self._cluster_prototypes()
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            # NoiseSGL: noise views need no sub-graph; Round-6 ED views need a freshly
            # edge-dropped graph each epoch (SGL-style structural augmentation).
            sub_graph1, sub_graph2 = None, None
            n_ed = self.view_mix.count("e")
            if n_ed >= 1:
                sub_graph1 = sp_mat_to_sp_tensor(
                    self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device)
                if n_ed >= 2:
                    sub_graph2 = sp_mat_to_sp_tensor(
                        self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)).to(self.device)
            data_iter = self._pairwise_iter()
            self.lightgcn.train()
            cur_eps = self._noise_eps(epoch)
            cur_temp = self._ssl_temp(epoch)
            if cur_temp != self.lightgcn.ssl_temp:
                self.lightgcn.ssl_temp = cur_temp
                self.ssl_temp = cur_temp
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, infonce_loss, fcl_loss, dau_loss = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items, eps=cur_eps)
                
                # BPR Loss (Round-5: type-aware per-pair reweighting when nt_ssm=1)
                if self.nt_ssm_w is not None:
                    keys = bat_users.detach().cpu().numpy().astype(np.int64) * np.int64(self.num_items) \
                        + bat_pos_items.detach().cpu().numpy().astype(np.int64)
                    pw = np.fromiter((self.nt_ssm_w.get(int(k), 1.0) for k in keys),
                                     dtype=np.float32, count=len(keys))
                    pw = torch.from_numpy(pw).to(self.device)
                    bpr_loss = -torch.sum(pw * F.logsigmoid(sup_logits))
                else:
                    bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss is computed inside forward() so that >=2 views can be
                # mixed; for view_mix="nn" it is identical to the original single pair.
                # DMLC: multi-layer contrast (disable with use_mlc=0 for pure SimGCL-style).
                if self.use_mlc:
                    layers2_v1 = self.lightgcn._forward_gcn(self.lightgcn.norm_adj, return_all=True,
                                                            perturb=True, eps=cur_eps)
                    layers2_v2 = self.lightgcn._forward_gcn(self.lightgcn.norm_adj, return_all=True,
                                                            perturb=True, eps=cur_eps)

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

                ssl_weight = self._ssl_weight(epoch)
                if fcl_loss is not None:
                    loss = bpr_loss + self.ssl_reg * ssl_weight * infonce_loss \
                        + fcl_loss + self.reg * reg_loss
                else:
                    loss = bpr_loss + self.ssl_reg * ssl_weight * infonce_loss + self.reg * reg_loss
                if dau_loss is not None:
                    loss = loss + dau_loss
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

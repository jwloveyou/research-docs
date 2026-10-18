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
import os
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
        # Round-47: FROZEN LLM semantic item features (task #6, 4th/5th-component
        # semantic axis). item_sem is a plain (non-Parameter) tensor [num_items, embed_dim]
        # built by util/build_semantic_features.py (TF-IDF tag vector -> frozen seeded
        # Gaussian projection -> row L2-normalised). When sem_lam!=0 it is added to the
        # layer-0 item embedding inside _forward_gcn, BEFORE the shared GCN operator and
        # before any SSL noise, so the semantic prior flows identically through the main
        # branch and both noise views (respects R4/R6/R7). Static & parameter-free (the
        # projection is frozen, no projector training) -- the *static* counterpart of
        # LLMRec's learnable feature projector, chosen to respect this repo's
        # R1/R15/R18 'static wins' law. Default None/0.0 = off (backward compatible).
        self.item_sem = None
        self.sem_lam = 0.0
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
        # Round-49: DropEdge-style edge dropout on the MAIN propagation branch only
        # (4th-component candidate #16). dropedge_keep=1.0 keeps the full normalized
        # adjacency (backward compatible); <1.0 drops a fraction (1-keep) of the
        # bipartite training edges ONCE PER EPOCH and re-normalises, so the main
        # (prediction) branch propagates on a stochastically thinned graph during
        # training while eval() still uses the full norm_adj (the standard DropEdge
        # train-with-dropout / test-with-full protocol, Rong et al. ICLR 2020).
        # Mechanism: E11 shows depth-accumulated over-smoothing is still present in the
        # main branch (cos(l0,lL)=0.46 even with noise); the noise views only regularise
        # the CONTRASTIVE branch via InfoNCE, so the main branch currently has no direct
        # anti-over-smoothing. DropEdge breaks the graph connectivity during propagation
        # and is the canonical static remedy. Orthogonal to the 3 fusion components
        # (embedding noise R1 / SSL anneal R15 / capacity R18) and distinct from R25
        # (input-embedding dropout, neutral) and R6/R7 (edge dropout on the CONTRAST
        # views, harmful -- untouched here). Static, parameter-free (R1/R15/R18 law).
        self.dropedge_keep = 1.0
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
        # Round-39: static (non-uniform) layer-aggregation weights. None = uniform
        # mean-pooling over the L+1 layer embeddings (backward compatible). Otherwise a
        # length-(L+1) tensor summing to 1, applied to the FINAL layer pool in
        # _forward_gcn. The same operator is shared by the main branch, the noise views
        # and eval(), so it respects R4/R6/R7 (identical aggregation everywhere).
        self.agg_weights = None
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
        # Round-34: GCNII/APPNP-style *static* initial-residual connection.
        # res_alpha=0 keeps the plain LightGCN propagation (backward compatible);
        # res_alpha>0 blends a fraction of the layer-0 embedding back after each
        # propagation step: h_{k+1} = (1-alpha)*(A h_k) + alpha*h_0. This directly
        # counters depth-accumulated over-smoothing (E11: the final embedding collapses
        # toward the principal eigenvector) by keeping node-idiosyncratic (personalised)
        # signal alive, complementing the noise augmentation's rank-release rather than
        # replacing it. Static and parameter-free (respects the R1/R15/R18 'static wins'
        # law); applied identically on the main branch and the noise views so the
        # propagation operator stays shared (respects R4/R6/R7).
        self.res_alpha = 0.0
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
        # HCCF baseline (SIGIR'22, arXiv:2204.12200, official repo akaxlh/HCCF
        # torchVersion/Model.py). hccf=False keeps the NoiseSGL backbone (backward
        # compatible); hccf=True switches to the faithful HCCF architecture for the
        # No.1-baseline same-protocol comparison (task #8): a *learnable* hypergraph
        # dependency matrix (u_hyper / i_hyper, [d, H]) projects the embeddings onto
        # H hyperedges, and the hypergraph view is HH^T E (HGNN: adj^T@embeds then
        # adj@lat). Per layer the local GCN embedding (edge-dropped in training) is
        # ADDED to the hypergraph embedding, and the final embedding = sum over all
        # L+1 layer embeddings (paper: embeds = sum(lats)). The SSL objective
        # contrasts the *detached* local GCN view against the hypergraph view per
        # layer (user- and item-side), exactly as in the official code (contrastLoss).
        # This is a learnable-structure baseline, deliberately placed side-by-side with
        # our 'static wins' iron law (R4/R19) as the cleanest counterfactual experiment.
        self.hccf = False
        self.hccf_hyper_num = 128
        self.hccf_keep_rate = 0.5
        self.hccf_temp = 1.0
        self.u_hyper = None   # [embed_dim, H] learnable user->hyperedge dependency
        self.i_hyper = None   # [embed_dim, H] learnable item->hyperedge dependency
        # DCCF baseline (SIGIR'23, arXiv:2305.02759, official repo HKUDS/DCCF model.py).
        # dccf=False keeps the NoiseSGL backbone (backward compatible); dccf=True switches
        # to the faithful DCCF architecture for the No.1-baseline same-protocol comparison
        # (task #8): learnable intent prototypes (user_intent/item_intent, [d, K]) route
        # each node's embedding into K latent intents (softmax assignment + reconstruction),
        # then per layer four views are formed -- (gnn) the plain GCN propagation,
        # (int) the intent-aware aggregation, (gaa) the graph-adaptive and (iaa) the
        # intent-adaptive subgraphs (row-normalised cosine-affinity masks learned from the
        # embeddings). The layer state is the residual sum gnn+int+gaa+iaa+prev and the
        # final embedding = sum over all layer states. SSL contrasts the gnn view against
        # int/gaa/iaa (user- and item-side, per layer, batch-limited negatives). This is a
        # learnable-structure baseline, deliberately placed side-by-side with our 'static
        # wins' iron law (R4/R19) as the cleanest counterfactual (like HCCF, R43).
        self.dccf = False
        self.dccf_n_intents = 128
        self.dccf_temp = 1.0
        self.user_intent = None    # [embed_dim, K] learnable user->intent dependency
        self.item_intent = None    # [embed_dim, K] learnable item->intent dependency
        self.dccf_h = None         # [2E] edge source indices (long, on device)
        self.dccf_t = None         # [2E] edge target indices (long, on device)
        self.dccf_indices = None   # [2, 2E] stacked edge indices for torch.sparse_coo_tensor
        self.dccf_n_nodes = 0
        # AdaGCL baseline (KDD'23, arXiv:2305.10837, official repo HKUDS/AdaGCL Model.py).
        # adagcl=False keeps the NoiseSGL backbone (backward compatible); adagcl=True
        # switches to the faithful AdaGCL architecture for the No.1-baseline same-protocol
        # comparison (task #8, the third and last No.1 baseline): TWO trainable contrastive
        # view generators -- (1) a variational graph auto-encoder (VGAE) that learns a new
        # edge-probability view (its OWN embedding tables + encoder mean/std MLPs + an
        # inner-product decoder; an edge is kept iff its predicted probability >= 0.5, and
        # the surviving normalised edge weights are rescaled by the keep ratio), and (2) a
        # graph-denoising net that learns per-layer edge masks via L0 hard-concrete gating
        # (row-normalised attention on the edge endpoints). The InfoNCE contrasts the VGAE
        # view against the denoised view (batch-limited negatives, temp 0.5) plus an
        # information-bottleneck term. This is a learnable-structure baseline, the third
        # clean counterfactual to our 'static wins' iron law (R4/R19), after HCCF (R43) and
        # DCCF (R44). The generator submodules (created in NoiseSGL.__init__ when adagcl=1)
        # are named with an 'adagcl_' prefix so their parameters can be cleanly split into
        # the three official optimizers (main / VGAE / denoising).
        self.adagcl = False
        self.adagcl_temp = 0.5
        self.adagcl_beta = 0.1
        self.adagcl_reg = 1e-5
        self.adagcl_lambda0 = 1e-4
        self.adagcl_gamma = -0.45
        self.adagcl_zeta = 1.05
        self.adagcl_h = None        # [2E] edge source indices (long, on device)
        self.adagcl_t = None        # [2E] edge target indices (long, on device)
        self.adagcl_indices = None  # [2, 2E] stacked edge indices
        self.adagcl_vals = None     # [2E] normalised edge weights
        self.adagcl_n_nodes = 0
        self.adagcl_emb_u = None    # nn.Embedding (VGAE's own user embeddings)
        self.adagcl_emb_i = None    # nn.Embedding (VGAE's own item embeddings)
        self.adagcl_enc_mean = None # VGAE encoder mean MLP
        self.adagcl_enc_std = None  # VGAE encoder std MLP
        self.adagcl_dec = None      # VGAE inner-product decoder MLP
        self.adagcl_nb = None       # nn.ModuleList (per-layer denoising neighbour MLP)
        self.adagcl_self = None     # nn.ModuleList (per-layer denoising self MLP)
        self.adagcl_attn = None     # nn.ModuleList (per-layer denoising attention MLP)
        self._adagcl_edge_weights = []   # accumulated log_alpha per layer (L0 loss)
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
        # Round-35: DirectSpec-style spectral balancing (ACM TORS 2025, arXiv:2406.12032
        # "Balancing Embedding Spectrum for Recommendation"). spec_bal>0 adds a static,
        # parameter-free regulariser that flattens the embedding spectrum (raises the
        # effective rank) by minimising the scale-invariant spectral concentration
        # ||C||_F^2 / trace(C)^2 of the batch Gram C = E^T E (rows unit-normalised inside
        # the loss so trace(C)=B fixes the scale, then scaled by B to match the summed
        # BPR/InfoNCE scale). This directly counters the embedding-collapse that E11
        # identified as the root pathology. It is computed on the *batch* of fused
        # main-branch embeddings (representation layer only, like R8) and does NOT touch
        # the propagation operator (respects R4/R6/R7). Default 0 = off (backward
        # compatible). spec_bal_center=1 removes the per-dimension mean first, which is
        # exactly the 'collapse to a constant vector' component DirectSpec isolates.
        self.spec_bal = 0.0
        self.spec_bal_center = 1
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

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items, eps=0.0, main_adj=None):
        if self.hccf:
            # HCCF baseline (task #8): faithful local-GCN + hypergraph architecture with
            # local-vs-hypergraph per-layer contrast. sub_graph1 is the edge-dropped adj
            # (training) or self.norm_adj (fallback); the hypergraph dropout is handled
            # inside _forward_hccf via self.training.
            adj = sub_graph1 if sub_graph1 is not None else self.norm_adj
            embeds, gnn_lats, hyper_lats = self._forward_hccf(adj)
            user_embeddings, item_embeddings = torch.split(
                embeds, [self.num_users, self.num_items], dim=0)
            user_embs = F.embedding(users, user_embeddings)
            item_embs = F.embedding(items, item_embeddings)
            neg_item_embs = F.embedding(neg_items, item_embeddings)
            sup_pos_ratings = inner_product(user_embs, item_embs)
            sup_neg_ratings = inner_product(user_embs, neg_item_embs)
            sup_logits = sup_pos_ratings - sup_neg_ratings
            infonce = self._hccf_infonce(users, items, gnn_lats, hyper_lats)
            return sup_logits, infonce, None, None, None

        if self.dccf:
            # DCCF baseline (task #8): faithful disentangled-contrast architecture. The
            # augmentation is fully deterministic (learned cosine-affinity adaptive masks),
            # so no edge-dropped sub-graph is used; the full norm_adj drives the gnn view.
            embeds, gnn_lats, int_lats, gaa_lats, iaa_lats = self._forward_dccf()
            user_embeddings, item_embeddings = torch.split(
                embeds, [self.num_users, self.num_items], dim=0)
            user_embs = F.embedding(users, user_embeddings)
            item_embs = F.embedding(items, item_embeddings)
            neg_item_embs = F.embedding(neg_items, item_embeddings)
            sup_pos_ratings = inner_product(user_embs, item_embs)
            sup_neg_ratings = inner_product(user_embs, neg_item_embs)
            sup_logits = sup_pos_ratings - sup_neg_ratings
            infonce = self._dccf_infonce(users, items, gnn_lats, int_lats, gaa_lats, iaa_lats)
            return sup_logits, infonce, None, None, None

        user_embeddings, item_embeddings = self._forward_gcn(
            main_adj if main_adj is not None else self.norm_adj,
            use_freq=self.freq_on_main, apply_proto=True, apply_dropout=True)

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

        spec_loss = None
        if self.spec_bal > 0:
            # Round-35: DirectSpec-style spectral balancing on the batch of fused main
            # embeddings (representation layer only). user_embs/item_embs are the batch
            # rows of the final fused embeddings, so this is orthogonal to the propagation
            # operator (R4/R6/R7) and to the three fusion components.
            spec_loss = self._spec_bal_loss(user_embs, item_embs)

        return sup_logits, infonce, fcl_loss, dau_loss, spec_loss

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

    def _spec_bal_loss(self, user_embs, item_embs):
        """DirectSpec-style spectral balancing (Round-35, 4th-component candidate #10).

        Flattens the singular-value spectrum of the batch embedding matrix by minimising
        the scale-invariant spectral concentration ratio  ||C||_F^2 / trace(C)^2  where
        C = E^T E is the [d, d] Gram of the batch embeddings. For a perfectly flat
        spectrum (orthogonal, equal-variance dimensions) the ratio is 1/d (minimum); for
        a rank-1 collapse it is 1 (maximum). Minimising it therefore raises the effective
        rank / spectral entropy -- the exact quantity E11 showed is the mechanism behind
        the noise augmentation's benefit (41% -> 97% effective rank).

        To keep the gradient well-scaled WITHOUT imposing an arbitrary variance target
        (which would collide with the BPR inner-product scale and the R23 'norm carries
        popularity signal' finding), the rows are unit-normalised *inside this loss only*
        (so trace(C)=B is fixed and the objective becomes a pure spectral-shape term on
        the unit sphere) and the scalar ratio is multiplied by the batch size B so the
        term is on the same O(B) scale as the summed BPR/InfoNCE losses. Cheap closed
        form: no SVD / eigendecomposition is needed.
        """
        def concentration(E):
            if self.spec_bal_center:
                E = E - E.mean(dim=0, keepdim=True)
            E = F.normalize(E, dim=1)          # unit-norm rows: trace(C) = B (fixed scale)
            B = float(E.size(0))
            C = E.t() @ E                       # [d, d]
            tr = torch.trace(C)                 # == B up to fp error
            fro2 = torch.sum(C * C)             # sum of fourth powers of singular values
            return B * fro2 / (tr * tr + 1e-12)  # in [B/d, B], O(B) scale
        return 0.5 * (concentration(user_embs) + concentration(item_embs))

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

    def _forward_hccf(self, adj):
        """Faithful HCCF forward pass (official akaxlh/HCCF torchVersion/Model.py).

        Local GCN view:   tem = D^{-1/2} A' D^{-1/2} . lat_{k-1}   (A' = edge-dropped adj
        in training, full adj at eval).
        Hypergraph view:  uu_hyper = E_u @ u_hyper  ([I,H]);  hyper_u = uu_hyper @ (uu_hyper^T @ u_lat).
        Per layer:        lat_k = tem + [hyper_u ; hyper_i].
        Final:            embeds = sum(lats)  (sum over the L+1 layer embeddings).
        Returns (embeds, gnn_lats, hyper_lats) with gnn_lats/hyper_lats of length n_layers.
        """
        embeds = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        lats = [embeds]
        gnn_lats = []
        hyper_lats = []
        # learnable hyperedge-dependency matrices. The official code applies a FRESH
        # dropout mask to uu_hyper/ii_hyper at EVERY layer (inside the loop), so we do the
        # same here; nn.functional.dropout is a no-op when self.training is False -> eval
        # stays clean (keep_rate=1.0 effectively).
        uu_hyper_raw = self.user_embeddings.weight @ self.u_hyper   # [I, H]
        ii_hyper_raw = self.item_embeddings.weight @ self.i_hyper   # [J, H]
        for _k in range(self.n_layers):
            tem_embeds = torch_sp.mm(adj, lats[-1])
            uu_hyper = F.dropout(uu_hyper_raw, p=1.0 - self.hccf_keep_rate, training=self.training)
            ii_hyper = F.dropout(ii_hyper_raw, p=1.0 - self.hccf_keep_rate, training=self.training)
            u_lat = lats[-1][:self.num_users]
            i_lat = lats[-1][self.num_users:]
            hyper_u = uu_hyper @ (uu_hyper.t() @ u_lat)   # [I, H] @ [H, I] @ [I, d]
            hyper_i = ii_hyper @ (ii_hyper.t() @ i_lat)   # [J, H] @ [H, J] @ [J, d]
            gnn_lats.append(tem_embeds)
            hyper_lats.append(torch.cat([hyper_u, hyper_i], dim=0))
            lats.append(tem_embeds + hyper_lats[-1])
        embeds = sum(lats)
        return embeds, gnn_lats, hyper_lats

    def _hccf_infonce(self, users, items, gnn_lats, hyper_lats):
        """Faithful HCCF contrastive loss (official Utils/Utils.contrastLoss).

        For each GNN layer: contrast the *detached* local GCN view (stop-gradient, the
        official code does embeds1.detach()) against the hypergraph view, over the
        unique users / unique positive items in the batch (batch-limited negatives, NOT
        the full-graph logsumexp used by our NoiseSGL InfoNCE). Mean-scaled per node
        (the official .mean()), summed over user + item side and over layers.
        """
        loss = 0.0
        uniq_u = torch.unique(users)
        uniq_i = torch.unique(items)
        for k in range(len(gnn_lats)):
            embeds1 = gnn_lats[k].detach()      # local view: stop-gradient (official)
            embeds2 = hyper_lats[k]             # hypergraph view
            u1 = F.normalize(embeds1[:self.num_users] + 1e-8, dim=1)
            u2 = F.normalize(embeds2[:self.num_users] + 1e-8, dim=1)
            i1 = F.normalize(embeds1[self.num_users:] + 1e-8, dim=1)
            i2 = F.normalize(embeds2[self.num_users:] + 1e-8, dim=1)
            # user side
            p1, p2 = u1[uniq_u], u2[uniq_u]
            nume = torch.exp(torch.sum(p1 * p2, dim=-1) / self.hccf_temp)
            deno = torch.exp(p1 @ p2.t() / self.hccf_temp).sum(-1) + 1e-8
            loss = loss + (-torch.log(nume / deno)).mean()
            # item side
            p1, p2 = i1[uniq_i], i2[uniq_i]
            nume = torch.exp(torch.sum(p1 * p2, dim=-1) / self.hccf_temp)
            deno = torch.exp(p1 @ p2.t() / self.hccf_temp).sum(-1) + 1e-8
            loss = loss + (-torch.log(nume / deno)).mean()
        return loss

    def _adaptive_mask(self, matrix):
        """DCCF adaptive subgraph: edge weight = cosine affinity mapped to [0,1]
        (official: (sum(head*tail)+1)/2 on L2-normalised embeddings), then row-normalised
        (D^{-1} A) so each node aggregates a convex combination of its neighbours.
        head/tail always come from the SAME matrix (official _adaptive_mask is called with
        gnn_head/gnn_tail both gathered from gnn_layer, and int_head/int_tail both from
        int_layer), so we normalise the full [n,d] matrix once and compute the per-edge dot
        products in CHUNKS over edges. The edge weights are computed under torch.no_grad():
        materialising an autograd graph through the O(E*d) adaptive-mask cosine would need
        tens of GB at embed_size=256 (the official code runs at 32 dims where it is cheap),
        and this is the only gradient path we sacrifice for a capacity-matched baseline --
        the message values (embeddings[t]) still receive gradient via the sparse mm.
        Returns detached [2E] message-passing weights consumed by _masked_spmm."""
        m = F.normalize(matrix, dim=1)
        E = int(self.dccf_h.shape[0])
        edge_alpha = torch.empty(E, device=matrix.device)
        chunk = 1 << 20   # ~1M edges per chunk: peak ~ chunk*d*4 bytes (~1 GiB at d=256)
        with torch.no_grad():
            for s in range(0, E, chunk):
                e = min(s + chunk, E)
                h = self.dccf_h[s:e]
                t = self.dccf_t[s:e]
                edge_alpha[s:e] = (torch.sum(m[h] * m[t], dim=1) + 1.0) / 2.0
            d_scores = torch.zeros(self.dccf_n_nodes, device=matrix.device)
            d_scores.index_add_(0, self.dccf_h, edge_alpha)          # row degree of weighted adj
            d_inv = 1.0 / (d_scores + 1e-12)
            d_inv = torch.where(torch.isfinite(d_inv), d_inv, torch.zeros_like(d_inv))
            weights = d_inv[self.dccf_h] * edge_alpha
        return weights

    def _masked_spmm(self, values, embeddings):
        """Sparse message passing with the adaptive-mask weights (row-normalised):
        out = A_adaptive @ embeddings. Implemented with torch.sparse.mm so the autograd
        backward is O(nnz*d) without materialising the [2E, d] gather (the index_add
        version would keep a [2E, d] autograd graph and OOM at embed_size=256)."""
        adj = torch.sparse_coo_tensor(
            self.dccf_indices, values, (self.dccf_n_nodes, self.dccf_n_nodes))
        return torch.sparse.mm(adj, embeddings)

    def _forward_dccf(self):
        """Faithful DCCF forward pass (official HKUDS/DCCF model.py DCCF.inference).

        Per layer:  gnn = G @ state  (plain D^{-1/2} A D^{-1/2} propagation);
                    int = softmax(e @ intent) @ intent^T  (intent-aware reconstruction);
                    gaa = adaptive_mask(gnn) @ state  (graph-adaptive augmentation);
                    iaa = adaptive_mask(int) @ state  (intent-adaptive augmentation);
                    state <- gnn + int + gaa + iaa + prev (residual).
        Final embedding = sum over the L+1 layer states. Returns (embeds, gnn_lats,
        int_lats, gaa_lats, iaa_lats) with per-layer views of length n_layers.
        """
        embeds = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [embeds]
        gnn_lats, int_lats, gaa_lats, iaa_lats = [], [], [], []
        for _ in range(self.n_layers):
            gnn_layer = torch_sp.mm(self.norm_adj, all_embeddings[-1])
            u_lat, i_lat = torch.split(all_embeddings[-1], [self.num_users, self.num_items], dim=0)
            u_int = torch.softmax(u_lat @ self.user_intent, dim=1) @ self.user_intent.t()
            i_int = torch.softmax(i_lat @ self.item_intent, dim=1) @ self.item_intent.t()
            int_layer = torch.cat([u_int, i_int], dim=0)
            gaa = self._masked_spmm(
                self._adaptive_mask(gnn_layer), all_embeddings[-1])
            iaa = self._masked_spmm(
                self._adaptive_mask(int_layer), all_embeddings[-1])
            gnn_lats.append(gnn_layer)
            int_lats.append(int_layer)
            gaa_lats.append(gaa)
            iaa_lats.append(iaa)
            all_embeddings.append(gnn_layer + int_layer + gaa + iaa + all_embeddings[-1])
        embeds = torch.stack(all_embeddings, dim=1).sum(dim=1)
        return embeds, gnn_lats, int_lats, gaa_lats, iaa_lats

    def _dccf_infonce(self, users, items, gnn_lats, int_lats, gaa_lats, iaa_lats):
        """Faithful DCCF contrastive loss (official DCCF.cal_ssl_loss).

        For each layer, contrast the gnn view (anchor) against the int / gaa / iaa views
        (user- and item-side), over the unique users / unique positive items in the batch
        (batch-limited negatives, NOT the full-graph logsumexp used by our NoiseSGL
        InfoNCE). All views are L2-normalised before contrasting (official F.normalize);
        the positive is the same node in the two views, negatives are the other batch
        nodes. Mean-scaled per node, summed over 6 pairs x L layers. No detach (official).
        """
        uniq_u = torch.unique(users)
        uniq_i = torch.unique(items)

        def cal_loss(e1, e2):
            pos = torch.exp(torch.sum(e1 * e2, dim=1) / self.dccf_temp)
            neg = torch.exp(e1 @ e2.t() / self.dccf_temp).sum(1)
            return (-torch.log(pos / (neg + 1e-8) + 1e-8)).mean()

        loss = 0.0
        for k in range(len(gnn_lats)):
            u_gnn, i_gnn = torch.split(gnn_lats[k], [self.num_users, self.num_items], dim=0)
            u_int, i_int = torch.split(int_lats[k], [self.num_users, self.num_items], dim=0)
            u_gaa, i_gaa = torch.split(gaa_lats[k], [self.num_users, self.num_items], dim=0)
            u_iaa, i_iaa = torch.split(iaa_lats[k], [self.num_users, self.num_items], dim=0)
            u_gnn = F.normalize(u_gnn[uniq_u], dim=1)
            u_int = F.normalize(u_int[uniq_u], dim=1)
            u_gaa = F.normalize(u_gaa[uniq_u], dim=1)
            u_iaa = F.normalize(u_iaa[uniq_u], dim=1)
            i_gnn = F.normalize(i_gnn[uniq_i], dim=1)
            i_int = F.normalize(i_int[uniq_i], dim=1)
            i_gaa = F.normalize(i_gaa[uniq_i], dim=1)
            i_iaa = F.normalize(i_iaa[uniq_i], dim=1)
            loss += cal_loss(u_gnn, u_int) + cal_loss(u_gnn, u_gaa) + cal_loss(u_gnn, u_iaa)
            loss += cal_loss(i_gnn, i_int) + cal_loss(i_gnn, i_gaa) + cal_loss(i_gnn, i_iaa)
        return loss

    # ------------------------------------------------------------------ #
    # AdaGCL baseline (task #8, third No.1 baseline). Faithful port of the
    # official HKUDS/AdaGCL Model.py. All the trainable view-generator logic
    # lives here (VGAE + denoising net); the optimiser orchestration lives in
    # NoiseSGL._adagcl_batch_loop.
    # ------------------------------------------------------------------ #
    def _adagcl_main_embeds(self):
        return torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

    def _adagcl_gcn_sum(self, adj, init_embeds):
        """AdaGCL backbone: SUM over the L+1 layer embeddings (official forward_graphcl /
        forward_gcn both do `sum(embedsLst)`, NOT the LightGCN mean-pooling)."""
        embeds_lst = [init_embeds]
        for _ in range(self.n_layers):
            embeds_lst.append(torch_sp.mm(adj, embeds_lst[-1]))
        return sum(embeds_lst)

    def _adagcl_vgae_encode(self):
        """VGAE encoder: LightGCN-sum over the VGAE's OWN embeddings, then reparameterise
        with the mean/std MLPs (official vgae_encoder.forward)."""
        x = self._adagcl_gcn_sum(
            self.norm_adj, torch.cat([self.adagcl_emb_u.weight, self.adagcl_emb_i.weight], dim=0))
        x_mean = self.adagcl_enc_mean(x)
        x_std = self.adagcl_enc_std(x)
        gaussian_noise = torch.randn_like(x_mean)
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std

    def _adagcl_vgae_generate(self):
        """VGAE contrastive view (official vgae.generate, called under torch.no_grad by the
        caller): predict an edge-probability for every (directed) edge of the full normalised
        adjacency, keep the edges with p>=0.5, and rescale the surviving normalised edge
        weights by E/kept so the total mass is preserved. Chunked over edges because the
        [2E, d] endpoint-product materialisation would OOM at embed_size=256 (the official
        runs at latdim=32 where it is cheap)."""
        x, _, _ = self._adagcl_vgae_encode()
        E = int(self.adagcl_h.shape[0])
        probs = torch.empty(E, device=x.device)
        chunk = 1 << 20
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            h = self.adagcl_h[s:e]
            t = self.adagcl_t[s:e]
            probs[s:e] = torch.sigmoid(self.adagcl_dec(x[h] * x[t])).squeeze(-1)
        mask = ((probs + 0.5).floor()).bool()
        kept = int(mask.sum().item())
        if kept == 0:
            return self.norm_adj
        new_vals = self.adagcl_vals[mask] / (float(kept) / float(E))
        new_idx = self.adagcl_indices[:, mask]
        return torch.sparse_coo_tensor(
            new_idx, new_vals, (self.adagcl_n_nodes, self.adagcl_n_nodes)).coalesce()

    def _adagcl_denoise_log_alpha(self, x, layer):
        """Chunked per-edge attention logits of the L0 gate (official DenoisingNet.
        get_attention): logit_e = attn_k([nb_k(x_h) ; self_k(x_t)])."""
        E = int(self.adagcl_h.shape[0])
        log_alpha = torch.empty(E, device=x.device)
        nb = self.adagcl_nb[layer]
        sl = self.adagcl_self[layer]
        attn = self.adagcl_attn[layer]
        chunk = 1 << 20
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            f1 = nb(x[self.adagcl_h[s:e]])
            f2 = sl(x[self.adagcl_t[s:e]])
            log_alpha[s:e] = attn(torch.cat([f1, f2], dim=1)).squeeze(-1)
        return log_alpha

    def _adagcl_hard_concrete(self, log_alpha, temperature, training):
        """L0 hard-concrete gate (official hard_concrete_sample with beta=1.0): training
        uses the Gumbel-style stochastic relaxation, eval uses the deterministic
        sigmoid(log_alpha)."""
        if training:
            u = torch.rand_like(log_alpha).clamp_(1e-7, 1.0 - 1e-7)
            gate = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + log_alpha) / 1.0)
        else:
            gate = torch.sigmoid(log_alpha)
        stretched = gate * (self.adagcl_zeta - self.adagcl_gamma) + self.adagcl_gamma
        return torch.clamp(stretched, 0.0, 1.0)

    def _adagcl_denoise_support(self, x, layer, temperature, training):
        """Per-layer denoised adjacency: mask = hard-concrete(attention(x_h, x_t)), then
        degree-renormalise D^{-1/2} (masked A) D^{-1/2} (official DenoisingNet.generate /
        call). Returns (support, log_alpha) so the caller can accumulate log_alpha for the
        L0 sparsity loss."""
        log_alpha = self._adagcl_denoise_log_alpha(x, layer)
        mask = self._adagcl_hard_concrete(log_alpha, temperature, training)
        d_scores = torch.zeros(self.adagcl_n_nodes, device=x.device)
        d_scores.index_add_(0, self.adagcl_h, mask)
        d_inv_sqrt = torch.clamp(torch.pow(d_scores + 1e-12, -0.5), 0.0, 10.0)
        vals = mask * d_inv_sqrt[self.adagcl_h] * d_inv_sqrt[self.adagcl_t]
        support = torch.sparse_coo_tensor(
            self.adagcl_indices, vals, (self.adagcl_n_nodes, self.adagcl_n_nodes)).coalesce()
        return support, log_alpha

    def _adagcl_denoise_propagate(self, x, mask):
        """Message passing for a grad-requiring denoised mask WITHOUT torch.sparse.mm:
        out[i] = sum_{e: h_e=i} mask_e * d_inv[h_e] * d_inv[t_e] * x[t_e]. Chunked over
        edges. This explicit gather + index_add formulation is REQUIRED because
        torch.sparse.mm's backward w.r.t. a grad-requiring sparse VALUES densifies to
        O(n^2) (it materialises grad @ x^T); the explicit message keeps the autograd
        graph O(E*d) (the official AdaGCL uses torch-sparse's spmm, which has the same
        efficient sparse-value backward)."""
        n = self.adagcl_n_nodes
        d_scores = torch.zeros(n, device=x.device)
        d_scores.index_add_(0, self.adagcl_h, mask)
        d_inv_sqrt = torch.clamp(torch.pow(d_scores + 1e-12, -0.5), 0.0, 10.0)
        w = mask * d_inv_sqrt[self.adagcl_h] * d_inv_sqrt[self.adagcl_t]   # [E]
        out = torch.zeros(n, x.shape[1], device=x.device)
        E = int(self.adagcl_h.shape[0])
        chunk = 1 << 20
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            out.index_add_(0, self.adagcl_h[s:e], w[s:e].unsqueeze(1) * x[self.adagcl_t[s:e]])
        return out

    def _adagcl_denoise_view(self):
        """Denoised contrastive view (official Model.forward_graphcl_): the MAIN embeddings
        propagated through a per-layer denoised adjacency; each layer's mask is computed
        under torch.no_grad() with training=False (deterministic), so no gradient flows to
        the denoising net from the contrastive view (it is trained only by its own loss)."""
        embeds = self._adagcl_main_embeds()
        embeds_lst = [embeds]
        for k in range(self.n_layers):
            with torch.no_grad():
                support, _ = self._adagcl_denoise_support(embeds_lst[-1], k, 1.0, training=False)
            embeds_lst.append(torch_sp.mm(support, embeds_lst[-1]))
        return sum(embeds_lst)

    def _adagcl_infonce(self, x1, x2, users, items):
        """Official AdaGCL contrastive loss (Model.loss_graphcl): L2-normalise both views,
        gather the batch users + positive items from each view into a combined [2B, d] set,
        and compute a single InfoNCE whose positives are the diagonal (same node across the
        two views) and whose negatives are all OTHER batch nodes (users AND items). Returns
        a [2B] vector; the caller applies .mean() (official)."""
        T = self.adagcl_temp
        u1, i1 = torch.split(x1, [self.num_users, self.num_items], dim=0)
        u2, i2 = torch.split(x2, [self.num_users, self.num_items], dim=0)
        u1 = F.normalize(u1, dim=1); i1 = F.normalize(i1, dim=1)
        u2 = F.normalize(u2, dim=1); i2 = F.normalize(i2, dim=1)
        all1 = torch.cat([u1[users], i1[items]], dim=0)
        all2 = torch.cat([u2[users], i2[items]], dim=0)
        sim = torch.exp(all1 @ all2.t() / T)
        pos = torch.diag(sim)
        loss = -torch.log(pos / (sim.sum(dim=1) - pos + 1e-8) + 1e-8)
        return loss

    def _adagcl_vgae_loss(self, users, items, neg_items):
        """VGAE training loss (official vgae_decoder.forward): per-edge reconstruction BCE
        (positive edges -> 1, negative edges -> 0) + beta*KL + a BPR on the VGAE's own
        embeddings + reg over the VGAE encoder (embeddings + mean/std MLPs)."""
        x, x_mean, x_std = self._adagcl_vgae_encode()
        x_u, x_i = torch.split(x, [self.num_users, self.num_items], dim=0)
        pos_pred = torch.sigmoid(self.adagcl_dec(x_u[users] * x_i[items])).squeeze(-1)
        neg_pred = torch.sigmoid(self.adagcl_dec(x_u[users] * x_i[neg_items])).squeeze(-1)
        loss_rec = -torch.log(pos_pred + 1e-8) - torch.log(1.0 - neg_pred + 1e-8)
        kl = -0.5 * (1.0 + 2.0 * torch.log(x_std + 1e-8) - x_mean.pow(2) - x_std.pow(2)).sum(dim=1)
        anc = x_u[users]; pos = x_i[items]; neg = x_i[neg_items]
        score_diff = (anc * pos).sum(-1) - (anc * neg).sum(-1)
        bpr = -torch.log(torch.sigmoid(score_diff) + 1e-8).sum() / float(users.shape[0])
        reg = self.adagcl_emb_u.weight.norm(2).square() + self.adagcl_emb_i.weight.norm(2).square()
        for p in self.adagcl_enc_mean.parameters():
            reg = reg + p.norm(2).square()
        for p in self.adagcl_enc_std.parameters():
            reg = reg + p.norm(2).square()
        loss = (loss_rec + self.adagcl_beta * kl.mean() + bpr + self.adagcl_reg * reg).mean()
        return loss

    def _adagcl_denoise_loss(self, users, items, neg_items, temperature):
        """Denoising-net training loss (official DenoisingNet.forward): BPR over the
        denoised-view embeddings (main embeddings DETACHED, per-layer masks sampled with
        training=True) + reg over the attention MLPs + lambda0 * L0 sparsity loss."""
        x = self._adagcl_main_embeds().detach()
        embeds_lst = [x]
        self._adagcl_edge_weights = []
        for k in range(self.n_layers):
            log_alpha = self._adagcl_denoise_log_alpha(embeds_lst[-1], k)
            self._adagcl_edge_weights.append(log_alpha)
            mask = self._adagcl_hard_concrete(log_alpha, temperature, training=True)
            embeds_lst.append(self._adagcl_denoise_propagate(embeds_lst[-1], mask))
        x = sum(embeds_lst)
        x_u, x_i = torch.split(x, [self.num_users, self.num_items], dim=0)
        anc = x_u[users]; pos = x_i[items]; neg = x_i[neg_items]
        score_diff = (anc * pos).sum(-1) - (anc * neg).sum(-1)
        bpr = -torch.log(torch.sigmoid(score_diff) + 1e-8).sum() / float(users.shape[0])
        reg = 0.0
        for mlist in (self.adagcl_nb, self.adagcl_self, self.adagcl_attn):
            for p in mlist.parameters():
                reg = reg + p.norm(2).square()
        l0 = 0.0
        for log_alpha in self._adagcl_edge_weights:
            l0 = l0 + torch.sigmoid(
                log_alpha - temperature * math.log(-self.adagcl_gamma / self.adagcl_zeta)).mean()
        loss = bpr + self.adagcl_reg * reg + self.adagcl_lambda0 * l0
        return loss

    def _forward_gcn(self, norm_adj, return_all=False, perturb=False, eps=0.0, use_freq=None,
                     use_gl=False, apply_proto=False, apply_dropout=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        if self.item_sem is not None and self.sem_lam != 0.0:
            # Round-47: inject the FROZEN LLM semantic prior into the layer-0 item
            # embedding (users get no semantic offset). item_sem is non-Parameter and
            # moved to the embedding device on load, so it never receives gradients and
            # never enters the optimizer. Applied before dropout/SSL-noise/propagation so
            # main branch and noise views share the identical operator (R4/R6/R7).
            sem = self.item_sem.to(ego_embeddings.device)
            ego_embeddings = torch.cat([
                ego_embeddings[:self.num_users],
                ego_embeddings[self.num_users:] + self.sem_lam * sem,
            ], dim=0)
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
        # Round-34: layer-0 snapshot for the static initial-residual connection.
        initial_embeddings = ego_embeddings
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
            if self.res_alpha > 0.0:
                # Round-34: GCNII/APPNP static initial residual (see _LightGCN.__init__).
                # Applied BEFORE the SSL noise so the residual is part of the clean
                # propagation operator and the noise stays the only SSL-specific
                # perturbation (main branch and noise views share the identical operator).
                ego_embeddings = (1.0 - self.res_alpha) * ego_embeddings + self.res_alpha * initial_embeddings
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
            if self.agg_weights is not None:
                _w = self.agg_weights.to(ego_embeddings.device).view(1, -1, 1)
                all_embeddings = (torch.stack(all_embeddings, dim=1) * _w).sum(dim=1)
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
        if self.hccf:
            # HCCF: full (non-dropped) adjacency, no hypergraph dropout (training=False).
            embeds, _, _ = self._forward_hccf(self.norm_adj)
            self._user_embeddings_final, self._item_embeddings_final = torch.split(
                embeds, [self.num_users, self.num_items], dim=0)
            return
        if self.dccf:
            # DCCF: deterministic (no dropout), full norm_adj drives all four views.
            embeds, _, _, _, _ = self._forward_dccf()
            self._user_embeddings_final, self._item_embeddings_final = torch.split(
                embeds, [self.num_users, self.num_items], dim=0)
            return
        if self.adagcl:
            # AdaGCL: sum-over-layers embeddings of the MAIN model on the full adjacency
            # (official forward_gcn). The generators are NOT used at eval time.
            embeds = self._adagcl_gcn_sum(self.norm_adj, self._adagcl_main_embeds())
            self._user_embeddings_final, self._item_embeddings_final = torch.split(
                embeds, [self.num_users, self.num_items], dim=0)
            return
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
        # Round-38: Adam decoupled weight decay (regularization-axis 4th-component
        # candidate #12). weight_decay=0 keeps the default Adam (backward compatible);
        # >0 shrinks the FULL embedding table every step (decoupled from the batch-L2
        # reg_loss which is controlled by self.reg). Static, no learnable parameters
        # (respects the R1/R15/R18 'static wins' law), orthogonal to the 3 fusion
        # components (noise / SSL anneal / capacity) and to the 6 queued candidates.
        # Motivation: R12c/R15/R18 show a mild deterministic overfit in the pure-BPR
        # fine-tune phase (after ssl_anneal_epochs=30 the SSL weight is 0); stronger L2
        # regularisation counters it. Literature: 'Revisiting BPR: A Replicability Study'
        # (arXiv:2409.14217) -- proper regularisation is the main gap between BPR and
        # SOTA; 'On the Overlooked Pitfalls of Weight Decay' (arXiv:2011.11152).
        self.weight_decay = float(config["weight_decay"]) if "weight_decay" in config else 0.0
        self.param_init = config["param_init"]
        # Round-37: learning-rate schedule (4th-component candidate #11). lr_anneal=0
        # keeps the fixed lr (backward compatible); 1 = linear, 2 = cosine decay from
        # self.lr down to self.lr*self.lr_floor over lr_anneal_epochs epochs (0 => full
        # training budget) and then holds. It is an OPTIMIZER-level schedule, orthogonal
        # to the SSL-weight schedule (R13/R15, which scales the InfoNCE magnitude) and to
        # the augmentation/embedding axes. Motivation: R12c/R15/R23 show the BPR
        # fine-tune phase (after the SSL weight hits floor at ep~30) peaks then overfits
        # (gowalla embed256 peaks ~ep61 then declines); a decaying lr shrinks the
        # fine-tune step size to counter that late-stage overfitting. Static, no learnable
        # parameters (respects the R1/R15/R18 'static wins' law).
        # Literature: 'Optimal Linear Decay Learning Rate Schedules and Further
        # Refinements' (arXiv:2310.07831) + cosine schedule (Loshchilov & Hutter 2017).
        self.lr_anneal = int(config["lr_anneal"]) if "lr_anneal" in config else 0
        assert self.lr_anneal in (0, 1, 2)
        self.lr_anneal_epochs = int(config["lr_anneal_epochs"]) if "lr_anneal_epochs" in config else 0
        self.lr_floor = float(config["lr_floor"]) if "lr_floor" in config else 0.1
        assert 0.0 < self.lr_floor <= 1.0
        # Round-40: stochastic weight averaging (SWA) of the embedding tables as the
        # 4th-component candidate #14 (CV cross-domain: SWA UAI'18 arXiv:1803.05407;
        # also the EMA/momentum-encoder trick of BYOL arXiv:2006.07733 / MoCo
        # arXiv:1911.05722 / Mean Teacher arXiv:1703.01780, transplanted to CF).
        # swa_start=0 keeps the standard single-best-snapshot protocol (backward
        # compatible). swa_start>0 accumulates an EQUAL-weight average of the embedding
        # weights over epochs [swa_start, E] and evaluates the ranking with that average
        # (not the last/best single snapshot). It directly targets the one remaining
        # documented weakness -- the pure-BPR fine-tune phase (after ssl_anneal_epochs=30
        # the SSL weight is 0) peaks (gowalla embed256 ~ep61) and then deterministically
        # overfits (R12c/R15/R23) -- by flattening the late-stage loss surface.
        # Static, parameter-free (respects the R1/R15/R18 'static wins' law); orthogonal
        # to noise (R1) / SSL anneal (R15) / capacity (R18) and to all queued candidates.
        self.swa_start = int(config["swa_start"]) if "swa_start" in config else 0
        assert self.swa_start >= 0
        self.swa_n = 0
        self.swa_user = None      # running sum of user embedding weights
        self.swa_item = None      # running sum of item embedding weights
        self._swa_bk_user = None  # persistent backup for the swap-in/out
        self._swa_bk_item = None

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
        # Round-41: BPR ranking margin (4th-component candidate #15). bpr_margin=0 keeps
        # the plain BPR (backward compatible). margin>0 subtracts gamma from the BPR
        # logits: loss = -log sigmoid(pos - neg - gamma), which requires a STRONGER
        # separation between the positive and the negative item and sharpens the ranking
        # signal in the BPR fine-tune phase (post SSL-anneal). Static, parameter-free,
        # orthogonal to the three fusion components (noise / SSL-anneal / capacity) and
        # to every previously queued candidate: it modifies the BPR margin directly rather
        # than adding an auxiliary loss (unlike R8/R9/R21/R35).
        self.bpr_margin = float(config["bpr_margin"]) if "bpr_margin" in config else 0.0
        assert self.bpr_margin >= 0.0
        # Round-47: FROZEN LLM semantic item feature injection (task #6, 4th/5th-
        # component semantic axis). sem_emb=1 loads dataset/<ds>/item_sem.npy (built by
        # util/build_semantic_features.py) and adds sem_lam * item_sem to the layer-0
        # item embedding (static, parameter-free; orthogonal to noise/SSL-anneal/capacity
        # and to every structural/optimizer candidate). sem_emb=0 = off (backward
        # compatible). sem_lam controls the prior strength (1.0 = unit-norm semantic
        # vector added at full strength). Literature: LLMRec arXiv:2311.00423 (LLM item-
        # attribute enhancement) / TIGER arXiv:2305.14395 (title semantics for cold-start).
        self.sem_emb = int(config["sem_emb"]) if "sem_emb" in config else 0
        assert self.sem_emb in (0, 1)
        self.sem_lam = float(config["sem_lam"]) if "sem_lam" in config else 1.0
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
        # Round-35: DirectSpec-style spectral balancing coefficient (4th-component
        # candidate #10). 0 = off (backward compatible).
        self.spec_bal = float(config["spec_bal"]) if "spec_bal" in config else 0.0
        self.spec_bal_center = int(config["spec_bal_center"]) if "spec_bal_center" in config else 1
        assert self.spec_bal_center in (0, 1)
        # Round-23: L2-normalised (cosine) scoring (4th-component candidate). Static,
        # parameter-free; normalises the final fused embeddings before BPR/eval scoring.
        self.l2_norm = int(config["l2_norm"]) if "l2_norm" in config else 0
        assert self.l2_norm in (0, 1)
        # Round-25: main-branch embedding (input/node) dropout rate. Static, parameter-
        # free regulariser for the BPR fine-tune phase (the R12c/R15 peak-then-overfit
        # problem). 0 = off (backward compatible).
        self.dropout = float(config["dropout"]) if "dropout" in config else 0.0
        assert 0.0 <= self.dropout < 1.0
        # Round-49: DropEdge-style edge dropout on the MAIN propagation branch
        # (4th-component candidate #16). dropedge_keep=1.0 keeps the full adjacency
        # (backward compatible); <1.0 drops (1-keep) of the bipartite training edges
        # once per epoch (re-normalised) for the main/prediction branch only -- the
        # contrastive noise views keep the full adjacency + embedding noise, so the SSL
        # signal is untouched (respects R4/R6/R7). eval() still uses the full norm_adj
        # (standard DropEdge train/test protocol). Static, parameter-free, orthogonal to
        # the 3 fusion components. Mechanism: the main branch currently has NO direct
        # anti-over-smoothing (the noise only regularises the InfoNCE views), and E11
        # shows residual over-smoothing (cos(l0,lL)=0.46); DropEdge breaks connectivity
        # during propagation and is the canonical static remedy (Rong et al. ICLR 2020).
        self.dropedge_keep = float(config["dropedge_keep"]) if "dropedge_keep" in config else 1.0
        assert 0.0 < self.dropedge_keep <= 1.0
        # Round-34: GCNII/APPNP-style static initial-residual strength (4th-component
        # candidate #9). 0 = plain LightGCN propagation (backward compatible). Static and
        # parameter-free; targets the depth-accumulated over-smoothing that E11 identified,
        # complementing (not replacing) the noise augmentation's rank-release effect.
        # Respects R4/R6/R7: applied identically on the main branch and the noise views.
        self.res_alpha = float(config["res_alpha"]) if "res_alpha" in config else 0.0
        assert 0.0 <= self.res_alpha < 1.0
        # Round-39: static non-uniform layer-aggregation weights (4th-component
        # candidate #13). agg_weights="uniform" keeps the LightGCN mean-pooling over the
        # L+1 layer embeddings (backward compatible). Named schemes set a static,
        # parameter-free weighting of the final layer pool (shared by main branch, noise
        # views and eval, so the aggregation operator stays identical, cf. R4/R6/R7):
        #   "shallow" favours layer-0 (w_k propto L+1-k), "deep" favours the last layer
        #   (w_k propto k+1), "geom" uses w_k propto agg_alpha^k, "first"/"last" keep a
        #   single layer, and an explicit comma list ("0.4,0.3,0.2,0.1") is also accepted.
        # Weights are normalised to sum to 1 so "uniform" matches mean-pooling exactly.
        # Motivation: E11 shows over-smoothing accumulates with depth (cos(l0,lL)=0.46
        # even with noise), so the deeper layers carry a partially-collapsed signal; a
        # static shallow-favoured pool is a direct anti-over-smoothing measure that
        # complements (not replaces) the noise augmentation's rank-release. Literature:
        # APPNP teleport (arXiv:1810.05997) / GCNII initial residual (arXiv:2007.02133)
        # / SGC layer-collapse analysis.
        self.agg_weights_str = str(config["agg_weights"]) if "agg_weights" in config else "uniform"
        self.agg_alpha = float(config["agg_alpha"]) if "agg_alpha" in config else 0.5
        _aw = None
        if self.agg_weights_str == "uniform":
            _aw = None
        elif self.agg_weights_str == "shallow":
            _aw = [float(self.n_layers + 1 - k) for k in range(self.n_layers + 1)]
        elif self.agg_weights_str == "deep":
            _aw = [float(k + 1) for k in range(self.n_layers + 1)]
        elif self.agg_weights_str == "geom":
            _aw = [self.agg_alpha ** k for k in range(self.n_layers + 1)]
        elif self.agg_weights_str == "first":
            _aw = [1.0] + [0.0] * self.n_layers
        elif self.agg_weights_str == "last":
            _aw = [0.0] * self.n_layers + [1.0]
        else:
            _raw = self.agg_weights_str.replace("(", "").replace(")", "").replace(" ", "")
            _aw = [float(x) for x in _raw.split(",")]
            assert len(_aw) == self.n_layers + 1, \
                "agg_weights explicit list must have L+1=%d entries, got %r" % (
                    self.n_layers + 1, self.agg_weights_str)
        if _aw is not None:
            _s = float(sum(_aw))
            assert _s > 0.0
            _aw = [x / _s for x in _aw]
        self._agg_weights_vec = torch.tensor(_aw, dtype=torch.float32) if _aw is not None else None
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
        # HCCF baseline (task #8, No.1 baseline). hccf=1 switches the whole forward/eval to
        # the faithful HCCF architecture (official akaxlh/HCCF): learnable hyperedge
        # dependency matrices u_hyper/i_hyper ([d,H]), per-layer local-GCN + hypergraph
        # HH^T E residual, sum-over-layers final embedding, and local(detached)-vs-
        # hypergraph per-layer contrast (batch-limited negatives). hccf_keep_rate is the
        # edge-drop keep probability (official 0.5); hccf_ssl_reg the contrast weight
        # (official 1e-3); hccf_temp the contrast temperature (official 1.0). These are
        # used INSTEAD of our ssl_reg/ssl_temp/ssl_anneal (which do not apply to HCCF).
        self.hccf = int(config["hccf"]) if "hccf" in config else 0
        assert self.hccf in (0, 1)
        self.hccf_hyper_num = int(config["hccf_hyper_num"]) if "hccf_hyper_num" in config else 128
        self.hccf_keep_rate = float(config["hccf_keep_rate"]) if "hccf_keep_rate" in config else 0.5
        assert 0.0 < self.hccf_keep_rate <= 1.0
        self.hccf_ssl_reg = float(config["hccf_ssl_reg"]) if "hccf_ssl_reg" in config else 1e-3
        self.hccf_temp = float(config["hccf_temp"]) if "hccf_temp" in config else 1.0
        # DCCF baseline (task #8, No.1 baseline, arXiv:2305.02759). dccf=1 switches the
        # whole forward/eval to the faithful DCCF architecture (official HKUDS/DCCF):
        # learnable intent prototypes user_intent/item_intent ([d,K]), per-layer four-view
        # disentangled aggregation (gnn / intent / graph-adaptive / intent-adaptive), and
        # gnn-vs-{int,gaa,iaa} contrast (batch-limited negatives). dccf_* hyperparameters
        # are the official defaults (n_intents=128, temp=1.0, emb_reg=2.5e-5,
        # cen_reg=5e-3, ssl_reg=1e-1); they REPLACE our ssl_reg/ssl_temp/ssl_anneal/reg
        # which do not apply to DCCF.
        self.dccf = int(config["dccf"]) if "dccf" in config else 0
        assert self.dccf in (0, 1)
        self.dccf_n_intents = int(config["dccf_n_intents"]) if "dccf_n_intents" in config else 128
        self.dccf_temp = float(config["dccf_temp"]) if "dccf_temp" in config else 1.0
        self.dccf_emb_reg = float(config["dccf_emb_reg"]) if "dccf_emb_reg" in config else 2.5e-5
        self.dccf_cen_reg = float(config["dccf_cen_reg"]) if "dccf_cen_reg" in config else 5e-3
        self.dccf_ssl_reg = float(config["dccf_ssl_reg"]) if "dccf_ssl_reg" in config else 1e-1
        # AdaGCL baseline (task #8, third No.1 baseline, arXiv:2305.10837). adagcl=1
        # switches the whole forward/eval/train to the faithful AdaGCL architecture
        # (official HKUDS/AdaGCL): a VGAE view generator + a graph-denoising view
        # generator, contrastive IM loss + information-bottleneck loss + mean-BPR, with
        # THREE separate optimisers (main / VGAE / denoising). The adagcl_* values are the
        # official defaults (ssl_reg=0.1, ib_reg=0.1, temp=0.5, beta=0.1, lambda0=1e-4,
        # gamma=-0.45, zeta=1.05, init_temperature=2.0, temperature_decay=0.98, reg=1e-5,
        # Adam eps for the denoising optimiser=1e-3); they REPLACE our
        # ssl_reg/ssl_temp/ssl_anneal/reg which do not apply to AdaGCL.
        self.adagcl = int(config["adagcl"]) if "adagcl" in config else 0
        assert self.adagcl in (0, 1)
        self.adagcl_ssl_reg = float(config["adagcl_ssl_reg"]) if "adagcl_ssl_reg" in config else 0.1
        self.adagcl_ib_reg = float(config["adagcl_ib_reg"]) if "adagcl_ib_reg" in config else 0.1
        self.adagcl_reg = float(config["adagcl_reg"]) if "adagcl_reg" in config else 1e-5
        self.adagcl_temp = float(config["adagcl_temp"]) if "adagcl_temp" in config else 0.5
        self.adagcl_beta = float(config["adagcl_beta"]) if "adagcl_beta" in config else 0.1
        self.adagcl_lambda0 = float(config["adagcl_lambda0"]) if "adagcl_lambda0" in config else 1e-4
        self.adagcl_gamma = float(config["adagcl_gamma"]) if "adagcl_gamma" in config else -0.45
        self.adagcl_zeta = float(config["adagcl_zeta"]) if "adagcl_zeta" in config else 1.05
        self.adagcl_init_temp = float(config["adagcl_init_temp"]) if "adagcl_init_temp" in config else 2.0
        self.adagcl_temp_decay = float(config["adagcl_temp_decay"]) if "adagcl_temp_decay" in config else 0.98
        self.adagcl_eps = float(config["adagcl_eps"]) if "adagcl_eps" in config else 1e-3

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)
        if self.swa_start > 0:
            self.logger.info("[swa] swa_start=%d (embedding-table weight averaging from this epoch)" % self.swa_start)

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
        if self.sem_emb:
            # Round-47: load the frozen semantic item features and attach them to the
            # _LightGCN module as a plain (non-Parameter) tensor. Must be created on
            # self.device (like the HCCF/DCCF parameters, a tensor assigned after .to()
            # does not move automatically). Loaded via torch.load-style numpy so it works
            # on both python 3.8 (scnumath) and 3.9 (masmath).
            sem_path = os.path.join(self.config.data_dir, self.dataset_name, "item_sem.npy")
            if os.path.exists(sem_path):
                item_sem = np.load(sem_path)
                assert item_sem.shape[0] == self.num_items, (
                    "item_sem.npy has %d rows but dataset has %d items"
                    % (item_sem.shape[0], self.num_items))
                assert item_sem.shape[1] == self.emb_size, (
                    "item_sem.npy has %d dims but embed_size=%d"
                    % (item_sem.shape[1], self.emb_size))
                self.lightgcn.item_sem = torch.from_numpy(
                    item_sem.astype(np.float32)).to(self.device)
                self.lightgcn.sem_lam = self.sem_lam
                self.logger.info("[sem] loaded item_sem %s -> shape=%s lam=%.2f" % (
                    sem_path, item_sem.shape, self.sem_lam))
            else:
                self.logger.warning("[sem] sem_emb=1 but %s missing; run "
                                    "util/build_semantic_features.py first" % sem_path)
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
        self.lightgcn.dropedge_keep = self.dropedge_keep
        self.lightgcn.res_alpha = self.res_alpha
        self.lightgcn.agg_weights = self._agg_weights_vec
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
        self.lightgcn.hccf = bool(self.hccf)
        self.lightgcn.hccf_hyper_num = self.hccf_hyper_num
        self.lightgcn.hccf_keep_rate = self.hccf_keep_rate
        self.lightgcn.hccf_temp = self.hccf_temp
        if self.hccf:
            # learnable hyperedge-dependency matrices (HCCF official: xavier_uniform on
            # [latdim, hyperNum] Parameters, registered on the _LightGCN module so they
            # are optimised by the same Adam). NOTE: created directly on self.device --
            # assigning a fresh Parameter after .to(device) does not move it (it would
            # otherwise stay on CPU and crash the GPU forward with a device mismatch).
            self.lightgcn.u_hyper = nn.Parameter(
                torch.empty(self.emb_size, self.hccf_hyper_num, device=self.device))
            self.lightgcn.i_hyper = nn.Parameter(
                torch.empty(self.emb_size, self.hccf_hyper_num, device=self.device))
            nn.init.xavier_uniform_(self.lightgcn.u_hyper)
            nn.init.xavier_uniform_(self.lightgcn.i_hyper)
            self.logger.info("[hccf] hyper_num=%d keep_rate=%.2f ssl_reg=%.1e temp=%.2f" % (
                self.hccf_hyper_num, self.hccf_keep_rate, self.hccf_ssl_reg, self.hccf_temp))
        self.lightgcn.dccf = bool(self.dccf)
        self.lightgcn.dccf_n_intents = self.dccf_n_intents
        self.lightgcn.dccf_temp = self.dccf_temp
        if self.dccf:
            # learnable intent prototypes (DCCF official: xavier_normal on [emb_dim, K]),
            # created directly on self.device (see HCCF note above).
            self.lightgcn.user_intent = nn.Parameter(
                torch.empty(self.emb_size, self.dccf_n_intents, device=self.device))
            self.lightgcn.item_intent = nn.Parameter(
                torch.empty(self.emb_size, self.dccf_n_intents, device=self.device))
            nn.init.xavier_normal_(self.lightgcn.user_intent)
            nn.init.xavier_normal_(self.lightgcn.item_intent)
            # bipartite edge list (both directions) for the adaptive-mask message passing;
            # matches the official plain_adj coo (adj + adj.T) used by _adaptive_mask.
            ui_pairs = self.dataset.train_data.to_user_item_pairs()
            users_np = ui_pairs[:, 0]
            items_np = ui_pairs[:, 1]
            nu = int(self.num_users)
            h = np.concatenate([users_np, items_np + nu]).astype(np.int64)
            t = np.concatenate([items_np + nu, users_np]).astype(np.int64)
            edge_idx = torch.from_numpy(np.stack([h, t], axis=0)).long().to(self.device)
            self.lightgcn.dccf_h = edge_idx[0]
            self.lightgcn.dccf_t = edge_idx[1]
            self.lightgcn.dccf_indices = edge_idx
            self.lightgcn.dccf_n_nodes = nu + int(self.num_items)
            self.logger.info("[dccf] n_intents=%d temp=%.2f emb_reg=%.1e cen_reg=%.1e ssl_reg=%.1e edges=%d" % (
                self.dccf_n_intents, self.dccf_temp, self.dccf_emb_reg,
                self.dccf_cen_reg, self.dccf_ssl_reg, int(edge_idx.shape[1])))
        self.lightgcn.adagcl = bool(self.adagcl)
        self.lightgcn.adagcl_temp = self.adagcl_temp
        self.lightgcn.adagcl_beta = self.adagcl_beta
        self.lightgcn.adagcl_reg = self.adagcl_reg
        self.lightgcn.adagcl_lambda0 = self.adagcl_lambda0
        self.lightgcn.adagcl_gamma = self.adagcl_gamma
        self.lightgcn.adagcl_zeta = self.adagcl_zeta
        if self.adagcl:
            # AdaGCL view-generator modules. Created directly on self.device (see the
            # HCCF note: a fresh Parameter/module assigned after .to(device) would stay on
            # CPU and crash the GPU forward). The full normalised adjacency's edge list is
            # cached once for the VGAE edge-probability decoder and the denoising attention.
            d = self.emb_size
            adj_c = self.lightgcn.norm_adj.coalesce()
            self.lightgcn.adagcl_indices = adj_c.indices()
            self.lightgcn.adagcl_h = adj_c.indices()[0]
            self.lightgcn.adagcl_t = adj_c.indices()[1]
            self.lightgcn.adagcl_vals = adj_c.values()
            self.lightgcn.adagcl_n_nodes = int(self.num_users + self.num_items)
            self.lightgcn.adagcl_emb_u = nn.Embedding(self.num_users, d).to(self.device)
            self.lightgcn.adagcl_emb_i = nn.Embedding(self.num_items, d).to(self.device)
            nn.init.xavier_uniform_(self.lightgcn.adagcl_emb_u.weight)
            nn.init.xavier_uniform_(self.lightgcn.adagcl_emb_i.weight)
            self.lightgcn.adagcl_enc_mean = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d)).to(self.device)
            self.lightgcn.adagcl_enc_std = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d), nn.Softplus()).to(self.device)
            self.lightgcn.adagcl_dec = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(d, d), nn.ReLU(inplace=True),
                nn.Linear(d, 1)).to(self.device)
            self.lightgcn.adagcl_nb = nn.ModuleList([
                nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True)).to(self.device)
                for _ in range(self.n_layers)])
            self.lightgcn.adagcl_self = nn.ModuleList([
                nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True)).to(self.device)
                for _ in range(self.n_layers)])
            self.lightgcn.adagcl_attn = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * d, 1)).to(self.device)
                for _ in range(self.n_layers)])
            self.logger.info("[adagcl] temp=%.2f ssl_reg=%.1e ib_reg=%.1e beta=%.2f lambda0=%.1e gamma=%.2f zeta=%.2f reg=%.1e init_temp=%.2f decay=%.3f eps=%.1e edges=%d" % (
                self.adagcl_temp, self.adagcl_ssl_reg, self.adagcl_ib_reg,
                self.adagcl_beta, self.adagcl_lambda0, self.adagcl_gamma,
                self.adagcl_zeta, self.adagcl_reg, self.adagcl_init_temp,
                self.adagcl_temp_decay, self.adagcl_eps,
                int(self.lightgcn.adagcl_h.shape[0])))
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
        self.lightgcn.spec_bal = self.spec_bal
        self.lightgcn.spec_bal_center = self.spec_bal_center
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
        if self.agg_weights_str != "uniform":
            self.logger.info("[layer-agg] agg_weights=%s alpha=%.3f vec=%s" % (
                self.agg_weights_str, self.agg_alpha,
                ",".join("%.3f" % float(x) for x in self._agg_weights_vec.tolist())))
        if self.weight_decay:
            self.logger.info("[weight-decay] wd=%.2e (Adam decoupled; batch-L2 reg=%.2e)" % (
                self.weight_decay, self.reg))
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
        if self.adagcl:
            # AdaGCL uses THREE optimisers (official Main.py): opt (main embeddings),
            # opt_gen_1 (VGAE generator), opt_gen_2 (denoising generator, with its own
            # Adam eps). The 'adagcl_' prefix on the generator submodules cleanly splits
            # the parameter groups; the main group additionally contains the unused
            # fcl/directau projectors (they receive no gradient in adagcl mode, harmless).
            main_params = [p for n, p in self.lightgcn.named_parameters()
                           if not n.startswith("adagcl_")]
            vgae_params = [p for n, p in self.lightgcn.named_parameters()
                           if n.startswith("adagcl_emb_") or n.startswith("adagcl_enc_")
                           or n.startswith("adagcl_dec")]
            denoise_params = [p for n, p in self.lightgcn.named_parameters()
                              if n.startswith("adagcl_nb") or n.startswith("adagcl_self")
                              or n.startswith("adagcl_attn")]
            self.optimizer = torch.optim.Adam(main_params, lr=self.lr, weight_decay=0.0)
            self.opt_gen_1 = torch.optim.Adam(vgae_params, lr=self.lr, weight_decay=0.0)
            self.opt_gen_2 = torch.optim.Adam(denoise_params, lr=self.lr, weight_decay=0.0,
                                              eps=self.adagcl_eps)
            self.logger.info("[adagcl] optimisers: main=%d params, vgae=%d, denoise=%d" % (
                len(main_params), len(vgae_params), len(denoise_params)))
        else:
            self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay)

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

    def _lr(self, epoch):
        """Effective learning rate at a given epoch (Round-37: LR schedule).

        lr_anneal=0 returns the fixed self.lr (backward compatible). Otherwise the
        optimizer lr is decayed from self.lr to self.lr*self.lr_floor over
        lr_anneal_epochs epochs (linear or cosine) and then held. This is the
        optimizer-step-size counterpart of the SSL-weight annealing (R13/R15): the
        SSL weight controls *what* objective is optimised, the lr controls *how
        aggressively* it is optimised, so the two schedules are independent.
        """
        if not self.lr_anneal or self.epochs <= 1:
            return self.lr
        total = float(self.lr_anneal_epochs) if self.lr_anneal_epochs > 0 else float(self.epochs)
        t = min(max(float(epoch - 1) / max(total - 1.0, 1.0), 0.0), 1.0)
        floor_lr = self.lr * self.lr_floor
        if self.lr_anneal == 1:          # linear decay to floor
            sched = self.lr + (floor_lr - self.lr) * t
        elif self.lr_anneal == 2:        # cosine decay to floor
            sched = floor_lr + (self.lr - floor_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            sched = self.lr
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

    def _adagcl_temperature(self, epoch):
        """L0 hard-concrete temperature schedule (official Main.py):
        temperature = max(0.05, init_temperature * decay^(epoch-1))."""
        return max(0.05, self.adagcl_init_temp * (self.adagcl_temp_decay ** float(epoch - 1)))

    def _adagcl_batch_loop(self, epoch, data_iter):
        """One epoch of the faithful AdaGCL training loop (official Main.py trainEpoch),
        with the three-optimiser orchestration: (1) IM InfoNCE between the VGAE view and
        the denoised view -> step main; (2) information-bottleneck contrast of the
        post-step views against the detached pre-step views -> step main; (3) mean-BPR +
        reg on the main model + the two generator losses -> step main/VGAE/denoising.
        Returns (total_loss, total_bpr, total_reg) for the epoch log line."""
        temperature = self._adagcl_temperature(epoch)
        full_adj = self.lightgcn.norm_adj
        total_loss = total_bpr = total_reg = 0.0
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            bat_users = torch.from_numpy(bat_users).long().to(self.device)
            bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
            bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
            # VGAE view is generated ONCE per batch under no_grad (official
            # generator_generate wraps generate in torch.no_grad).
            with torch.no_grad():
                data1 = self.lightgcn._adagcl_vgae_generate()
            self.optimizer.zero_grad()
            self.opt_gen_1.zero_grad()
            self.opt_gen_2.zero_grad()
            # (1) IM loss
            out1 = self.lightgcn._adagcl_gcn_sum(data1, self.lightgcn._adagcl_main_embeds())
            out2 = self.lightgcn._adagcl_denoise_view()
            im = self.adagcl_ssl_reg * self.lightgcn._adagcl_infonce(
                out1, out2, bat_users, bat_pos_items).mean()
            im.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # (2) information-bottleneck loss (post-step views vs detached pre-step views)
            _out1 = self.lightgcn._adagcl_gcn_sum(data1, self.lightgcn._adagcl_main_embeds())
            _out2 = self.lightgcn._adagcl_denoise_view()
            ib = self.adagcl_ib_reg * (
                self.lightgcn._adagcl_infonce(_out1, out1.detach(), bat_users, bat_pos_items).mean()
                + self.lightgcn._adagcl_infonce(_out2, out2.detach(), bat_users, bat_pos_items).mean())
            ib.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # (3) main mean-BPR + reg (full-embedding L2, official calcRegLoss)
            emb = self.lightgcn._adagcl_gcn_sum(full_adj, self.lightgcn._adagcl_main_embeds())
            ue, ie = torch.split(emb, [self.num_users, self.num_items], dim=0)
            anc = ue[bat_users]; pos = ie[bat_pos_items]; neg = ie[bat_neg_items]
            score_diff = (anc * pos).sum(-1) - (anc * neg).sum(-1)
            bpr = -torch.log(torch.sigmoid(score_diff) + 1e-8).mean()
            reg = self.adagcl_reg * (
                self.lightgcn.user_embeddings.weight.norm(2).square()
                + self.lightgcn.item_embeddings.weight.norm(2).square())
            (bpr + reg).backward()
            # generator losses (VGAE reconstruction+KL+BPR, denoising BPR+L0)
            loss1 = self.lightgcn._adagcl_vgae_loss(bat_users, bat_pos_items, bat_neg_items)
            loss2 = self.lightgcn._adagcl_denoise_loss(
                bat_users, bat_pos_items, bat_neg_items, temperature)
            (loss1 + loss2).backward()
            self.optimizer.step()
            self.opt_gen_1.step()
            self.opt_gen_2.step()
            total_loss += float((im + ib + bpr + loss1 + loss2).detach())
            total_bpr += float(bpr.detach())
            total_reg += float(reg.detach())
        return total_loss, total_bpr, total_reg

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            if self.adagcl:
                # AdaGCL baseline: a fully separate training loop (three optimisers, IM/IB/
                # BPR/generator losses) that does not go through the NoiseSGL forward().
                total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
                training_start_time = time()
                self.lightgcn.train()
                data_iter = self._pairwise_iter()
                total_loss, total_bpr_loss, total_reg_loss = self._adagcl_batch_loop(epoch, data_iter)
                self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                    epoch,
                    total_loss / self.num_ratings,
                    total_bpr_loss / self.num_ratings,
                    (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,
                    total_reg_loss / self.num_ratings,
                    time() - training_start_time,))
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
                continue
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
            if self.hccf:
                # HCCF: one edge-dropped adjacency per epoch (keep hccf_keep_rate of edges;
                # our create_adj_mat drops self.ssl_ratio of edges, so override it to
                # 1 - keep_rate for this call only). The hypergraph-side dropout is applied
                # inside _forward_hccf (training-only).
                _saved_ratio = self.ssl_ratio
                self.ssl_ratio = 1.0 - self.hccf_keep_rate
                sub_graph1 = sp_mat_to_sp_tensor(
                    self.create_adj_mat(is_subgraph=True, aug_type="ed")).to(self.device)
                self.ssl_ratio = _saved_ratio
            # Round-49: DropEdge-style edge dropout on the MAIN propagation branch.
            # One freshly-thinned (and re-normalised) adjacency per epoch, used ONLY by
            # the prediction branch (BPR); the noise views keep the full norm_adj, so
            # the SSL signal is unchanged. eval() still uses self.norm_adj (full graph).
            main_adj = None
            if self.dropedge_keep < 1.0:
                _saved_ratio = self.ssl_ratio
                self.ssl_ratio = 1.0 - self.dropedge_keep
                main_adj = sp_mat_to_sp_tensor(
                    self.create_adj_mat(is_subgraph=True, aug_type="ed")).to(self.device)
                self.ssl_ratio = _saved_ratio
            data_iter = self._pairwise_iter()
            self.lightgcn.train()
            cur_eps = self._noise_eps(epoch)
            cur_temp = self._ssl_temp(epoch)
            if cur_temp != self.lightgcn.ssl_temp:
                self.lightgcn.ssl_temp = cur_temp
                self.ssl_temp = cur_temp
            # Round-37: per-epoch LR schedule (optimizer-level). Applied before the batch
            # loop so every step of this epoch uses the scheduled lr; lr_anneal=0 keeps the
            # initial self.lr (backward compatible, no-op).
            if self.lr_anneal:
                cur_lr = self._lr(epoch)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = cur_lr
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, infonce_loss, fcl_loss, dau_loss, spec_loss = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items, eps=cur_eps,
                    main_adj=main_adj)
                # Round-41: BPR ranking margin. Applies to both the [B] and [B,K] logit
                # shapes (multi-negative R21), and ONLY to the BPR term (InfoNCE keeps its
                # own temperature). margin=0 is a no-op (backward compatible).
                if self.bpr_margin > 0.0:
                    sup_logits = sup_logits - self.bpr_margin
                
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
                if self.hccf or self.dccf:
                    # HCCF/DCCF official use a MEAN-scaled BPR (batch mean), matching their
                    # mean-scaled contrast losses so the official SSL:BPR ratios hold.
                    bpr_loss = -torch.mean(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss is computed inside forward() so that >=2 views can be
                # mixed; for view_mix="nn" it is identical to the original single pair.
                # DMLC: multi-layer contrast (disable with use_mlc=0 for pure SimGCL-style).
                if self.use_mlc and not self.hccf and not self.dccf:
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

                if self.hccf:
                    # HCCF faithful loss: mean-BPR + hccf_ssl_reg * mean-contrast + reg * L2.
                    # No annealing / fcl / dau / spec -- those are our fusion components
                    # (R13/R15/R8/R9/R35), not part of HCCF, and would break faithfulness.
                    loss = bpr_loss + self.hccf_ssl_reg * infonce_loss + self.reg * reg_loss
                elif self.dccf:
                    # DCCF faithful loss: mean-BPR + emb_reg*L2(batch embeddings) +
                    # cen_reg*||intent||^2 (prototype regularisation) + ssl_reg*contrast.
                    # No annealing / fcl / dau / spec (our fusion components, not DCCF's).
                    cen_loss = torch.sum(self.lightgcn.user_intent.pow(2)) + \
                        torch.sum(self.lightgcn.item_intent.pow(2))
                    loss = bpr_loss + self.dccf_emb_reg * reg_loss + \
                        self.dccf_cen_reg * cen_loss + self.dccf_ssl_reg * infonce_loss
                else:
                    ssl_weight = self._ssl_weight(epoch)
                    loss = bpr_loss + self.ssl_reg * ssl_weight * infonce_loss + self.reg * reg_loss
                    if fcl_loss is not None:
                        loss = loss + fcl_loss
                    if dau_loss is not None:
                        loss = loss + dau_loss
                    if spec_loss is not None:
                        loss = loss + self.spec_bal * spec_loss
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

            # Round-40: accumulate the embedding weights into the SWA running average
            # (after this epoch's optimizer steps, before the periodic evaluation).
            if self.swa_start > 0 and epoch >= self.swa_start:
                with torch.no_grad():
                    if self.swa_n == 0:
                        self.swa_user = self.lightgcn.user_embeddings.weight.data.clone()
                        self.swa_item = self.lightgcn.item_embeddings.weight.data.clone()
                    else:
                        self.swa_user.add_(self.lightgcn.user_embeddings.weight.data)
                        self.swa_item.add_(self.lightgcn.item_embeddings.weight.data)
                    self.swa_n += 1

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
    def _swap_in_swa(self):
        """Temporarily replace the embedding tables with the SWA running average for
        evaluation; the current weights are backed up into persistent buffers."""
        with torch.no_grad():
            if self._swa_bk_user is None:
                self._swa_bk_user = self.lightgcn.user_embeddings.weight.data.clone()
                self._swa_bk_item = self.lightgcn.item_embeddings.weight.data.clone()
            else:
                self._swa_bk_user.copy_(self.lightgcn.user_embeddings.weight.data)
                self._swa_bk_item.copy_(self.lightgcn.item_embeddings.weight.data)
            self.lightgcn.user_embeddings.weight.data.copy_(self.swa_user / float(self.swa_n))
            self.lightgcn.item_embeddings.weight.data.copy_(self.swa_item / float(self.swa_n))

    def _swap_out_swa(self):
        """Restore the original (non-averaged) embedding weights after SWA evaluation."""
        with torch.no_grad():
            self.lightgcn.user_embeddings.weight.data.copy_(self._swa_bk_user)
            self.lightgcn.item_embeddings.weight.data.copy_(self._swa_bk_item)

    def evaluate_model(self):
        flag = False
        swa_active = self.swa_n > 0
        if swa_active:
            self._swap_in_swa()
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if swa_active:
            self._swap_out_swa()
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()

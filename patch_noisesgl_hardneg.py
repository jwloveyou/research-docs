import io, sys

p = "model/general_recommender/NoiseSGL.py"
s = io.open(p, encoding="utf-8").read()

# --- 1) add config reads -------------------------------------------------
old1 = '        self.use_mlc = config["use_mlc"] if "use_mlc" in config else 1\n'
new1 = old1 + (
    '        # Round-2: hardness-calibrated (semi-hard) negative sampling.\n'
    '        self.neg_mode = config["neg_mode"] if "neg_mode" in config else "uniform"\n'
    '        assert self.neg_mode in ["uniform", "semihard"]\n'
    '        self.neg_K = int(config["neg_K"]) if "neg_K" in config else 5\n'
)
assert s.count(old1) == 1, ("old1 count", s.count(old1))
s = s.replace(old1, new1)

# --- 2) insert sampler methods + restructure epoch loop ------------------
old2 = (
    '    def train_model(self):\n'
    '        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    \n'
    '        self.logger.info(self.evaluator.metrics_info())\n'
    '        stopping_step = 0\n'
    '        for epoch in range(1, self.epochs + 1):\n'
    '            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0\n'
    '            training_start_time = time()\n'
    '            # NoiseSGL: views come from embedding-space perturbation, no sub-graphs needed.\n'
    '            sub_graph1, sub_graph2 = None, None\n'
    '            self.lightgcn.train()\n'
    '            for bat_users, bat_pos_items, bat_neg_items in data_iter:\n'
)

new2 = '''    @staticmethod
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
'''

assert s.count(old2) == 1, ("old2 count", s.count(old2))
s = s.replace(old2, new2)

io.open(p, "w", encoding="utf-8").write(s)
print("patched OK")

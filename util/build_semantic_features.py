"""Build FROZEN semantic item features (task #6 / 4th-component semantic axis).

For each item we assemble a text/tag description, encode it into a TF-IDF tag
vector, and project it with a FROZEN (seeded, non-learnable) Gaussian matrix to
the model embedding dimension. The resulting `item_sem.npy` (n_items x embed_dim,
row L2-normalised) is a static semantic prior injected into the item embeddings by
the NoiseSGL `sem_emb` switch (see model/general_recommender/NoiseSGL.py).

Semantic sources (public metadata, no external downloads beyond what is already
in dataset/_raw):
  * ml-1m     : movie title + genre list (movies.dat, already clean)
  * lastfm-2k : artist name + user-assigned tags (user_taggedartists.dat + tags.dat)

Optional LLM enrichment: `--llm_json <file>` merges DeepSeek-generated tags per
item (produced by util/llm_enrich_items.py) into the tag vocabulary, so the final
semantic vector carries both the raw metadata tags and the LLM's style/genre tags.
The projection matrix is FROZEN (torch.no_grad / never registered as a parameter),
which respects the R1/R15/R18 'static wins' law and the R4/R6/R7 'do not touch the
propagation operator' constraint (semantic features are only added to the layer-0
item embedding inside _forward_gcn, before the shared GCN operator).

Literature: LLMRec (arXiv:2311.00423) -- LLM item-attribute enhancement; TIGER
(arXiv:2305.14395) -- item-title semantics for cold-start; the injection here is
the *static* (frozen, no projector training) counterpart of LLMRec's learnable
feature projector, chosen to respect this repo's 'static wins' iron law.

Usage:
  python util/build_semantic_features.py [--datasets ml-1m,lastfm-2k]
                                         [--embed_size 256] [--llm_json path]
"""
import os
import sys
import json
import argparse
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw")
SEED = 42
CORE = 10


# ---------------------------------------------------------------- data loading
def load_ml1m_pairs():
    pairs = []
    with open(os.path.join(RAW, "ml-1m", "ratings.dat")) as f:
        for line in f:
            u, i, _r, _t = line.strip().split("::")
            pairs.append((int(u), int(i)))
    return np.asarray(pairs, dtype=np.int64)


def load_lastfm_pairs():
    pairs = []
    with open(os.path.join(RAW, "user_artists.dat")) as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            u, i, w = int(p[0]), int(p[1]), int(p[2])
            if w > 0:
                pairs.append((u, i))
    return np.asarray(pairs, dtype=np.int64)


def load_ml10m_pairs():
    """MovieLens-10M ratings.dat: UserID::MovieID::Rating::Timestamp (ml-1m format)."""
    pairs = []
    with open(os.path.join(RAW, "ml-10m", "ratings.dat")) as f:
        for line in f:
            u, i, _r, _t = line.strip().split("::")
            pairs.append((int(u), int(i)))
    return np.asarray(pairs, dtype=np.int64)


def k_core(pairs, k):
    pairs = np.asarray(pairs, dtype=np.int64)
    while True:
        users, cu = np.unique(pairs[:, 0], return_counts=True)
        items, ci = np.unique(pairs[:, 1], return_counts=True)
        ku = set(users[cu >= k].tolist())
        ki = set(items[ci >= k].tolist())
        m = np.array([(u in ku) and (i in ki) for u, i in pairs], dtype=bool)
        nxt = pairs[m]
        if len(nxt) == len(pairs):
            return nxt
        pairs = nxt


def reconstruct_org_item_ids(ds):
    """Return the sorted unique org item ids surviving 10-core filtering.

    The repo's prepare_new_datasets.py remaps via np.unique(pairs[:,1]) (sorted
    ascending) -> 0..n_items-1, and does not save the mapping, so we reconstruct
    it deterministically here (the id assignment is seed-independent).
    """
    loader = {"ml-1m": load_ml1m_pairs,
              "ml-10m": load_ml10m_pairs}.get(ds, load_lastfm_pairs)
    pairs = k_core(loader(), CORE)
    return np.unique(pairs[:, 1]).astype(np.int64)


def reconstruct_org_user_ids(ds):
    """Return the sorted unique org user ids surviving 10-core filtering.

    Symmetric to reconstruct_org_item_ids: prepare_new_datasets.py remaps the user
    column via np.unique(pairs[:,0]) (sorted ascending) -> 0..n_users-1. We rebuild
    the same sorted org-user list deterministically (seed-independent).
    """
    loader = {"ml-1m": load_ml1m_pairs,
              "ml-10m": load_ml10m_pairs}.get(ds, load_lastfm_pairs)
    pairs = k_core(loader(), CORE)
    return np.unique(pairs[:, 0]).astype(np.int64)


# ---------------------------------------------------------------- text sources
def load_lastfm_artist_tags():
    """org artist id -> (name, list of tag strings)."""
    id2name = {}
    with open(os.path.join(RAW, "artists.dat"), encoding="latin-1") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            p = line.strip().split("\t")
            if len(p) >= 2:
                try:
                    id2name[int(p[0])] = p[1]
                except ValueError:
                    continue

    id2tag = {}
    with open(os.path.join(RAW, "tags.dat"), encoding="latin-1") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            p = line.strip().split("\t")
            if len(p) >= 2:
                try:
                    id2tag[int(p[0])] = p[1].lower()
                except ValueError:
                    continue

    artist_tags = {}
    with open(os.path.join(RAW, "user_taggedartists.dat")) as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            try:
                artist = int(p[1])
                tag = int(p[2])
            except ValueError:
                continue
            tagname = id2tag.get(tag)
            if tagname:
                artist_tags.setdefault(artist, set()).add(tagname)

    return id2name, {a: sorted(t) for a, t in artist_tags.items()}


def load_ml1m_item_text():
    """org movie id -> (title, list of genre tags)."""
    id2text = {}
    with open(os.path.join(RAW, "ml-1m", "movies.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) < 3:
                continue
            mid = int(p[0])
            title = p[1]
            genres = [g.lower() for g in p[2].split("|")]
            id2text[mid] = (title, genres)
    return id2text


def load_ml10m_item_text():
    """org movie id -> (title, list of genre + user tags) for MovieLens-10M."""
    id2text = {}
    with open(os.path.join(RAW, "ml-10m", "movies.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) < 3:
                continue
            mid = int(p[0])
            title = p[1]
            genres = [g.lower() for g in p[2].split("|")]
            id2text[mid] = (title, genres)
    # merge user-assigned free-text tags (tags.dat: user::movie::tag::ts)
    id2tags = {}
    with open(os.path.join(RAW, "ml-10m", "tags.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) < 3:
                continue
            try:
                mid = int(p[1])
            except ValueError:
                continue
            id2tags.setdefault(mid, set()).add(p[2].lower())
    for mid, (title, genres) in id2text.items():
        id2text[mid] = (title, genres + sorted(id2tags.get(mid, set())))
    return id2text


def load_lastfm_user_tags():
    """org user id -> (label, list of tag strings) from user_taggedartists.dat.

    lastfm-2k records which tags a user applied to which artist. We aggregate all
    tag VALUES a user has ever applied (across all artists) into one per-user tag
    vocabulary -- the user-side counterpart of the item-side artist-tag features.
    """
    id2tag = {}
    with open(os.path.join(RAW, "tags.dat"), encoding="latin-1") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            p = line.strip().split("\t")
            if len(p) >= 2:
                try:
                    id2tag[int(p[0])] = p[1].lower()
                except ValueError:
                    continue

    user_tags = {}
    with open(os.path.join(RAW, "user_taggedartists.dat"), encoding="latin-1") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            p = line.strip().split("\t")
            if len(p) < 3:
                continue
            try:
                user = int(p[0])
                tag = int(p[2])
            except ValueError:
                continue
            tagname = id2tag.get(tag)
            if tagname:
                user_tags.setdefault(user, set()).add(tagname)

    return {u: ("user%d" % u, sorted(t)) for u, t in user_tags.items()}


def load_ml1m_user_text():
    """org user id -> (label, list of demographic tags) from users.dat.

    users.dat: UserID::Gender::Age::Occupation::Zip-code. Gender/age/occupation are
    coarse but meaningful demographic priors for a CF user embedding (the user-side
    counterpart of the item-side title/genre features). Zip-code is dropped (too
    sparse / quasi-identifying and not a semantic attribute).
    """
    id2text = {}
    with open(os.path.join(RAW, "ml-1m", "users.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            if len(p) < 4:
                continue
            try:
                uid = int(p[0])
            except ValueError:
                continue
            gender = p[1].strip().lower()
            age = p[2].strip().lower()
            occ = p[3].strip().lower()
            tags = ["gender:" + gender, "age:" + age, "occupation:" + occ]
            id2text[uid] = ("user%d" % uid, tags)
    return id2text


# ---------------------------------------------------------------- encoding
def build_tfidf(texts, min_df=1):
    """texts: list of (name, [tags]). Returns tfidf matrix + vocab list."""
    vocab = {}
    for _name, tags in texts:
        for t in tags:
            vocab[t] = vocab.get(t, 0) + 1
    # keep tags appearing in >= min_df items, sort by frequency desc then alpha
    kept = sorted([t for t, c in vocab.items() if c >= min_df],
                  key=lambda t: (-vocab[t], t))
    t2i = {t: i for i, t in enumerate(kept)}
    n = len(texts)
    d = len(kept)
    X = np.zeros((n, d), dtype=np.float32)
    for r, (_name, tags) in enumerate(texts):
        for t in tags:
            if t in t2i:
                X[r, t2i[t]] += 1.0
    # sublinear tf + idf (smooth)
    df = (X > 0).sum(axis=0)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
    X = np.log1p(X) * idf[None, :]
    # row L2-normalise (zero rows stay zero)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(norm > 0, X / np.maximum(norm, 1e-12), X)
    return X, kept


def frozen_projection(tag_dim, embed_dim, seed=SEED):
    """FROZEN seeded Gaussian projection matrix [tag_dim, embed_dim], columns L2-norm."""
    rng = np.random.RandomState(seed)
    P = rng.randn(tag_dim, embed_dim).astype(np.float32)
    P = P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ml-1m,lastfm-2k")
    ap.add_argument("--embed_size", type=int, default=256)
    ap.add_argument("--llm_json", default=None, help="optional LLM enrichment json")
    ap.add_argument("--side", default="item", choices=["item", "user"],
                    help="item = item_sem.npy (default, R47); user = user_sem.npy (R91)")
    args = ap.parse_args()

    llm = {}
    if args.llm_json:
        with open(args.llm_json) as f:
            llm = {int(k): v for k, v in json.load(f).items()}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        if args.side == "item":
            org_ids = reconstruct_org_item_ids(ds)
            n_ents = len(org_ids)
            if ds == "ml-1m":
                id2text = load_ml1m_item_text()
            elif ds == "ml-10m":
                id2text = load_ml10m_item_text()
            else:
                id2name, id2tags = load_lastfm_artist_tags()
                id2text = {a: (id2name.get(a, "artist%d" % a), id2tags.get(a, []))
                           for a in org_ids}
            default_label = "item%d"
            out_name = "item_sem.npy"
            meta_key = "n_items"
        else:
            org_ids = reconstruct_org_user_ids(ds)
            n_ents = len(org_ids)
            if ds == "ml-1m":
                id2text = load_ml1m_user_text()
            else:
                id2text = load_lastfm_user_tags()
            default_label = "user%d"
            out_name = "user_sem.npy"
            meta_key = "n_users"

        texts = []
        n_native_tagged = 0
        n_llm_tagged = 0
        for oid in org_ids:
            title, tags = id2text.get(int(oid), (default_label % oid, []))
            native = list(tags)
            if native:
                n_native_tagged += 1
            extra = [t.lower() for t in llm.get(int(oid), [])]
            if extra:
                n_llm_tagged += 1
            texts.append((title, native + extra))

        X, vocab = build_tfidf(texts)
        P = frozen_projection(len(vocab), args.embed_size)
        sem = X @ P  # [n_ents, embed_dim]
        # re-normalise projected rows (projection keeps ~unit norm for non-zero rows)
        norm = np.linalg.norm(sem, axis=1, keepdims=True)
        sem = np.where(norm > 0, sem / np.maximum(norm, 1e-12), sem)

        out_dir = os.path.join(BASE, ds)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, out_name), sem.astype(np.float32))
        meta = {
            "dataset": ds,
            "side": args.side,
            meta_key: n_ents,
            "embed_size": args.embed_size,
            "vocab_size": len(vocab),
            "top_vocab": vocab[:40],
            "n_native_tagged": n_native_tagged,
            "n_llm_tagged": n_llm_tagged,
            "zero_feature_rows": int((np.linalg.norm(sem, axis=1) < 1e-6).sum()),
            "projection": "frozen_seeded_gaussian_seed=%d" % SEED,
        }
        with open(os.path.join(out_dir, out_name.replace(".npy", "_meta.json")), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("[sem-%s] %s: %s=%d vocab=%d native_tagged=%d llm_tagged=%d zero_rows=%d"
              % (args.side, ds, meta_key, n_ents, len(vocab), n_native_tagged,
                 n_llm_tagged, meta["zero_feature_rows"]))
        print("[sem-%s]   -> %s" % (args.side, os.path.join(out_dir, out_name)))


if __name__ == "__main__":
    main()

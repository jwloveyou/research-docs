"""Prepare two extra public benchmark datasets (MovieLens-1M and LastFM-2K)
into the pair format expected by this repo (dataset/<name>/<name>.{train,test},
one "user,item" pair per line, 0-indexed contiguous ids).

Protocol (matches the existing yelp2018 / amazon-book / gowalla setup):
  * binarize every observed interaction into an implicit positive
  * 10-core filtering (iteratively drop users/items with < 10 interactions)
  * global random 80/20 train/test split (seed fixed for reproducibility)

Sources (GroupLens, direct download, no auth):
  * ml-1m      : https://files.grouplens.org/datasets/movielens/ml-1m.zip
                 (ratings.dat: UserID::MovieID::Rating::Timestamp)
  * lastfm-2k  : http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
                 (user_artists.dat: userID<TAB>artistID<TAB>weight)

Run:  python util/prepare_new_datasets.py
"""
import os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw")
SEED = 42
CORE = 10
TRAIN_RATIO = 0.8


def load_ml1m():
    pairs = []
    with open(os.path.join(RAW, "ml-1m", "ratings.dat")) as f:
        for line in f:
            u, i, _r, _t = line.strip().split("::")
            pairs.append((int(u), int(i)))
    return pairs


def load_lastfm():
    pairs = []
    with open(os.path.join(RAW, "user_artists.dat")) as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            u, i, w = int(parts[0]), int(parts[1]), int(parts[2])
            if w > 0:
                pairs.append((u, i))
    return pairs


def k_core(pairs, k):
    """Iteratively drop users/items with degree < k until stable."""
    pairs = np.asarray(pairs, dtype=np.int64)
    while True:
        users, counts_u = np.unique(pairs[:, 0], return_counts=True)
        items, counts_i = np.unique(pairs[:, 1], return_counts=True)
        keep_u = set(users[counts_u >= k].tolist())
        keep_i = set(items[counts_i >= k].tolist())
        mask = np.array([(u in keep_u) and (i in keep_i) for u, i in pairs], dtype=bool)
        new_pairs = pairs[mask]
        if len(new_pairs) == len(pairs):
            return new_pairs
        pairs = new_pairs


def remap(pairs):
    users = np.unique(pairs[:, 0])
    items = np.unique(pairs[:, 1])
    u2id = {u: i for i, u in enumerate(users)}
    i2id = {i: j for j, i in enumerate(items)}
    ru = np.array([u2id[u] for u in pairs[:, 0]], dtype=np.int64)
    ri = np.array([i2id[i] for i in pairs[:, 1]], dtype=np.int64)
    return np.stack([ru, ri], axis=1), len(users), len(items)


def write_pairs(name, pairs):
    d = os.path.join(BASE, name)
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(pairs))
    n_train = int(len(pairs) * TRAIN_RATIO)
    for split, idx in (("train", perm[:n_train]), ("test", perm[n_train:])):
        path = os.path.join(d, "%s.%s" % (name, split))
        with open(path, "w") as f:
            for r in idx:
                f.write("%d,%d\n" % (pairs[r, 0], pairs[r, 1]))
        print("WROTE %s pairs=%d" % (path, len(idx)))


def main():
    jobs = [("ml-1m", load_ml1m()), ("lastfm-2k", load_lastfm())]
    for name, raw in jobs:
        print("== %s: raw interactions=%d" % (name, len(raw)))
        pairs = k_core(raw, CORE)
        pairs, nu, ni = remap(pairs)
        print("   %d-core -> %d interactions, %d users, %d items" % (CORE, len(pairs), nu, ni))
        write_pairs(name, pairs)
    print("DONE")


if __name__ == "__main__":
    main()

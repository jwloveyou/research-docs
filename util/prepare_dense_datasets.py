"""Prepare extra DENSE public benchmarks to pin down the density flip-point of the
noise-augmentation method (task #7). ml-1m (graph density 5.07%) is currently the
single dense counter-example (fusion < LightGCN); we add MovieLens-10M (moderate
density, near the flip point) and Jester (extremely dense, ~70%+) to bracket it.

Protocol is IDENTICAL to util/prepare_new_datasets.py (R22, ml-1m / lastfm):
  * implicit binarisation: any observed rating != null -> positive pair
  * 10-core filtering (iteratively drop users/items with < 10 interactions)
  * global random 80/20 train/test split (seed 42)

Sources:
  * ml-10m  : https://files.grouplens.org/datasets/movielens/ml-10m.zip
              ratings.dat: UserID::MovieID::Rating::Timestamp (same as ml-1m)
  * jester-1: https://goldberg.berkeley.edu/jester-data/jester-data-1.zip
              jester-data-1.xls: 24983 users x 101 cols; col0 = #rated jokes,
              cols 1..100 = ratings in [-10, +10], 99.0 = null. We treat every
              non-null rating as an implicit positive (== ml-1m "any rating").
              (Book-Crossing: official link http://www2.informatik.uni-freiburg.de/
              ~cziegler/BX/BX-CSV-Dump.zip is 404/dead, and it is actually sparse
              (0.0015%), so it does not serve the dense-flip-point purpose -> skipped.)

Run:  python util/prepare_dense_datasets.py
"""
import os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw")
SEED = 42
CORE = 10
TRAIN_RATIO = 0.8


def load_ml10m():
    pairs = []
    with open(os.path.join(RAW, "ml-10m", "ratings.dat")) as f:
        for line in f:
            u, i, _r, _t = line.strip().split("::")
            pairs.append((int(u), int(i)))
    return pairs


def load_jester():
    """Parse jester-data-1.xls (24983 x 101). col0 = #rated, cols 1..100 = ratings,
    99.0 = null. Every non-null rating becomes an implicit positive pair."""
    import xlrd
    wb = xlrd.open_workbook(os.path.join(RAW, "jester", "jester-data-1.xls"))
    sh = wb.sheet_by_index(0)
    pairs = []
    for u in range(sh.nrows):
        for j in range(1, sh.ncols):  # jokes are columns 1..100 -> item id j-1
            v = sh.cell_value(u, j)
            if v != 99.0:
                pairs.append((u, j - 1))
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
    jobs = []
    ml10 = os.path.join(RAW, "ml-10m", "ratings.dat")
    if os.path.exists(ml10):
        jobs.append(("ml-10m", load_ml10m()))
    else:
        print("SKIP ml-10m (ratings.dat missing; unzip ml-10m.zip first)")
    jx = os.path.join(RAW, "jester", "jester-data-1.xls")
    if os.path.exists(jx):
        jobs.append(("jester", load_jester()))
    else:
        print("SKIP jester (jester-data-1.xls missing)")

    for name, raw in jobs:
        print("== %s: raw interactions=%d" % (name, len(raw)))
        pairs = k_core(raw, CORE)
        pairs, nu, ni = remap(pairs)
        density = len(pairs) / (nu * ni) if nu * ni else 0.0
        print("   %d-core -> %d interactions, %d users, %d items, density=%.4f" %
              (CORE, len(pairs), nu, ni, density))
        write_pairs(name, pairs)
    print("DONE")


if __name__ == "__main__":
    main()

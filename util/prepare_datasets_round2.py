"""Prepare three extra public benchmark datasets (Douban-Book, iFashion, Tmall)
into the pair format expected by this repo (dataset/<name>/<name>.{train,test},
one "user,item" pair per line, comma separated, 0-indexed contiguous ids).

Protocol (matches the existing setup; these come as pre-split standard benchmark
train/test, so we keep the authors' split, not a fresh 80/20):
  * binarize every observed interaction into an implicit positive
  * drop test pairs whose user or item has no training history (cold-start
    pairs cannot be scored under a pure-ID CF model; standard practice)
  * joint remap of train+test ids to 0-indexed contiguous

Sources (direct raw.githubusercontent download, no auth):
  * douban-book : Coder-Yu/SELFRec  dataset/douban-book/{train,test}.txt
                  ("user item rating", space separated, ids 1-based)
  * ifashion    : Coder-Yu/SELFRec  dataset/iFashion/{train,test}.txt
                  ("user item 1", space separated, ids 0-based contiguous)
  * tmall       : HKUDS/DCCF        data/tmall/{train_index,test_index}.pkl
                  (pickled (2,N) int32 ndarray: row0=user ids, row1=item ids)

Run:  python util/prepare_datasets_round2.py
"""
import os
import pickle
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw")


def read_space_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            pairs.append((int(parts[0]), int(parts[1])))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def read_tmall_pkl(path):
    arr = pickle.load(open(path, "rb"))
    arr = np.asarray(arr)
    if arr.shape[0] == 2:
        arr = arr.T
    return arr.astype(np.int64)


def filter_cold(train, test):
    tu = set(train[:, 0].tolist())
    ti = set(train[:, 1].tolist())
    mask = np.array([(u in tu) and (i in ti) for u, i in test], dtype=bool)
    return test[mask]


def remap_joint(train, test):
    all_pairs = np.concatenate([train, test], axis=0)
    users = np.unique(all_pairs[:, 0])
    items = np.unique(all_pairs[:, 1])
    u2id = {u: i for i, u in enumerate(users)}
    i2id = {i: j for j, i in enumerate(items)}
    remap = lambda p: np.stack(
        [np.array([u2id[u] for u in p[:, 0]]), np.array([i2id[i] for i in p[:, 1]])], axis=1)
    return remap(train), remap(test), len(users), len(items)


def write_pairs(name, pairs):
    d = os.path.join(BASE, name)
    os.makedirs(d, exist_ok=True)
    for split in ("train", "test"):
        pass
    # (kept separate writes below for clarity)
    return d


def write_split(name, split, pairs):
    path = os.path.join(BASE, name, "%s.%s" % (name, split))
    with open(path, "w") as f:
        for u, i in pairs:
            f.write("%d,%d\n" % (u, i))
    print("WROTE %s pairs=%d" % (path, len(pairs)))


def run_txt(name, train_path, test_path):
    train = read_space_pairs(train_path)
    test = read_space_pairs(test_path)
    print("== %s: raw train=%d test=%d" % (name, len(train), len(test)))
    before = len(test)
    test = filter_cold(train, test)
    print("   dropped cold test pairs: %d -> %d" % (before, len(test)))
    train, test, nu, ni = remap_joint(train, test)
    print("   %d users, %d items, train=%d test=%d" % (nu, ni, len(train), len(test)))
    os.makedirs(os.path.join(BASE, name), exist_ok=True)
    write_split(name, "train", train)
    write_split(name, "test", test)


def run_tmall(name, train_path, test_path):
    train = read_tmall_pkl(train_path)
    test = read_tmall_pkl(test_path)
    print("== %s: raw train=%d test=%d" % (name, len(train), len(test)))
    before = len(test)
    test = filter_cold(train, test)
    print("   dropped cold test pairs: %d -> %d" % (before, len(test)))
    train, test, nu, ni = remap_joint(train, test)
    print("   %d users, %d items, train=%d test=%d" % (nu, ni, len(train), len(test)))
    os.makedirs(os.path.join(BASE, name), exist_ok=True)
    write_split(name, "train", train)
    write_split(name, "test", test)


def main():
    run_txt("douban-book",
            os.path.join(RAW, "selfrec/douban-book/train.txt"),
            os.path.join(RAW, "selfrec/douban-book/test.txt"))
    run_txt("ifashion",
            os.path.join(RAW, "selfrec/iFashion/train.txt"),
            os.path.join(RAW, "selfrec/iFashion/test.txt"))
    run_tmall("tmall",
              os.path.join(RAW, "dccf/tmall_train_index.pkl"),
              os.path.join(RAW, "dccf/tmall_test_index.pkl"))
    print("DONE")


if __name__ == "__main__":
    main()

"""Prepare the full Yelp dataset (yelp_full) for this repo's pair format.

Source: scnumath ~/SGL/dataset/yelp_full/{yelp_full.train,yelp_full.test}
(pre-split "user,item" comma pairs, original non-contiguous Yelp ids).
This mirrors the protocol of prepare_datasets_round2.py:
  * keep the authors' train/test split (not a fresh 80/20)
  * drop cold-start test pairs (user/item with no train history)
  * joint remap train+test ids to 0-indexed contiguous
"""
import os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw", "yelp_full")
NAME = "yelp_full"


def load(path):
    return np.loadtxt(path, delimiter=",", dtype=np.int64).reshape(-1, 2)


def main():
    tr = load(os.path.join(RAW, "yelp_full.train"))
    te = load(os.path.join(RAW, "yelp_full.test"))
    print("raw train=%d test=%d" % (len(tr), len(te)))

    tu = set(tr[:, 0].tolist())
    ti = set(tr[:, 1].tolist())
    keep = np.array([(u in tu) and (i in ti) for u, i in te], dtype=bool)
    dropped = (~keep).sum()
    te = te[keep]
    print("dropped cold-start test pairs: %d -> %d" % (dropped, len(te)))

    all_pairs = np.concatenate([tr, te], axis=0)
    users = np.unique(all_pairs[:, 0])
    items = np.unique(all_pairs[:, 1])
    u2id = {u: i for i, u in enumerate(users)}
    i2id = {i: j for j, i in enumerate(items)}

    def remap(p):
        return np.stack([np.array([u2id[u] for u in p[:, 0]]),
                         np.array([i2id[i] for i in p[:, 1]])], axis=1)

    tr = remap(tr)
    te = remap(te)
    print("%d users, %d items, train=%d test=%d" % (len(users), len(items), len(tr), len(te)))
    print("density = %.4f%%" % (100.0 * len(tr) / (len(users) * len(items))))

    out = os.path.join(BASE, NAME)
    os.makedirs(out, exist_ok=True)
    for split, pairs in (("train", tr), ("test", te)):
        path = os.path.join(out, "%s.%s" % (NAME, split))
        with open(path, "w") as f:
            for u, i in pairs:
                f.write("%d,%d\n" % (u, i))
        print("WROTE %s pairs=%d" % (path, len(pairs)))


if __name__ == "__main__":
    main()

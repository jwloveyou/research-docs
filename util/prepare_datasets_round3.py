"""Prepare SELFRec "user: item item item" format datasets (amazon-beauty,
amazon-kindle) into this repo's pair format (dataset/<name>/<name>.{train,test},
one "user,item" comma pair per line, 0-indexed contiguous ids).

SELFRec stores some datasets as one line per user: "<uid>: <iid> <iid> ..."
(train.txt and test.txt are each a per-user positive-item list, ids 1-based).
Protocol (same as round 2): keep authors' split, binarize all listed items as
positive, drop cold-start test pairs, joint remap to 0-indexed contiguous.
"""
import os
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw", "selfrec")


def read_user_items(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, items_str = line.split(":", 1)
            u = int(u_str)
            for tok in items_str.split():
                pairs.append((u, int(tok)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def read_space_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            pairs.append((int(parts[0]), int(parts[1])))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def read_pairs_auto(path):
    """Detect 'user: item item' vs 'user item [rating]' line format."""
    with open(path) as f:
        first = f.readline()
    return read_user_items(path) if ":" in first else read_space_pairs(path)


def main():
    for name in ["amazon-beauty", "amazon-kindle"]:
        tr_path = os.path.join(RAW, name, "train.txt")
        te_path = os.path.join(RAW, name, "test.txt")
        if not (os.path.isfile(tr_path) and os.path.isfile(te_path)):
            print("SKIP %s (missing raw files)" % name)
            continue
        tr = read_pairs_auto(tr_path)
        te = read_pairs_auto(te_path)
        print("== %s: raw train=%d test=%d" % (name, len(tr), len(te)))

        tu = set(tr[:, 0].tolist())
        ti = set(tr[:, 1].tolist())
        keep = np.array([(u in tu) and (i in ti) for u, i in te], dtype=bool)
        te = te[keep]
        print("   dropped cold-start test pairs: %d left=%d" % ((~keep).sum(), len(te)))

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
        print("   %d users, %d items, train=%d test=%d, density=%.4f%%"
              % (len(users), len(items), len(tr), len(te),
                 100.0 * len(tr) / (len(users) * len(items))))

        out = os.path.join(BASE, name)
        os.makedirs(out, exist_ok=True)
        for split, pairs in (("train", tr), ("test", te)):
            path = os.path.join(out, "%s.%s" % (name, split))
            with open(path, "w") as f:
                for u, i in pairs:
                    f.write("%d,%d\n" % (u, i))
            print("   WROTE %s pairs=%d" % (path, len(pairs)))


if __name__ == "__main__":
    main()

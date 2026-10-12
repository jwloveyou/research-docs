"""Convert user-major long-list datasets (train.txt: "u i1 i2 ...") into the
pair format expected by this repo (train/.test: "user,item" one per line).
Run:  python util/convert_longlist_to_pairs.py amazon-book gowalla
"""
import os, sys

base = os.path.expanduser("~/SGL-Torch/dataset")
names = sys.argv[1:] or ["amazon-book", "gowalla"]
for name in names:
    d = os.path.join(base, name)
    for split in ("train", "test"):
        src, dst = os.path.join(d, split + ".txt"), os.path.join(d, name + "." + split)
        if not os.path.isfile(src):
            print("SKIP (missing)", src); continue
        n, umax, imax, nu = 0, -1, -1, 0
        with open(src) as fi, open(dst, "w") as fo:
            for line in fi:
                p = line.split()
                if len(p) < 2:
                    continue
                u = int(p[0]); nu += 1
                if u > umax: umax = u
                for it in p[1:]:
                    iv = int(it)
                    if iv > imax: imax = iv
                    fo.write("%d,%d\n" % (u, iv))
                n += len(p) - 1
        print("WROTE %s  pairs=%d users=%d umax=%d imax=%d" % (dst, n, nu, umax, imax))
print("DONE")

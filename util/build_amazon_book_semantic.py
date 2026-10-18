"""Build FROZEN semantic item features for amazon-book (core dataset, task #6).

Extends the component④ (frozen semantic injection, R47/R53) to one of the three
core datasets. amazon-book carries no in-repo text (only `item_list.txt` with
ASIN -> remap_id), so we stream the official McAuley-Lab Amazon Books metadata
(`meta_Books.json.gz`, 1.14 GB compressed / ~50 GB uncompressed) and extract the
title + category tags for the 91,600 ASINs in the dataset.

The resulting `dataset/amazon-book/item_sem.npy` (n_items x embed_dim, row
L2-normalised) is a static semantic prior injected into the item embeddings by
the NoiseSGL `sem_emb` switch — the exact same encoding pipeline (TF-IDF +
frozen seeded Gaussian projection) as `build_semantic_features.py`, so the
semantic vectors are comparable across datasets.

Streaming parser: ijson.kvitems over the top-level JSON dict (single pass, no
materialisation of the 50 GB uncompressed file). yajl2_c backend is used when
available for speed.

Literature: McAuley et al. (Amazon review datasets, WWW'15 / arXiv:1306.4520);
LLMRec (arXiv:2311.00423) item-attribute enhancement; TIGER (arXiv:2305.14395)
item-title semantics for cold-start.

Usage:
  python util/build_amazon_book_semantic.py [--embed_size 256]
                                            [--meta_gz dataset/_raw/meta_Books.json.gz]
"""
import os
import re
import sys
import json
import gzip
import argparse
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw")
SEED = 42

# import the shared TF-IDF + frozen-projection encoders so the semantic vectors
# stay on the exact same pipeline as ml-1m / lastfm-2k
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from build_semantic_features import build_tfidf, frozen_projection  # noqa: E402


STOPWORDS = set("""
a an and are as at be by for from has have he her his i in is it its of on or
that the their them they this to was were with you your the volume edition book
series novel books guide complete new old part ii iii iv v vi vii viii ix x
""".split())


def tokenize_title(title):
    toks = re.findall(r"[a-z0-9]+", title.lower())
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS]


def category_tags(cats):
    tags = set()
    for c in cats:
        c = c.lower().strip()
        if c:
            tags.add(c)
            # also add individual words for the finest-grained categories
            for w in re.findall(r"[a-z0-9]+", c):
                if len(w) >= 3 and w not in STOPWORDS:
                    tags.add(w)
    return sorted(tags)


def load_asin_remap(ds_dir):
    """item_list.txt: org_id(ASIN) remap_id. Return dict asin -> remap_id."""
    m = {}
    with open(os.path.join(ds_dir, "item_list.txt")) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2 and p[0] != "org_id":
                m[p[0]] = int(p[1])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_size", type=int, default=256)
    ap.add_argument("--meta_gz", default=os.path.join(RAW, "meta_Books.json.gz"))
    ap.add_argument("--ds_dir", default=os.path.join(BASE, "amazon-book"))
    args = ap.parse_args()

    asin2remap = load_asin_remap(args.ds_dir)
    n_items = max(asin2remap.values()) + 1 if asin2remap else 0
    print("[amazon-sem] n_items=%d (from item_list.txt)" % n_items,
          flush=True)

    # stream top-level {asin: {title, category, ...}} pairs
    titles = [""] * n_items
    cats = [[] for _ in range(n_items)]
    n_found = 0
    n_items_total = 0
    n_skipped = 0

    try:
        import ijson.backends.yajl2_c as yb
        backend = yb
    except Exception:
        import ijson
        backend = ijson

    opener = gzip.open if args.meta_gz.endswith(".gz") else open
    with opener(args.meta_gz, "rb") as f:
        parser = backend.kvitems(f, "")
        for asin, meta in parser:
            n_items_total += 1
            rid = asin2remap.get(asin)
            if rid is None:
                n_skipped += 1
                continue
            try:
                if isinstance(meta, dict):
                    t = meta.get("title")
                    if isinstance(t, list):
                        t = " ".join(str(x) for x in t)
                    if t:
                        titles[rid] = str(t)
                    c = meta.get("category")
                    if isinstance(c, list):
                        cats[rid] = [str(x) for x in c]
                    n_found += 1
            except Exception as e:
                n_skipped += 1
            if n_found % 10000 == 0 and n_found:
                print("[amazon-sem] scanned %d items, found %d/91600"
                      % (n_items_total, n_found), flush=True)

    print("[amazon-sem] scanned %d total, matched %d/%d, skipped %d"
          % (n_items_total, n_found, n_items, n_skipped), flush=True)

    # assemble (title, tags) per item; fallback = asin so zero rows are minimal
    texts = []
    n_titled = 0
    n_cat = 0
    for rid in range(n_items):
        t = titles[rid]
        if not t:
            # fall back to the ASIN (reverse lookup)
            asin = next((a for a, r in asin2remap.items() if r == rid), "item%d" % rid)
            t = "item %s" % asin
        else:
            n_titled += 1
        tag = tokenize_title(t) + category_tags(cats[rid])
        if cats[rid]:
            n_cat += 1
        texts.append((t, tag))

    X, vocab = build_tfidf(texts)
    P = frozen_projection(len(vocab), args.embed_size)
    item_sem = X @ P
    norm = np.linalg.norm(item_sem, axis=1, keepdims=True)
    item_sem = np.where(norm > 0, item_sem / np.maximum(norm, 1e-12), item_sem)

    np.save(os.path.join(args.ds_dir, "item_sem.npy"), item_sem.astype(np.float32))
    meta = {
        "dataset": "amazon-book",
        "n_items": n_items,
        "embed_size": args.embed_size,
        "vocab_size": len(vocab),
        "top_vocab": vocab[:40],
        "n_titled": n_titled,
        "n_with_category": n_cat,
        "zero_feature_rows": int((np.linalg.norm(item_sem, axis=1) < 1e-6).sum()),
        "projection": "frozen_seeded_gaussian_seed=%d" % SEED,
        "source": "meta_Books.json.gz (McAuley-Lab amazon_v2)",
    }
    with open(os.path.join(args.ds_dir, "item_sem_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[amazon-sem] titled=%d with_category=%d vocab=%d zero_rows=%d"
          % (n_titled, n_cat, len(vocab), meta["zero_feature_rows"]), flush=True)
    print("[amazon-sem] -> %s" % os.path.join(args.ds_dir, "item_sem.npy"),
          flush=True)


if __name__ == "__main__":
    main()

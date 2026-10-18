"""LLM (DeepSeek) item semantic-tag enrichment for task #6.

For every item in a dataset we ask the LLM to produce a few compact style/genre
tags from the item's public text (title + existing genre/tags). The result is a
resumable JSON cache {org_item_id: [tag, ...]} that util/build_semantic_features.py
merges into the frozen semantic item features.

LLM is inference-only (no fine-tuning), which respects the 'static wins' law and
the task-book LLM direction (API or local 7B, no training). Batched (<=32 items
per request) + resumable + rate-limit aware (linear backoff on 429/5xx).

Usage:
  python util/llm_enrich_items.py --dataset lastfm-2k [--max_items 1507] [--batch 25]
"""
import os
import sys
import json
import time
import argparse
import urllib.request

API = "https://api.deepseek.com/v1/chat/completions"
BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
RAW = os.path.join(BASE, "_raw")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.build_semantic_features import (  # noqa: E402
    reconstruct_org_item_ids, load_ml1m_item_text, load_lastfm_artist_tags)


def llm_call(messages, key, model="deepseek-chat", temperature=0.1, max_tokens=256):
    req = urllib.request.Request(
        API,
        data=json.dumps({"model": model, "messages": messages,
                         "max_tokens": max_tokens, "temperature": temperature}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    backoff = 1.0
    for _ in range(8):
        try:
            r = urllib.request.urlopen(req, timeout=120)
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (429, 500, 502, 503):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                continue
            raise
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
    raise RuntimeError("LLM call failed after retries")


def tags_of(ds, oid, id2text):
    title, native = id2text.get(int(oid), ("item%d" % oid, []))
    native_s = ", ".join(native) if native else "未知"
    if ds == "ml-1m":
        prompt = ("电影《%s》（已知类型：%s）。请输出 4 个最准确的子类型/风格标签"
                  "（英文小写，逗号分隔，只输出标签，不要任何解释）。" % (title, native_s))
    else:
        prompt = ("音乐人「%s」（已知标签：%s）。请输出 4 个最准确的音乐风格标签"
                  "（英文小写，逗号分隔，只输出标签，不要任何解释）。" % (title, native_s))
    return title, native, prompt


def parse_tags(resp):
    tags = []
    for part in resp.replace("\n", ",").split(","):
        t = part.strip().strip("。.、;； ").lower()
        if t and t not in tags:
            tags.append(t)
    return tags[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lastfm-2k")
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--cache", default=None, help="resumable json cache path")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("[llm] DEEPSEEK_API_KEY not set"); sys.exit(1)

    ds = args.dataset
    org_ids = reconstruct_org_item_ids(ds)
    if args.max_items:
        org_ids = org_ids[:args.max_items]
    if ds == "ml-1m":
        id2text = load_ml1m_item_text()
    else:
        id2name, id2tags = load_lastfm_artist_tags()
        id2text = {a: (id2name.get(a, "artist%d" % a), id2tags.get(a, [])) for a in org_ids}

    cache_path = args.cache or os.path.join(RAW, "llm_tags_%s.json" % ds)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = {int(k): v for k, v in json.load(f).items()}

    todo = [o for o in org_ids if int(o) not in cache]
    print("[llm] %s: total=%d done=%d todo=%d" % (ds, len(org_ids), len(cache), len(todo)))

    for s in range(0, len(todo), args.batch):
        chunk = todo[s:s + args.batch]
        prompts = [tags_of(ds, o, id2text)[2] for o in chunk]
        messages = [{"role": "user", "content":
                     "对下面 %d 个条目，每行输出一个条目对应的标签（标签间用逗号分隔，"
                     "保持条目顺序，不要编号，不要解释）：\n" % len(chunk) +
                     "\n".join("%d. %s" % (i + 1, p) for i, p in enumerate(prompts))}]
        resp = llm_call(messages, key)
        lines = [l for l in resp.strip().split("\n") if l.strip()]
        for j, oid in enumerate(chunk):
            if j < len(lines):
                # strip a possible leading "N." index
                body = lines[j].split(".", 1)[1].strip() if (len(lines[j]) > 2 and
                        lines[j][0].isdigit() and lines[j][1] in (".", "、", ")")) else lines[j].strip()
                cache[int(oid)] = parse_tags(body)
            else:
                cache[int(oid)] = []
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
        print("[llm] %s: %d/%d cached (%.0f%%)" % (ds, s + len(chunk), len(todo),
                                                   100.0 * (s + len(chunk)) / max(len(todo), 1)))
        time.sleep(1.0)
    print("[llm] DONE -> %s" % cache_path)


if __name__ == "__main__":
    main()

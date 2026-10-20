#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_final_results.py  (R84)
=====================================================================
从训练日志逐项核对并汇总「最终交付」所需的全部指标：

  1) 官方超参 light baseline（R80/R82，embed64/3层/batch2048）：
     LightGCN / SGL / SimGCL / XSimGCL  ×  {gowalla, yelp2018, amazon-book}
  2) 融合模型 embed512 3-seed（R59/R81/R82，view_mix=nn + win30 退火）：
     ×  {gowalla, yelp2018, amazon-book} ×  {seed42, seed43, seed44}
  3) 官方超参 LightGCN 300/100ep 的「欠训练」值（供诚实标注）

读取方式：
  - 从 `research_logs/r80_*_*.log` / `r81_*_*.log` / `r82_*_*.log` / `r59_*_*.log`
    直接解析 `best_result@epoch N:` 及其后一行三个浮点（precision/recall/ndcg）。
  - 从 `research_logs/r80_light_baselines_masmath_progress.txt` /
    `r81_embed512_3seed_progress.txt` / `r82_rerun_failed_progress.txt` /
    `r59_embed512_progress.txt` 取每个 run 的 rc（0=成功，非 0=失败）与 START/END 状态。
  - 状态判定：rc=0 且有 best_result => DONE；rc!=0 => FAILED；已 START 无 END => RUNNING；
    未出现在任何 progress 中 => MISSING（排队未跑）。

输出：
  - `research_logs/final_results.json`（机器可读，含每个 run 的 epoch/precision/recall/ndcg/status）
  - stdout 打印两张 markdown 表 + 一张 seed 表，可直接粘贴进 研究总报告.md §2.3 / §2.1。

用法：
  /home/Yehaikun/miniconda3/envs/sgl/bin/python util/collect_final_results.py
  或  python3 util/collect_final_results.py
=====================================================================
"""
import os
import re
import json
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(ROOT, "research_logs")


# ----------------------------------------------------------------------
# 1. 基础解析
# ----------------------------------------------------------------------
def extract_best(logpath):
    """从训练日志提取 best_result@epoch N 与 (precision, recall, ndcg)。"""
    if not os.path.exists(logpath):
        return None
    best_epoch = None
    values = None
    try:
        with open(logpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None
    for i, line in enumerate(lines):
        m = re.search(r"best_result@epoch\s+(\d+)", line)
        if m:
            best_epoch = int(m.group(1))
            # 下一非空行应为三个浮点（先剥掉 `2026-10-20 04:35:30.268:` 时间戳前缀）
            for j in range(i + 1, min(i + 4, len(lines))):
                cand = lines[j].strip()
                cand = re.sub(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+\s*:\s*", "", cand)
                nums = re.findall(r"[+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?", cand)
                if len(nums) >= 3:
                    try:
                        values = tuple(float(x) for x in nums[:3])
                        break
                    except ValueError:
                        continue
            break
    if values is None:
        return {"epoch": best_epoch, "status": "NO_VALUES"}
    return {"epoch": best_epoch, "precision": values[0], "recall": values[1], "ndcg": values[2]}


def parse_progress(progress_path):
    """解析 progress 文件，返回 {key -> {'rc': int|None, 'state': 'STARTED'|'ENDED'}}。

    key 形如 `gowalla_lightgcn`（tag 在前或后都可能，这里统一用「dataset_tag」）。
    """
    out = {}
    if not os.path.exists(progress_path):
        return out
    with open(progress_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m_start = re.search(r"^START\s+([\w.-]+)\s+", line)
            m_end = re.search(r"^END\s+([\w.-]+)\s+rc=(-?\d+)", line)
            if m_start:
                key = m_start.group(1)
                out.setdefault(key, {"rc": None, "state": "STARTED"})
                out[key]["state"] = "STARTED"
            elif m_end:
                key = m_end.group(1)
                rc = int(m_end.group(2))
                out.setdefault(key, {"rc": None, "state": "STARTED"})
                out[key]["rc"] = rc
                out[key]["state"] = "ENDED"
    return out


def status_of(run_key, progress_map, best):
    info = progress_map.get(run_key)
    if info is None:
        return "MISSING"
    if info.get("state") == "STARTED" and info.get("rc") is None:
        return "RUNNING"
    if info.get("rc") == 0 and best is not None and "precision" in best:
        return "DONE"
    if info.get("rc") not in (None, 0):
        return "FAILED"
    if best is None:
        return "FAILED"
    return "DONE"


# ----------------------------------------------------------------------
# 2. 注册表：所有最终交付需要的 run
# ----------------------------------------------------------------------
# (source, key, tag, dataset, seed)
#  key 用于 progress 映射；log 名 = f"{source}_{dataset}_{tag}.log"
BASELINE_RUNS = [
    # ---- R80 light baselines（官方超参）----
    ("r80", "gowalla_lightgcn",    "lightgcn", "gowalla",    None),
    ("r80", "yelp2018_lightgcn",   "lightgcn", "yelp2018",   None),
    ("r80", "amazon-book_lightgcn","lightgcn", "amazon-book", None),
    ("r80", "gowalla_sgl",         "sgl",      "gowalla",    None),
    ("r80", "yelp2018_sgl",        "sgl",      "yelp2018",   None),
    ("r80", "amazon-book_sgl",     "sgl",      "amazon-book", None),
    ("r80", "gowalla_simgcl",      "simgcl",   "gowalla",    None),
    ("r80", "yelp2018_simgcl",     "simgcl",   "yelp2018",   None),
    ("r80", "amazon-book_simgcl",  "simgcl",   "amazon-book", None),
    ("r80", "gowalla_xsimgcl",     "xsimgcl",  "gowalla",    None),
    ("r80", "yelp2018_xsimgcl",    "xsimgcl",  "yelp2018",   None),
    ("r80", "amazon-book_xsimgcl", "xsimgcl",  "amazon-book", None),
    # ---- R82 rerun（官方超参，覆盖 r80 失败项）----
    ("r82", "yelp2018_simgcl",     "simgcl",   "yelp2018",   None),
    ("r82", "amazon-book_simgcl",  "simgcl",   "amazon-book", None),
    ("r82", "gowalla_xsimgcl",     "xsimgcl",  "gowalla",    None),
    ("r82", "yelp2018_xsimgcl",    "xsimgcl",  "yelp2018",   None),
    ("r82", "amazon-book_xsimgcl", "xsimgcl",  "amazon-book", None),
    ("r82", "amazon-book_lightgcn","lightgcn", "amazon-book", None),
]

EMB512_RUNS = [
    # ---- R59 seed42（融合 embed512 单 seed，已定稿）----
    ("r59", "gowalla_emb512",    "emb512", "gowalla",    42),
    ("r59", "yelp2018_emb512",   "emb512", "yelp2018",   42),
    ("r59", "amazon-book_emb512","emb512", "amazon-book", 42),
    # ---- R81 seed43/44（3-seed 补全）----
    ("r81", "gowalla_emb512_s43",    "emb512_s43", "gowalla",    43),
    ("r81", "gowalla_emb512_s44",    "emb512_s44", "gowalla",    44),
    ("r81", "yelp2018_emb512_s43",   "emb512_s43", "yelp2018",   43),
    ("r81", "yelp2018_emb512_s44",   "emb512_s44", "yelp2018",   44),
    ("r81", "amazon-book_emb512_s43","emb512_s43", "amazon-book", 43),
    ("r81", "amazon-book_emb512_s44","emb512_s44", "amazon-book", 44),
    # ---- R82 rerun amazon s44（3-seed 最后一块）----
    ("r82", "amazon-book_emb512_s44","emb512_s44", "amazon-book", 44),
]

FUSION_SEED42 = {
    # 从 R59 日志直接核对的融合 embed512 seed42 终值（prec/recall/ndcg）
    "gowalla":    {"recall": 0.1894, "ndcg": 0.1596, "epoch": 29},
    "yelp2018":   {"recall": 0.0716, "ndcg": 0.0587, "epoch": 4},
    "amazon-book": {"recall": 0.0589, "ndcg": 0.0455, "epoch": 15},
}


def build_registry():
    baseline_progress = parse_progress(os.path.join(LOGDIR, "r80_light_baselines_masmath_progress.txt"))
    r82_progress = parse_progress(os.path.join(LOGDIR, "r82_rerun_failed_progress.txt"))
    r81_progress = parse_progress(os.path.join(LOGDIR, "r81_embed512_3seed_progress.txt"))
    r59_progress = parse_progress(os.path.join(LOGDIR, "r59_embed512_progress.txt"))

    # 合并 progress：r82 会覆盖 r80 的同名 key（以较新者为准）
    combined = {}
    for pm in (baseline_progress, r59_progress, r81_progress, r82_progress):
        for k, v in pm.items():
            # r82 里 key 与 r80 同名（dataset_tag），后写入者覆盖
            combined[k] = v

    results = {}
    for source, key, tag, ds, seed in BASELINE_RUNS + EMB512_RUNS:
        logpath = os.path.join(LOGDIR, f"{source}_{ds}_{tag}.log")
        best = extract_best(logpath)
        st = status_of(key, combined, best)
        results[key] = {
            "source": source, "tag": tag, "dataset": ds, "seed": seed,
            "log": os.path.relpath(logpath, ROOT),
            "status": st, "best": best,
        }
    return results


# ----------------------------------------------------------------------
# 3. 输出
# ----------------------------------------------------------------------
def fmt(best, nd=4):
    if not best or "recall" not in best:
        return "—"
    return f"{best['recall']:.{nd}f}/{best['ndcg']:.{nd}f}"


def main():
    results = build_registry()

    # 选择「每个 (dataset, tag) 的最佳 source」：r82 优先于 r80（rerun 覆盖）
    def pick(ds, tag):
        cands = [k for k, r in results.items()
                 if r["dataset"] == ds and r["tag"] == tag]
        # r82 优先
        order = {"r82": 0, "r80": 1, "r81": 2, "r59": 3}
        cands.sort(key=lambda k: order.get(results[k]["source"], 9))
        return results[cands[0]] if cands else None

    datasets = ["gowalla", "yelp2018", "amazon-book"]
    baseline_tags = ["lightgcn", "sgl", "simgcl", "xsimgcl"]

    print("# 官方超参 baseline 终值（R@20/NDCG@20，embed64/3层/batch2048，seed42）\n")
    header = "| 数据集 | LightGCN | SGL | SimGCL | XSimGCL | 融合 embed512(seed42) |"
    print(header)
    print("|---|---|---|---|---|---|")
    md_lines = [header, "|---|---|---|---|---|---|"]
    for ds in datasets:
        row = [f"**{ds}**"]
        for tag in baseline_tags:
            r = pick(ds, tag)
            if r is None:
                row.append("—")
            elif r["status"] == "DONE":
                row.append(f"{fmt(r['best'])}@ep{r['best']['epoch']}")
            elif r["status"] == "RUNNING":
                row.append("⏳ running")
            else:
                row.append(f"❌ {r['status'].lower()}")
        row.append(f"**{fmt(FUSION_SEED42[ds])}**@ep{FUSION_SEED42[ds]['epoch']}")
        line = "| " + " | ".join(row) + " |"
        print(line)
        md_lines.append(line)

    print("\n# 融合 embed512 3-seed（R@20/NDCG@20，view_mix=nn + win30 退火）\n")
    print("| 数据集 | seed42 | seed43 | seed44 | 3-seed 均值 |")
    print("|---|---|---|---|---|")
    md_lines.append("")
    md_lines.append("# 融合 embed512 3-seed（R@20/NDCG@20，view_mix=nn + win30 退火）")
    md_lines.append("| 数据集 | seed42 | seed43 | seed44 | 3-seed 均值 |")
    md_lines.append("|---|---|---|---|---|")
    for ds in datasets:
        seeds = {42: FUSION_SEED42[ds]}
        for k, r in results.items():
            if r["dataset"] == ds and r["tag"].startswith("emb512") and r["seed"] in (43, 44):
                seeds[r["seed"]] = r["best"]
        cells = []
        for s in (42, 43, 44):
            b = seeds.get(s)
            if b is None:
                cells.append("—")
            elif isinstance(b, dict) and "recall" in b:
                cells.append(f"{fmt(b)}@ep{b['epoch']}")
            else:
                cells.append("—")
        recalls = [seeds[s]["recall"] for s in (42, 43, 44)
                   if s in seeds and isinstance(seeds[s], dict) and "recall" in seeds[s]]
        ndcgs = [seeds[s]["ndcg"] for s in (42, 43, 44)
                 if s in seeds and isinstance(seeds[s], dict) and "ndcg" in seeds[s]]
        mean_cell = "—"
        if len(recalls) == 3:
            mean_cell = f"{sum(recalls)/3:.5f}/{sum(ndcgs)/3:.5f}"
        elif recalls:
            mean_cell = f"{sum(recalls)/len(recalls):.5f}/{sum(ndcgs)/len(ndcgs):.5f} ({len(recalls)}-seed)"
        line = "| " + " | ".join([f"**{ds}**"] + cells + [mean_cell]) + " |"
        print(line)
        md_lines.append(line)

    # 落盘 JSON 与 markdown
    outpath = os.path.join(LOGDIR, "final_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({"runs": results, "fusion_seed42": FUSION_SEED42}, f,
                  ensure_ascii=False, indent=2)
    mdpath = os.path.join(LOGDIR, "final_results_tables.md")
    with open(mdpath, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n[JSON] {outpath}")
    print(f"[MD]   {mdpath}")

    # 报告补丁候选：把 研究总报告.md §2.3/§2.1/§3.2 的「重跑中」占位符替换为终值。
    # 仅对 DONE 的 run 输出 old->new，保证「跑出什么就替换什么」，不凭空造值。
    # 占位符单元显式登记（只有这些是 R82 待补的空白格；其余格已有终值）。
    # 注：yelp2018 SimGCL 已在 R85 手工回填为 0.0676/0.0556@ep14，故不再登记该占位符。
    PLACEHOLDER = {
        ("gowalla", "xsimgcl"):    "重跑中（R82）",
        ("yelp2018", "xsimgcl"):   "重跑中（R82）",
        ("amazon-book", "lightgcn"): "重跑中（R82，10-20 OOM）",
        ("amazon-book", "simgcl"): "重跑中（R82）",
        ("amazon-book", "xsimgcl"): "重跑中（R82）",
    }
    print("\n# REPORT PATCH CANDIDATES（研究总报告.md 占位符 -> 终值；仅 DONE 项）\n")
    patch = {}
    for (ds, tag), old in PLACEHOLDER.items():
        r = pick(ds, tag)
        if r is None or r["status"] != "DONE":
            continue
        new = f"{fmt(r['best'])}@ep{r['best']['epoch']}"
        patch[(ds, tag)] = (old, new)
        print(f"  §2.3 [{ds}][{tag}]  {old}  ->  {new}")

    # amazon embed512 s44（补齐 §2.1 与 §3.2 的 3-seed）
    am_s44 = results.get("amazon-book_emb512_s44")
    if am_s44 is not None and am_s44["status"] == "DONE" and "recall" in am_s44["best"]:
        s42 = FUSION_SEED42["amazon-book"]
        s43 = results.get("amazon-book_emb512_s43", {}).get("best") or {}
        s44 = am_s44["best"]
        rec = [s42["recall"], s43.get("recall"), s44["recall"]]
        ndc = [s42["ndcg"], s43.get("ndcg"), s44["ndcg"]]
        if all(v is not None for v in rec) and all(v is not None for v in ndc):
            mean = f"{sum(rec)/3:.5f}/{sum(ndc)/3:.5f}"
            # 核心子串（§2.1 与 §3.2 各出现一次，前缀分别为 `amazon ` / `**`）——
            # 只替换核心子串，保留两处各自的前后缀。
            old_21 = "0.0589/0.0588/（s44 在途，2-seed 均值 0.05885/0.04545）"
            new_21 = (f"{s42['recall']:.4f}/{s43['recall']:.4f}/"
                      f"{s44['recall']:.4f}（均值 {mean}）")
            print(f"  §2.1/§3.2 [amazon embed512 s44]\n      {old_21}\n      ->  {new_21}")
            patch[("amazon-book", "emb512_s44")] = (old_21, new_21)

    # 落盘 patch 候选（供 patch_final_report.py 或人工核用）
    patchpath = os.path.join(LOGDIR, "final_report_patch.json")
    with open(patchpath, "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}|{k[1]}": list(v) for k, v in patch.items()}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[PATCH] {patchpath} ({len(patch)} replacements)")


if __name__ == "__main__":
    main()

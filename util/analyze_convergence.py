#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收敛性分析（任务#2「收敛性分析/早停行为」缺口补齐）
从训练日志提取 epoch -> (loss, R@20, NDCG@20) 轨迹，计算：
  1) epoch-to-peak（达峰轮次 = 收敛速度）
  2) peak 值
  3) 过拟合率 = (peak - final) / peak
  4) 收敛加速比 = 慢模型达峰轮次 / 快模型达峰轮次
  5) time-to-threshold：达到参照模型 peak 的 X% 所需轮次

用法：
  python util/analyze_convergence.py --log <label>:<file> [--log ...] \
      --ref <label> --threshold 0.95 --out research_logs/convergence.json
"""
import argparse
import json
import re
import sys

EPOCH_RE = re.compile(r'epoch\s+(\d+)\s*:\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)')
BEST_RE = re.compile(r'best_result@epoch\s+(\d+)')


def parse_log(path):
    """返回 (epochs: list[int], loss: list, rec: list, ndcg: list, best_ep: int, best: dict)"""
    epochs, loss, rec, ndcg = [], [], [], []
    best_ep, best = None, None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        m = EPOCH_RE.search(ln)
        if m:
            epochs.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            rec.append(float(m.group(3)))
            ndcg.append(float(m.group(4)))
            continue
        b = BEST_RE.search(ln)
        if b:
            best_ep = int(b.group(1))
            # 下一非空行含 "loss R NDCG"
            for j in range(i + 1, min(i + 4, len(lines))):
                cand = lines[j].strip()
                nums = re.findall(r'([\d.]+)', cand)
                if len(nums) >= 3 and not EPOCH_RE.search(cand):
                    try:
                        best = {'loss': float(nums[0]), 'rec': float(nums[1]), 'ndcg': float(nums[2])}
                    except ValueError:
                        best = None
                    break
            continue
    # 若 best_ep 缺失，用峰值轮次兜底
    if best_ep is None and rec:
        best_ep = epochs[rec.index(max(rec))]
        best = {'loss': loss[best_ep - epochs[0]] if False else None,
                'rec': max(rec), 'ndcg': max(ndcg)}
    return epochs, loss, rec, ndcg, best_ep, best


def metrics(label, ep, rec, ndcg, best_ep, best):
    if not ep:
        return {'label': label, 'n_epochs': 0, 'error': 'no data'}
    peak_rec = max(rec)
    peak_ndcg = max(ndcg)
    peak_ep = ep[rec.index(peak_rec)]
    final_rec = rec[-1]
    final_ndcg = ndcg[-1]
    # 过拟合率：峰后相对峰值跌落（final 可能已回升到平台，取峰后最低点更敏感）
    if peak_ep < ep[-1]:
        tail_rec = rec[rec.index(peak_rec):]
        min_tail_rec = min(tail_rec)
        overfit = (peak_rec - min_tail_rec) / peak_rec if peak_rec > 0 else 0.0
    else:
        overfit = 0.0
    return {
        'label': label,
        'n_epochs': len(ep),
        'best_ep': best_ep if best_ep is not None else peak_ep,
        'peak_ep': peak_ep,
        'peak_rec': peak_rec,
        'peak_ndcg': peak_ndcg,
        'final_rec': final_rec,
        'final_ndcg': final_ndcg,
        'overfit_rate': overfit,
        'best': best,
    }


def time_to_threshold(ep, rec, target):
    """达到 target 值所需的轮次（若从未达到返回 None）"""
    for e, r in zip(ep, rec):
        if r >= target:
            return e
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', action='append', default=[], help='label:path')
    ap.add_argument('--ref', default=None, help='参照模型 label（计算 time-to-threshold 与加速比）')
    ap.add_argument('--threshold', type=float, default=0.95)
    ap.add_argument('--out', default='research_logs/convergence.json')
    args = ap.parse_args()

    results = []
    traj = {}
    for spec in args.log:
        label, path = spec.split(':', 1)
        ep, loss, rec, ndcg, best_ep, best = parse_log(path)
        m = metrics(label, ep, rec, ndcg, best_ep, best)
        results.append(m)
        traj[label] = (ep, rec, ndcg)
        print(f'[{label}] n_ep={m["n_epochs"]} best_ep={m["best_ep"]} '
              f'peak@ep{m["peak_ep"]} R@20={m["peak_rec"]:.4f} NDCG={m["peak_ndcg"]:.4f} '
              f'final R@20={m["final_rec"]:.4f} overfit={m["overfit_rate"]*100:.1f}%')

    # 参照模型对照
    if args.ref:
        ref_m = next((r for r in results if r['label'] == args.ref), None)
        if ref_m and ref_m.get('best_ep'):
            ref_target_rec = ref_m['peak_rec'] * args.threshold
            ref_target_ndcg = ref_m['peak_ndcg'] * args.threshold
            print(f'\n参照模型 {args.ref}: peak R@20={ref_m["peak_rec"]:.4f}@ep{ref_m["best_ep"]} '
                  f'NDCG={ref_m["peak_ndcg"]:.4f}')
            for r in results:
                if r['label'] == args.ref or not r.get('best_ep'):
                    continue
                ep_i, rec_i, _ = traj[r['label']]
                # 达到参照模型 peak 的 X% 所需轮次（R@20 口径）
                t_rec = time_to_threshold(ep_i, rec_i, ref_target_rec)
                accel = ref_m['best_ep'] / r['best_ep'] if r['best_ep'] else None
                r['time_to_ref_frac'] = t_rec
                r['accel_vs_' + args.ref] = accel
                print(f'  vs {r["label"]}: 加速比={ref_m["best_ep"]}/{r["best_ep"]} = {accel:.1f}×, '
                      f'达到 ref peak {args.threshold*100:.0f}% 需 {t_rec} 轮')

    # 输出 JSON（含完整轨迹）
    out = {'results': results, 'trajectories': {}}
    for spec in args.log:
        label, path = spec.split(':', 1)
        ep, loss, rec, ndcg, best_ep, best = parse_log(path)
        out['trajectories'][label] = {
            'epoch': ep, 'loss': loss, 'rec': rec, 'ndcg': ndcg,
            'best_ep': best_ep, 'best': best,
        }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n轨迹 JSON 已写 {args.out}')


if __name__ == '__main__':
    main()

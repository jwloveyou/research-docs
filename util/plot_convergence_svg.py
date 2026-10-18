#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 Python 生成收敛曲线 SVG（无需 matplotlib），用于论文可视化（任务#2「收敛性分析」）。

用法：python util/plot_convergence_svg.py --json research_logs/convergence.json \
      --out research_logs/convergence_gowalla.svg --labels a,b,c --xmax 300
"""
import argparse
import json


def parse_labels(s):
    return [x for x in s.split(',') if x]


def make_svg(traj, labels, out, title, xmax=None, w=720, h=420):
    # 数据范围
    xs_all, ys_all = [], []
    for lb in labels:
        ep, rec = traj[lb]['epoch'], traj[lb]['rec']
        xs_all += ep
        ys_all += rec
    if xmax is None:
        xmax = max(xs_all)
    ymin, ymax = min(ys_all), max(ys_all)
    pad_y = (ymax - ymin) * 0.08 + 1e-9
    ymin = max(0, ymin - pad_y)
    ymax = ymax + pad_y
    pad_x = xmax * 0.03
    L, R, T, B = 70, w - 20, 30, h - 55
    sx = (R - L) / xmax
    sy = (B - T) / (ymax - ymin)

    def X(x):
        return L + x * sx

    def Y(y):
        return B - (y - ymin) * sy

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']
    strokes = []
    for i, lb in enumerate(labels):
        ep, rec = traj[lb]['epoch'], traj[lb]['rec']
        pts = ' '.join(f'{X(e):.1f},{Y(r):.1f}' for e, r in zip(ep, rec))
        c = colors[i % len(colors)]
        strokes.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"/>')

    # 坐标轴 + 网格
    grid = ''
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        grid += f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{R}" y2="{Y(yv):.1f}" stroke="#eee" stroke-width="1"/>'
        grid += f'<text x="{L-8}" y="{Y(yv)+4:.1f}" font-size="11" text-anchor="end" fill="#555">{yv:.3f}</text>'
    for i in range(7):
        xv = xmax * i / 6
        grid += f'<line x1="{X(xv):.1f}" y1="{T}" x2="{X(xv):.1f}" y2="{B}" stroke="#eee" stroke-width="1"/>'
        grid += f'<text x="{X(xv):.1f}" y="{B+18:.1f}" font-size="11" text-anchor="middle" fill="#555">{xv:.0f}</text>'

    # 图例
    legend = ''
    for i, lb in enumerate(labels):
        c = colors[i % len(colors)]
        legend += (f'<rect x="{L + 10 + i*190}" y="{T+2}" width="14" height="10" fill="{c}"/>'
                   f'<text x="{L + 28 + i*190}" y="{T+11}" font-size="11" fill="#333">{lb}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="{w}" height="{h}" fill="white"/>
<text x="{L}" y="{T-8}" font-size="14" font-weight="bold" fill="#222">{title}</text>
<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#333" stroke-width="1.5"/>
<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="#333" stroke-width="1.5"/>
<text x="{(L+R)/2:.1f}" y="{h-18}" font-size="13" text-anchor="middle" fill="#333">Epoch</text>
<text x="16" y="{(T+B)/2:.1f}" font-size="13" text-anchor="middle" fill="#333" transform="rotate(-90 16 {(T+B)/2:.1f})">Recall@20</text>
{grid}
{''.join(strokes)}
{legend}
</svg>'''
    with open(out, 'w') as f:
        f.write(svg)
    print(f'SVG 已写 {out}（{len(labels)} 条曲线，xmax={xmax:.0f}）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--title', default='Convergence curves (Recall@20)')
    ap.add_argument('--xmax', type=float, default=None)
    args = ap.parse_args()
    d = json.load(open(args.json))
    labels = parse_labels(args.labels)
    make_svg(d['trajectories'], labels, args.out, args.title, args.xmax)


if __name__ == '__main__':
    main()

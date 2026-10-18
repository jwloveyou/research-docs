#!/usr/bin/env python3
"""
Pure-Python SVG plotter for the bottom-level visualisation evidence (task #2).

Reads two viz JSONs produced by research_logs/analyze_embedding_viz.py (one for
NoiseSGL, one for pure LightGCN) and renders three figures as self-contained SVG:

  1. over-smoothing curve       : layer depth (0..L) vs mean pairwise cosine (user emb.)
  2. spectral energy distribution: singular-value index vs log10(energy fraction)
  3. PCA 2-D scatter             : user vs item embedding projection (two panels)

No matplotlib dependency; coordinates are drawn directly as SVG <path>/<circle>.

Usage:
  python util/plot_embedding_viz.py \
      --noise research_logs/viz_gowalla_noise.json \
      --lgn   research_logs/viz_gowalla_lgn.json \
      --out   research_logs/embedding_viz_gowalla.svg
"""
import os
import sys
import json
import math
import argparse

W = 860
H1 = 300
H2 = 300
H3 = 360
PAD = 56


def _svg_header(w, h):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d">\n'
            '<rect width="100%%" height="100%%" fill="#ffffff"/>\n'
            '<style>text{font-family:DejaVu Sans,Helvetica,Arial,sans-serif;font-size:13px;}'
            '.axis{stroke:#999;stroke-width:1;}'
            '.grid{stroke:#eee;stroke-width:1;}'
            '.noise{stroke:#c0392b;stroke-width:2;fill:none;}'
            '.lgn{stroke:#2980b9;stroke-width:2;fill:none;}'
            '.noise-dot{fill:#c0392b;} .lgn-dot{fill:#2980b9;}'
            '.user{fill:#e67e22;fill-opacity:0.55;} .item{fill:#3498db;fill-opacity:0.55;}'
            '</style>\n' % (w, h, w, h))


def _frame(y, title, xlabel):
    return ('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#333"/>'
            '<text x="%d" y="%d" font-weight="bold" font-size="15">%s</text>'
            '<text x="%d" y="%d" text-anchor="middle" font-size="13">%s</text>'
            % (PAD, y, W - 2 * PAD, H1 - 40, PAD, y - 14, title, W // 2, y + H1 - 40 + 24, xlabel))


def _line_xy(xs, ys, x0, x1, y0, y1):
    px = [x0 + (x - xs[0]) / (xs[-1] - xs[0] + 1e-12) * (x1 - x0) for x in xs]
    py = [y1 - (y - y0) / (y1 - y0 + 1e-12) * (y1 - y0) for y in ys]
    return px, py


def _polyline(px, py, cls):
    d = "M " + " L ".join("%.1f,%.1f" % (x, y) for x, y in zip(px, py))
    return '<path class="%s" d="%s"/>' % (cls, d)


def _scatter(pts, labels, x0, x1, y0, y1):
    out = []
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    for (x, y), lab in zip(pts, labels):
        px = x0 + (x - xmin) / (xmax - xmin + 1e-12) * (x1 - x0)
        py = y1 - (y - ymin) / (ymax - ymin + 1e-12) * (y1 - y0)
        cls = "user" if lab == "user" else "item"
        out.append('<circle cx="%.1f" cy="%.1f" r="1.2" class="%s"/>' % (px, py, cls))
    return "".join(out)


def _plot_curve(svg, y, title, xlabel, ylabel, xs, ys_noise, ys_lgn, ymax=None):
    svg.append(_frame(y, title, xlabel))
    x0, x1 = PAD + 6, W - PAD - 6
    y0, y1 = y + 8, y + H1 - 40 - 8
    if ymax is None:
        ymax = max(max(ys_noise), max(ys_lgn)) * 1.15
    ymin = min(min(ys_noise), min(ys_lgn)) * 0.9
    # grid + axis ticks
    for i in range(5):
        gy = y0 + (y1 - y0) * i / 4
        svg.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (x0, gy, x1, gy))
        svg.append('<text x="%d" y="%.1f" font-size="11" fill="#666">%.3f</text>'
                   % (x0 - 8, gy + 4, ymax - (ymax - ymin) * i / 4))
    px_n, py_n = _line_xy(xs, ys_noise, x0, x1, ymin, ymax)
    px_l, py_l = _line_xy(xs, ys_lgn, x0, x1, ymin, ymax)
    svg.append(_polyline(px_n, py_n, "noise"))
    svg.append(_polyline(px_l, py_l, "lgn"))
    for x, yn, yl in zip(xs, ys_noise, ys_lgn):
        pxn, pyn = _line_xy([x], [yn], x0, x1, ymin, ymax)
        pxl, pyl = _line_xy([x], [yl], x0, x1, ymin, ymax)
        svg.append('<circle cx="%.1f" cy="%.1f" r="3" class="noise-dot"/>' % (pxn[0], pyn[0]))
        svg.append('<circle cx="%.1f" cy="%.1f" r="3" class="lgn-dot"/>' % (pxl[0], pyl[0]))
        svg.append('<text x="%.1f" y="%d" text-anchor="middle" font-size="11" fill="#666">%d</text>'
                   % (pxn[0], y1 + 16, x))
    # legend
    svg.append('<text x="%d" y="%d" fill="#c0392b">NoiseSGL</text>' % (x1 - 150, y + 20))
    svg.append('<text x="%d" y="%d" fill="#2980b9">LightGCN</text>' % (x1 - 150, y + 38))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", required=True)
    ap.add_argument("--lgn", required=True)
    ap.add_argument("--out", default="research_logs/embedding_viz.svg")
    args = ap.parse_args()

    with open(args.noise) as f:
        N = json.load(f)
    with open(args.lgn) as f:
        L = json.load(f)

    nlayers = N["n_layers"]
    xs = list(range(nlayers + 1))
    svg = [_svg_header(W, H1 + H2 + 2 * H3 + 3 * PAD)]

    # figure 1: over-smoothing curve
    _plot_curve(svg, PAD + 20, "Over-smoothing vs layer depth (user embeddings)",
                "layer k", "mean pairwise cosine",
                xs, N["layer_smooth"], L["layer_smooth"])

    # figure 2: spectral energy distribution
    sv_n = N["spectrum"]["sv"]; sv_l = L["spectrum"]["sv"]
    # energy fraction per singular value (normalised by total energy)
    tot_n = sum(v * v for v in sv_n) + 1e-12
    tot_l = sum(v * v for v in sv_l) + 1e-12
    en_n = [math.log10(v * v / tot_n + 1e-12) for v in sv_n]
    en_l = [math.log10(v * v / tot_l + 1e-12) for v in sv_l]
    sx = list(range(1, len(sv_n) + 1))
    svg.append(_frame(PAD + H1 + 30, "Spectral energy distribution (top-64 singular values)",
                      "singular value index"))
    x0, x1 = PAD + 6, W - PAD - 6
    y0, y1 = PAD + H1 + 30 + 8, PAD + H1 + 30 + H2 - 40 - 8
    emax = max(max(en_n), max(en_l)); emin = min(min(en_n), min(en_l))
    for i in range(5):
        gy = y0 + (y1 - y0) * i / 4
        svg.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (x0, gy, x1, gy))
        svg.append('<text x="%d" y="%.1f" font-size="11" fill="#666">%.1f</text>'
                   % (x0 - 8, gy + 4, emax - (emax - emin) * i / 4))
    px_n, py_n = _line_xy(sx, en_n, x0, x1, emin, emax)
    px_l, py_l = _line_xy(sx, en_l, x0, x1, emin, emax)
    svg.append(_polyline(px_n, py_n, "noise"))
    svg.append(_polyline(px_l, py_l, "lgn"))
    svg.append('<text x="%d" y="%d" fill="#c0392b">NoiseSGL (erank=%.1f)</text>'
               % (x1 - 200, y0 + 18, N["spectrum"]["eff_rank"]))
    svg.append('<text x="%d" y="%d" fill="#2980b9">LightGCN (erank=%.1f)</text>'
               % (x1 - 200, y0 + 36, L["spectrum"]["eff_rank"]))

    # figure 3: PCA scatter (two panels side by side)
    y = PAD + H1 + H2 + 50
    svg.append('<text x="%d" y="%d" font-weight="bold" font-size="15">PCA 2-D projection '
               '(orange=user, blue=item)</text>' % (PAD, y - 14))
    panel_w = (W - 2 * PAD - 20) // 2
    for pi, (dat, title) in enumerate([(N, "NoiseSGL"), (L, "LightGCN")]):
        bx0 = PAD + pi * (panel_w + 20)
        bx1 = bx0 + panel_w
        by0, by1 = y + 8, y + H3 - 20
        svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#333"/>'
                   % (bx0, by0, panel_w, by1 - by0))
        svg.append('<text x="%d" y="%d" text-anchor="middle">%s</text>'
                   % ((bx0 + bx1) // 2, by1 + 16, title))
        svg.append(_scatter(dat["pca"]["xy"], dat["pca"]["labels"],
                            bx0 + 4, bx1 - 4, by0 + 4, by1 - 4))

    # figure 4: t-SNE scatter (two panels side by side)
    y2 = y + H3 + 10
    svg.append('<text x="%d" y="%d" font-weight="bold" font-size="15">t-SNE 2-D projection '
               '(orange=user, blue=item)</text>' % (PAD, y2 - 14))
    for pi, (dat, title) in enumerate([(N, "NoiseSGL"), (L, "LightGCN")]):
        bx0 = PAD + pi * (panel_w + 20)
        bx1 = bx0 + panel_w
        by0, by1 = y2 + 8, y2 + H3 - 20
        svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#333"/>'
                   % (bx0, by0, panel_w, by1 - by0))
        svg.append('<text x="%d" y="%d" text-anchor="middle">%s</text>'
                   % ((bx0 + bx1) // 2, by1 + 16, title))
        svg.append(_scatter(dat["tsne"]["xy"], dat["tsne"]["labels"],
                            bx0 + 4, bx1 - 4, by0 + 4, by1 - 4))
    svg.append("</svg>\n")

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("".join(svg))
    print("SAVED", out)


if __name__ == "__main__":
    main()

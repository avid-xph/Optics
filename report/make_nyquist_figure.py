"""PSNR-vs-cell curves for the Nyquist study (task 6).

One panel per sigma (blur); one curve per shift (misregistration) per method.
The optimal cell is marked on each curve.

  python report/make_nyquist_figure.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "ablation")
OUT = os.path.join(ROOT, "report", "figures")

SIGMAS = [0, 0.5, 1.0]
SHIFTS = [0, 0.25, 0.5]
METHODS = ["admm_tv", "tfree"]
LABEL = {"admm_tv": "ADMM-TV", "tfree": "T-free"}
# categorical slots 1-4, fixed order; marker as secondary encoding
COLOR = {"admm_tv": "#eb6834", "tfree": "#2a78d6"}
MARKER = {"admm_tv": "s", "tfree": "o"}
DASH = {0: "-", 0.25: (0, (5, 2)), 0.5: (0, (1.5, 1.5))}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BG = "#fcfcfb"


def load(sigma, shift):
    fn = os.path.join(RES, f"nyquist_s{sigma}_d{shift}.csv")
    if not os.path.exists(fn):
        return {}
    per = defaultdict(lambda: defaultdict(list))    # method -> cell -> scene -> psnr
    with open(fn, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["err"] or r["method"] not in METHODS:
                continue
            per[r["method"]][int(r["cell"])].append((r["scene"], float(r["psnr"])))
    out = {}
    for m, cells in per.items():
        d = {}
        for c, rows in cells.items():
            by_scene = defaultdict(list)
            for sc, p in rows:
                by_scene[sc].append(p)
            if len(by_scene) == 6:
                d[c] = sum(sum(v) / len(v) for v in by_scene.values()) / 6
        out[m] = d
    return out


def main():
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(11.0, 3.7), dpi=200,
                             sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, sigma in zip(axes, SIGMAS):
        ax.set_facecolor(BG)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xticks([1, 2, 3, 4, 6, 8])
        ax.set_xlim(0.7, 8.6)
        ax.set_xlabel("Coding cell size c (CCD pixels)", color=INK, fontsize=9)
        if sigma == 0:
            ax.set_ylabel("PSNR (dB), 6-scene mean, T=8", color=INK, fontsize=9)
        ax.set_title(f"blur $\\sigma$ = {sigma} px", color=INK, fontsize=10)

        for shift in SHIFTS:
            data = load(sigma, shift)
            if not data:
                continue
            for m in METHODS:
                d = data.get(m, {})
                if not d:
                    continue
                cs = sorted(d)
                ps = [d[c] for c in cs]
                ax.plot(cs, ps, color=COLOR[m], lw=1.8, ls=DASH[shift],
                        marker=MARKER[m], ms=5, markeredgecolor=BG,
                        markeredgewidth=1.2, zorder=3)
                # mark the optimum
                bi = max(range(len(ps)), key=lambda i: ps[i])
                ax.plot([cs[bi]], [ps[bi]], marker="o", ms=10, mfc="none",
                        mec=COLOR[m], mew=1.6, zorder=4)
        # legend: only on the first panel
        if sigma == 0:
            from matplotlib.lines import Line2D
            handles = []
            for m in METHODS:
                handles.append(Line2D([0], [0], color=COLOR[m], lw=1.8,
                                       marker=MARKER[m], ms=5, label=LABEL[m]))
            for shift in SHIFTS:
                handles.append(Line2D([0], [0], color=INK2, lw=1.8,
                                       ls=DASH[shift],
                                       label=f"shift {shift} px"))
            handles.append(Line2D([0], [0], color=INK2, lw=0, marker="o", ms=9,
                                   mfc="none", mec=INK2, label="optimal c"))
            ax.legend(handles=handles, frameon=False, fontsize=7.5,
                      loc="lower left", ncol=2)

    fig.suptitle("Task 6: coding cell size vs mask-imaging defects "
                 "(T=8, 6-scene mean)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_nyquist_cell.png")
    fig.savefig(path, facecolor=BG)
    print("saved", path)


if __name__ == "__main__":
    main()

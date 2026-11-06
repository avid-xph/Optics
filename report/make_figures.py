"""Report figures from results/ablation/*.csv.

  python report/make_figures.py      -> report/figures/fig_T.png

Averaging matches the reports: mean over windows per scene, then mean over scenes.
Only T with all 6 scenes are plotted (T=36/40 have 3 scenes and are left out).
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

# reference palette, categorical slots 1-4 in fixed order; markers as secondary encoding
SERIES = [  # (label, method key, csv files, color, marker)
    ("T-free (one model)", "tfree",
     ["tfree_bench_20261106.csv", "tfree_unseen_T_20261106.csv"], "#2a78d6", "o"),
    ("ADMM-TV", "admm_tv", ["exp_T_v2.csv"], "#eb6834", "s"),
    ("TwIST", "twist", ["exp_T_twist_v3.csv"], "#1baf7a", "^"),
    ("E2E (one model per T)", "e2e", ["exp_T_e2e.csv"], "#eda100", "D"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def scene_mean(files, method):
    per = defaultdict(lambda: defaultdict(list))      # T -> scene -> [psnr]
    for fn in files:
        with open(os.path.join(RES, fn), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["method"] == method and not r["err"] and r["cell"] == "1":
                    per[int(r["T"])][r["scene"]].append(float(r["psnr"]))
    out = {}
    for T, sc in per.items():
        if len(sc) == 6:
            out[T] = sum(sum(v) / len(v) for v in sc.values()) / 6
    return dict(sorted(out.items()))


def fig_T():
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for label, key, files, color, marker in SERIES:
        d = scene_mean(files, key)
        Ts, ps = list(d), list(d.values())
        ax.plot(Ts, ps, color=color, lw=2, marker=marker, ms=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label, zorder=3)
        dy = -10 if key == "e2e" else 0                 # E2E ends next to ADMM-TV at T=16
        ax.annotate(label.split(" (")[0], (Ts[-1], ps[-1]), xytext=(6, dy),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    dd = scene_mean(["anchor_T8_20261106.csv"], "ddmadun")[8]
    ax.plot([8], [dd], ls="none", marker="*", ms=11, color="#e87ba4",
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=4,
            label="DD-MADUN (T=8 only)")
    ax.annotate("DD-MADUN, T=8 only", (8, dd), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK2)
    for T in (12, 20, 28):                              # T unseen in T-free training
        ax.axvline(T, color=GRID, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.text(12, ax.get_ylim()[1], "dashed: T not seen in T-free training",
            fontsize=7, color=INK2, va="bottom")
    ax.set_xticks([8, 12, 16, 20, 24, 28, 32])
    ax.set_xlim(6, 37)
    ax.set_xlabel("Frames per snapshot T", color=INK)
    ax.set_ylabel("PSNR (dB), 6-scene mean", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_T.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    print("saved", path)


if __name__ == "__main__":
    fig_T()

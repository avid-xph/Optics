"""Scale-up progress: T-free vs per-T specialists as training grows (v2 -> stage A -> stage B ...).

  python report/plot_scale.py --out report/figures/scale_progress

Each stage is one eval dir of sci_ablation CSVs (<model>_<nonEMA|EMA>.csv); a stage whose
directory is missing is skipped. Means are flat over all windows of the 6 scenes at each T.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGES = [("v2", "perT_sweep_v2/eval", 6), ("A", "perT_sweep_scaleA/eval", 12),
          ("B", "perT_sweep_scaleB/eval", 24)]          # (name, dir, cumulative k steps)
SPECS = ["T8", "T16", "T24", "T32"]
COLOR = {"T8": "#2a78d6", "T16": "#eb6834", "T24": "#1baf7a", "T32": "#eda100"}
INK, INK2, GRID, LEGACY = "#0b0b0b", "#52514e", "#e4e3df", "#6250d6"


def own_T(path, T):
    d = pd.read_csv(path)
    return d[d["T"] == T].psnr.mean()


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures/scale_progress")
    ap.add_argument("--variant", default="nonEMA", choices=["nonEMA", "EMA"])
    a = ap.parse_args()
    w = a.variant
    stages = [(n, ROOT / d, k) for n, d, k in STAGES if (ROOT / d / f"tfree_{w}.csv").exists()]
    ks = [k for _, _, k in stages]

    rows = []  # (stage, k, spec, T, spec_psnr, tfree_psnr)
    for n, d, k in stages:
        for m in SPECS:
            T = int(m[1:])
            if (d / f"{m}_{w}.csv").exists():
                rows.append((n, k, m, T, own_T(d / f"{m}_{w}.csv", T), own_T(d / f"tfree_{w}.csv", T)))
    df = pd.DataFrame(rows, columns=["stage", "k", "spec", "T", "spec_psnr", "tfree_psnr"])
    df["gap"] = df.spec_psnr - df.tfree_psnr

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8))
    for m in SPECS:
        s = df[df.spec == m]
        if s.empty:
            continue
        ax.plot(s.k, s.spec_psnr, color=COLOR[m], lw=2, ls="--", marker="D", ms=7, mec="white", mew=1.5)
        ax.plot(s.k, s.tfree_psnr, color=COLOR[m], lw=2.4, marker="o", ms=7)
        ax.annotate(f"T={m[1:]}", (s.k.iloc[-1], s.tfree_psnr.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
        bx.plot(s.k, s.gap, color=COLOR[m], lw=2.4, marker="o", ms=7, label=f"T={m[1:]}")
        dy = {"T24": 6, "T32": -6}.get(m, 0)   # the two near-zero gaps sit on top of each other
        bx.annotate(f"{s.gap.iloc[-1]:+.2f}", (s.k.iloc[-1], s.gap.iloc[-1]), xytext=(8, dy),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    leg = pd.read_csv(ROOT / "results/ablation/ddmadun_T8_20261107.csv").psnr.mean()
    ax.axhline(leg, color=LEGACY, lw=1.2, ls=":")
    ax.annotate(f"legacy T=8 model, ~72k steps: {leg:.2f}", (ks[0], leg), xytext=(0, 4),
                textcoords="offset points", fontsize=8.5, color=LEGACY)
    ax.plot([], [], color=INK2, lw=2.4, marker="o", label="T-free (one model)")
    ax.plot([], [], color=INK2, lw=2, ls="--", marker="D", label="specialist at its own T")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.55))
    for x, t in ((ax, "(a) PSNR at each specialist's own T"), (bx, "(b) Specialist − T-free (dB)")):
        x.set_xticks(ks, [f"{k}k\n({n})" for n, _, k in stages])
        x.set_xlim(ks[0] - 1.5, ks[-1] + 3)
        x.set_xlabel("cumulative training steps", color=INK2, fontsize=10)
        x.set_title(t, loc="left", fontsize=11, color=INK, fontweight="bold")
        style(x)
    ax.set_ylabel("PSNR (dB)", color=INK2, fontsize=10)
    bx.axhline(0, color=INK, lw=1)
    bx.set_ylabel("> 0: specialist better", color=INK2, fontsize=10)
    fig.suptitle(f"Does more data + training close the gap? ({w} weights, 6 real scenes)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    t = df.round(2).astype(str)
    md = "\n".join(["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
                   + ["| " + " | ".join(r) + " |" for r in t.values])
    Path(f"{out}_table.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wrote {out}.png / .pdf / _table.md")


if __name__ == "__main__":
    main()

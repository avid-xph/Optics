"""Figure: one T-free model vs per-T specialists on the 6 real benchmark scenes.

Reads sci_ablation.run_ablation CSVs (one per weight) from --eval_dir.
File stem = label, e.g. tfree_nonEMA, T8_nonEMA, published_tfree_EMA
(v1 names eval_<label>.csv / existing_tfree_* are accepted too).

  python report/plot_perT.py --eval_dir perT_sweep_v2/eval --out report/figures/perT_v2
  python report/plot_perT.py --eval_dir perT_sweep --out report/figures/perT_v1   # v1 data

Writes <out>.png, <out>.pdf, <out>_table.md. Means are flat over all disjoint
windows of all scenes (same aggregation as 20261107_perT_sweep_v1_results.md).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPECS = ["T8", "T16", "T24", "T32"]
COLOR = {"T8": "#2a78d6", "T16": "#eb6834", "T24": "#1baf7a", "T32": "#eda100",
         "tfree": "#0b0b0b", "published_tfree": "#8a8985"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(eval_dir, variant):
    """label -> {'psnr': {T: mean}, 'ssim': {...}, 'scene_psnr': {T: {scene: mean}}}"""
    out = {}
    for p in sorted(Path(eval_dir).glob("*.csv")):
        label = p.stem.removeprefix("eval_").replace("existing_tfree", "published_tfree")
        if not label.endswith("_" + variant):
            continue
        name = label[: -len(variant) - 1]
        acc = defaultdict(list)
        scene = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("err", "").strip():
                continue
            T = int(r["T"])
            acc[("psnr", T)].append(float(r["psnr"]))
            acc[("ssim", T)].append(float(r["ssim"]))
            scene[T][r["scene"]].append(float(r["psnr"]))
        mean = lambda v: sum(v) / len(v)
        out[name] = {
            "psnr": {T: mean(v) for (k, T), v in acc.items() if k == "psnr"},
            "ssim": {T: mean(v) for (k, T), v in acc.items() if k == "ssim"},
            "scene_psnr": {T: {s: mean(v) for s, v in d.items()} for T, d in scene.items()},
        }
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def curves(ax, data, key, Ts, ylabel):
    for name in ["published_tfree", *SPECS, "tfree"]:
        if name not in data:
            continue
        d = data[name][key]
        xs = [T for T in Ts if T in d]
        if name == "tfree":
            kw = dict(lw=2.6, marker="o", ms=7, zorder=5, label="T-free (one model)")
        elif name == "published_tfree":
            kw = dict(lw=1.5, ls="--", marker="o", ms=4, label="published T-free")
        else:
            kw = dict(lw=2, marker="o", ms=5, alpha=0.9, label=f"{name} specialist")
        ax.plot(xs, [d[T] for T in xs], color=COLOR[name], **kw)
        if name in SPECS:  # the specialist's own T: the fair head-to-head point
            T0 = int(name[1:])
            if T0 in d:
                ax.plot([T0], [d[T0]], marker="D", ms=10, color=COLOR[name],
                        mec="white", mew=2, zorder=6)
    ax.set_xticks(Ts)
    ax.set_xlabel("compression ratio T (frames per snapshot)", color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    style(ax)


def gap_bars(ax, data, key, unit, fmt):
    xs, gaps = [], []
    for name in SPECS:
        T0 = int(name[1:])
        if name in data and "tfree" in data and T0 in data[name][key]:
            xs.append(name)
            gaps.append(data[name][key][T0] - data["tfree"][key][T0])
    bars = ax.bar(xs, gaps, color=[COLOR[n] for n in xs], width=0.55)
    ax.axhline(0, color=INK, lw=1)
    lim = max(abs(g) for g in gaps) * 1.35
    ax.set_ylim(-lim, lim)
    for b, g in zip(bars, gaps):
        ax.annotate(format(g, fmt), (b.get_x() + b.get_width() / 2, g),
                    xytext=(0, 4 if g >= 0 else -4), textcoords="offset points",
                    ha="center", va="bottom" if g >= 0 else "top", fontsize=9, color=INK)
    ax.set_ylabel(f"specialist − T-free at own T ({unit})", color=INK2, fontsize=10)
    ax.set_xlabel("> 0: specialist better    < 0: T-free better", color=INK2, fontsize=9)
    style(ax)


def table(data, Ts):
    rows = ["| weight | " + " | ".join(f"T={T}" for T in Ts) + " |",
            "|---|" + "---|" * len(Ts)]
    for name in ["tfree", *SPECS, "published_tfree"]:
        if name in data:
            d = data[name]
            rows.append(f"| {name} | " + " | ".join(
                f"{d['psnr'][T]:.2f} / {d['ssim'][T]:.4f}" if T in d["psnr"] else "–"
                for T in Ts) + " |")
    rows += ["", "Specialist at own T vs T-free (PSNR):", "",
             "| T | specialist | T-free | Δ dB | scenes won by specialist |", "|---|---|---|---|---|"]
    for name in SPECS:
        T0 = int(name[1:])
        if name in data and "tfree" in data and T0 in data[name]["psnr"]:
            sp, tf = data[name]["scene_psnr"][T0], data["tfree"]["scene_psnr"][T0]
            won = sum(sp[s] > tf[s] for s in sp)
            a, b = data[name]["psnr"][T0], data["tfree"]["psnr"][T0]
            rows.append(f"| {T0} | {a:.2f} | {b:.2f} | {a - b:+.2f} | {won}/{len(sp)} |")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--variant", default="nonEMA", choices=["nonEMA", "EMA"])
    a = ap.parse_args()

    data = load(a.eval_dir, a.variant)
    Ts = sorted({T for d in data.values() for T in d["psnr"]})

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2),
                            gridspec_kw=dict(width_ratios=[1.6, 1]))
    curves(axs[0, 0], data, "psnr", Ts, "PSNR (dB)")
    curves(axs[1, 0], data, "ssim", Ts, "SSIM")
    gap_bars(axs[0, 1], data, "psnr", "dB", "+.2f")
    gap_bars(axs[1, 1], data, "ssim", "SSIM", "+.4f")
    for ax, t in zip(axs.flat, ["(a) PSNR across T", "(b) Specialist advantage, PSNR",
                                "(c) SSIM across T", "(d) Specialist advantage, SSIM"]):
        ax.set_title(t, loc="left", fontsize=11, color=INK, fontweight="bold")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"One T-free model vs per-T specialists — 6 real scenes, equal compute "
                 f"({a.variant} weights; ◆ = specialist at its own T)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    md = table(data, Ts)
    Path(f"{out}_table.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out}.png / .pdf / _table.md")


if __name__ == "__main__":
    main()

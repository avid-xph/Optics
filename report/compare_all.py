"""All known results on the 6 real benchmark scenes, in one figure + one table.

  python report/compare_all.py --out report/figures/compare_all

Sources (all sci_ablation.run_ablation CSVs, noise 0.01, same windows per T):
  results/ablation/admmtv_realT_20261107.csv   ADMM-TV (iterative baseline)
  results/ablation/ddmadun_T8_20261107.csv     legacy DD-MADUN, T=8 only (145 epochs)
  perT_sweep/eval_*.csv                        v1: 1.5k steps, unequal data
  perT_sweep_v2/eval/*.csv                     v2: 6k steps, 200 clips per T (incl. published T-free)
Means are flat over all windows of all scenes at each T.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPECS = ["T8", "T16", "T24", "T32"]
COLOR = {"T8": "#2a78d6", "T16": "#eb6834", "T24": "#1baf7a", "T32": "#eda100",
         "tfree": "#0b0b0b", "admm": "#8a8985", "legacy": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def mean_by_T(path, key="psnr"):
    d = pd.read_csv(path)
    d = d[d["err"].isna()] if "err" in d else d
    return d.groupby("T")[key].mean().to_dict()


def collect(key):
    """row label -> {T: mean}; v1/v2 rows for both weight variants."""
    rows = {"ADMM-TV": mean_by_T(ROOT / "results/ablation/admmtv_realT_20261107.csv", key),
            "legacy DD-MADUN (T=8 only, 145 ep)": mean_by_T(ROOT / "results/ablation/ddmadun_T8_20261107.csv", key)}
    for w in ("nonEMA", "EMA"):
        for ver, d, pre in (("v1", ROOT / "perT_sweep", "eval_"), ("v2", ROOT / "perT_sweep_v2/eval", "")):
            for m in ["tfree", *SPECS]:
                p = d / f"{pre}{m}_{w}.csv"
                if p.exists():
                    rows[f"{ver} {m} [{w}]"] = mean_by_T(p, key)
        p = ROOT / f"perT_sweep_v2/eval/published_tfree_{w}.csv"
        rows[f"published T-free [{w}]"] = mean_by_T(p, key)
    return rows


def gaps(rows, ver, w):
    out = {}
    for m in SPECS:
        sp, tf = rows.get(f"{ver} {m} [{w}]"), rows.get(f"{ver} tfree [{w}]")
        T0 = int(m[1:])
        if sp and tf and T0 in sp and T0 in tf:
            out[m] = sp[T0] - tf[T0]
    return out


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
    ap.add_argument("--out", default="report/figures/compare_all")
    a = ap.parse_args()
    P, S = collect("psnr"), collect("ssim")
    Ts = sorted(P["ADMM-TV"])
    w = "nonEMA"

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))

    # (a) absolute PSNR: baselines, T-free v1 -> v2, specialists at their own T
    ad = P["ADMM-TV"]
    ax.plot(Ts, [ad[T] for T in Ts], color=COLOR["admm"], lw=2, marker="o", ms=5, label="ADMM-TV")
    v1 = P[f"v1 tfree [{w}]"]
    x1 = [T for T in Ts if T in v1]
    ax.plot(x1, [v1[T] for T in x1], color=COLOR["tfree"], lw=1.5, ls="--", marker="o", ms=5,
            mfc="white", label="T-free v1 (1.5k steps)")
    v2 = P[f"v2 tfree [{w}]"]
    ax.plot(Ts, [v2[T] for T in Ts], color=COLOR["tfree"], lw=2.6, marker="o", ms=7, zorder=5,
            label="T-free v2 (6k steps)")
    for m in SPECS:
        d, T0 = P.get(f"v2 {m} [{w}]"), int(m[1:])
        if d and T0 in d:
            ax.plot([T0], [d[T0]], marker="D", ms=10, color=COLOR[m], mec="white", mew=2,
                    zorder=6, ls="none", label=f"{m} specialist v2 (own T)")
    lg = P["legacy DD-MADUN (T=8 only, 145 ep)"][8]
    ax.plot([8], [lg], marker="*", ms=16, color=COLOR["legacy"], mec="white", mew=1.5, zorder=7,
            ls="none", label="legacy DD-MADUN, T=8 only (145 epochs)")
    ax.set_xticks(Ts)
    ax.set_xlabel("compression ratio T (frames per snapshot)", color=INK2, fontsize=10)
    ax.set_ylabel("PSNR (dB)", color=INK2, fontsize=10)
    ax.set_title("(a) PSNR on 6 real scenes", loc="left", fontsize=11, color=INK, fontweight="bold")
    style(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # (b) cost of generality at each specialist's own T: v1 vs v2
    g1, g2 = gaps(P, "v1", w), gaps(P, "v2", w)
    xs = range(len(SPECS))
    bw = 0.36
    for i, m in enumerate(SPECS):
        for off, g, alpha, lab in ((-bw / 2 - 0.01, g1, 0.35, "v1"), (bw / 2 + 0.01, g2, 1.0, "v2")):
            if m in g:
                bx.bar(i + off, g[m], bw, color=COLOR[m], alpha=alpha)
                bx.annotate(f"{g[m]:+.2f}", (i + off, g[m]), xytext=(0, 3 if g[m] >= 0 else -3),
                            textcoords="offset points", ha="center",
                            va="bottom" if g[m] >= 0 else "top", fontsize=8.5, color=INK)
    bx.axhline(0, color=INK, lw=1)
    bx.set_xticks(list(xs), SPECS)
    lim = max(abs(v) for g in (g1, g2) for v in g.values()) * 1.3
    bx.set_ylim(-lim / 3, lim)
    bx.set_ylabel("specialist − T-free at own T (dB)", color=INK2, fontsize=10)
    bx.set_xlabel("pale = v1 (1.5k steps, unequal data)   solid = v2 (6k, equal)\n"
                  "> 0: specialist better   < 0: T-free better", color=INK2, fontsize=9)
    bx.set_title("(b) Cost of generality, v1 → v2", loc="left", fontsize=11, color=INK, fontweight="bold")
    style(bx)

    fig.suptitle(f"All results so far ({w} weights; means over all windows of 6 scenes)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")

    # table: every row, PSNR / SSIM per T, then the gaps for both weight variants
    md = ["| model | " + " | ".join(f"T={T}" for T in Ts) + " |", "|---|" + "---|" * len(Ts)]
    for k in P:
        md.append(f"| {k} | " + " | ".join(
            f"{P[k][T]:.2f} / {S[k][T]:.4f}" if T in P[k] else "–" for T in Ts) + " |")
    md += ["", "Specialist − T-free at the specialist's own T (PSNR dB / SSIM):", "",
           "| | " + " | ".join(SPECS) + " |", "|---|" + "---|" * len(SPECS)]
    for ver in ("v1", "v2"):
        for ww in ("nonEMA", "EMA"):
            gp, gs = gaps(P, ver, ww), gaps(S, ver, ww)
            md.append(f"| {ver} {ww} | " + " | ".join(
                f"{gp[m]:+.2f} / {gs[m]:+.4f}" if m in gp else "–" for m in SPECS) + " |")
    Path(f"{out}_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out}.png / .pdf / _table.md")


if __name__ == "__main__":
    main()

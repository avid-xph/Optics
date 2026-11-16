"""Figures for the T-free mentor report -> pic/pNN_<content>.png

  python report/make_mentor_figs.py            # all data figures (seconds)
  python report/make_mentor_figs.py --recon    # + reconstruction visual (CPU inference, minutes)

Data sources (all measured, 6 real benchmark scenes, cell=1, noise=0.01, disjoint windows):
  results/ablation/tfree_realT_20261107.csv   published T-free, 7 T incl. unseen 12/20/28
  results/ablation/admmtv_realT_20261107.csv  ADMM-TV, same grid
  results/ablation/ddmadun_T8_20261107.csv    legacy T-bound DD-MADUN, T=8 only
  perT_sweep/eval_*_nonEMA.csv                v1 per-T sweep (1500 steps each)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIC = ROOT / "pic"
ABL = ROOT / "results/ablation"
SWEEP = ROOT / "perT_sweep"

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TFREE, ADMM, LEGACY = INK, "#8a8985", "#6250d6"
SPEC = {"T8": "#2a78d6", "T16": "#eb6834", "T24": "#1baf7a", "T32": "#eda100"}
UNSEEN = {12, 20, 28}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})


def style(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    PIC.mkdir(exist_ok=True)
    fig.savefig(PIC / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", PIC / f"{name}.png")


def rows(path):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if not r["err"].strip()]


def mean_by(path, key="psnr", by=("T",)):
    acc = defaultdict(list)
    for r in rows(path):
        acc[tuple(r[b] if b == "scene" else int(r[b]) for b in by)].append(float(r[key]))
    return {k if len(k) > 1 else k[0]: float(np.mean(v)) for k, v in acc.items()}


# --------------------------------------------------------------------------- p02
def p02_measurement_shapes():
    sys.path.insert(0, str(ROOT))
    import torch
    from sci_ablation import data
    from sci_ablation.forward import ExperimentCUP
    scene = data.load_scene("kobe")
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 2.6], hspace=0.35, wspace=0.08)
    for row, T in enumerate([8, 32]):
        gt = scene[:T].unsqueeze(0)
        torch.manual_seed(0)
        E = ExperimentCUP(T=T, seed=0).measure(gt)[0].numpy()
        for j, f in enumerate([0, T // 3, 2 * T // 3, T - 1]):
            ax = fig.add_subplot(gs[row, j])
            ax.imshow(gt[0, f], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"frame {f + 1}/{T}", fontsize=9, fontweight="normal", color=INK2)
            if j == 0:
                ax.set_ylabel(f"T = {T}", fontsize=13, color=INK, fontweight="bold")
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(E, cmap="gray", aspect="equal")
        ax.set_xlim(0, 256 + 31)
        ax.set_xticks([0, E.shape[1]]); ax.set_yticks([])
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_title(f"one snapshot: 256 × {E.shape[1]}  (width = W + T − 1)",
                     fontsize=10, fontweight="normal", color=INK)
    fig.suptitle("Same scene, different T → different number of frames AND a different "
                 "measurement width.\nA network trained for one T cannot even load another T.",
                 x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "p02_problem_measurement_depends_on_T")


# --------------------------------------------------------------------------- p04
def p04_tfree_vs_admm():
    tf, ad = mean_by(ABL / "tfree_realT_20261107.csv"), mean_by(ABL / "admmtv_realT_20261107.csv")
    tfs = mean_by(ABL / "tfree_realT_20261107.csv", "ssim")
    ads = mean_by(ABL / "admmtv_realT_20261107.csv", "ssim")
    Ts = sorted(tf)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, a, b, lab in [(axs[0], tf, ad, "PSNR (dB)"), (axs[1], tfs, ads, "SSIM")]:
        ax.plot(Ts, [b[T] for T in Ts], color=ADMM, lw=2, ls="--", marker="s", ms=6,
                label="ADMM-TV (iterative, any T)")
        ax.plot(Ts, [a[T] for T in Ts], color=TFREE, lw=2.6, zorder=4,
                label="T-free (ONE network)")
        for T in Ts:
            unseen = T in UNSEEN
            ax.plot([T], [a[T]], marker="o", ms=9, zorder=5, color=TFREE,
                    mfc="white" if unseen else TFREE, mew=2)
        ax.set_xticks(Ts)
        ax.set_xlabel("T (frames per snapshot)    ○ = T never seen in training", color=INK2)
        ax.set_ylabel(lab, color=INK2)
        style(ax)
    for T in Ts:
        axs[0].annotate(f"+{tf[T] - ad[T]:.1f}", (T, tf[T]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
    axs[0].set_title("(a) PSNR — one T-free model across 7 T")
    axs[1].set_title("(b) SSIM")
    axs[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Claim A: one weight set reconstructs every T — including unseen 12/20/28 — "
                 "and beats ADMM-TV by 2.2–3.0 dB", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "p04_Tfree_one_model_7T_vs_ADMMTV")


# --------------------------------------------------------------------------- p05
def p05_per_scene_gain():
    tf = mean_by(ABL / "tfree_realT_20261107.csv", by=("scene", "T"))
    ad = mean_by(ABL / "admmtv_realT_20261107.csv", by=("scene", "T"))
    scenes = sorted({s for s, _ in tf})
    Ts = sorted({T for _, T in tf})
    M = np.array([[tf[(s, T)] - ad[(s, T)] for T in Ts] for s in scenes])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15, aspect="auto")
    for i in range(len(scenes)):
        for j in range(len(Ts)):
            ax.text(j, i, f"+{M[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > M.max() * 0.6 else INK)
    ax.set_xticks(range(len(Ts)), [f"{T}{'*' if T in UNSEEN else ''}" for T in Ts])
    ax.set_yticks(range(len(scenes)), scenes)
    ax.set_xlabel("T   (* = never seen in training)", color=INK2)
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("T-free − ADMM-TV (dB)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"T-free beats ADMM-TV in all {M.size}/{M.size} scene × T cells "
                 f"(min +{M.min():.1f} dB)")
    save(fig, "p05_per_scene_gain_over_ADMMTV_heatmap")


# --------------------------------------------------------------------------- p06 / p07
def sweep():
    out = {}
    for n in ["tfree", *SPEC]:
        out[n] = mean_by(SWEEP / f"eval_{n}_nonEMA.csv")
    return out


def p06_specialists_collapse():
    d, Ts = sweep(), [8, 16, 24, 32]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for n, c in SPEC.items():
        ax.plot(Ts, [d[n][T] for T in Ts], color=c, lw=2, marker="o", ms=5,
                label=f"{n} specialist (trained on T={n[1:]} only)")
        T0 = int(n[1:])
        ax.plot([T0], [d[n][T0]], marker="D", ms=11, color=c, mec="white", mew=2, zorder=6)
    ax.plot(Ts, [d["tfree"][T] for T in Ts], color=TFREE, lw=3, marker="o", ms=7,
            zorder=5, label="T-free (one model, same compute)")
    ax.annotate(f"T8 specialist: −{d['T8'][8] - d['T8'][32]:.1f} dB from T=8→32",
                (32, d["T8"][32]), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", fontsize=10, color=INK)
    ax.annotate(f"T-free: −{d['tfree'][8] - d['tfree'][32]:.1f} dB",
                (32, d["tfree"][32]), xytext=(0, 14), textcoords="offset points",
                ha="right", fontsize=10, color=INK)
    ax.set_xticks(Ts)
    ax.set_xlabel("T at test time     ◆ = specialist at its own T", color=INK2)
    ax.set_ylabel("PSNR (dB)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style(ax)
    ax.set_title("Specialists collapse away from their own T; T-free degrades only with "
                 "intrinsic difficulty")
    save(fig, "p06_specialists_collapse_off_their_T")


def p07_cost_of_generality():
    d = sweep()
    Ts = [8, 16, 24, 32]
    x = np.arange(len(Ts))
    fig, ax = plt.subplots(figsize=(10, 5))
    sp = [d[f"T{T}"][T] for T in Ts]
    tf = [d["tfree"][T] for T in Ts]
    ax.bar(x - 0.2, sp, 0.38, color=[SPEC[f"T{T}"] for T in Ts], label="specialist at its own T")
    ax.bar(x + 0.2, tf, 0.38, color=TFREE, label="T-free")
    for i, (a, b) in enumerate(zip(sp, tf)):
        top = max(a, b)
        ax.annotate(f"{a - b:+.2f} dB", (x[i], top), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x, [f"T = {T}" for T in Ts])
    ax.set_ylim(min(sp + tf) - 3, max(sp + tf) + 1.2)
    ax.set_ylabel("PSNR (dB)", color=INK2)
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    ax.set_title("Claim B: cost of one model vs 4 specialists — mean 0.58 dB, "
                 "concentrated at T=8\n(label = specialist − T-free; v1: 1500 steps each, "
                 "T=8 specialist had 4× the clips)", fontsize=12)
    save(fig, "p07_cost_of_generality_specialist_vs_Tfree")


# --------------------------------------------------------------------------- p08
def p08_T8_honest():
    d = sweep()
    vals = [
        ("ADMM-TV\n(iterative)", mean_by(ABL / "admmtv_realT_20261107.csv")[8], ADMM),
        ("T-free v1\n1500 steps", d["tfree"][8], "#b9b8b3"),
        ("T-free published\n6000 steps", mean_by(ABL / "tfree_realT_20261107.csv")[8], TFREE),
        ("T8 specialist v1\n1500 steps", d["T8"][8], SPEC["T8"]),
        ("legacy T-bound\nDD-MADUN (145 ep)", mean_by(ABL / "ddmadun_T8_20261107.csv")[8], LEGACY),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals], width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v[1]:.2f}", (b.get_x() + b.get_width() / 2, v[1]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=11, color=INK)
    ax.set_ylim(24, 31)
    ax.set_ylabel("PSNR at T = 8 (dB)", color=INK2)
    ax.tick_params(axis="x", labelsize=9)
    style(ax)
    ax.set_title("T = 8 is where T-free still pays: −2.1 dB vs the long-trained T-bound model\n"
                 "(not compute-matched — v2 sweep isolates the true cost)", fontsize=12)
    save(fig, "p08_T8_gap_honest_comparison")


# --------------------------------------------------------------------------- p09
def p09_recon(scene_name):
    sys.path.insert(0, str(ROOT))
    import torch
    from baselines import compute_psnr
    from sci_ablation import data
    from sci_ablation.forward import ExperimentCUP
    from sci_ablation.methods import build_method
    scene = data.load_scene(scene_name)
    tfree = build_method("tfree", {"ckpt": str(ROOT / "checkpoints/tfree_trained/best_model.pth")})
    admm = build_method("admm_tv")
    Ts = [8, 20, 32]
    fig, axs = plt.subplots(len(Ts), 3, figsize=(10, 3.4 * len(Ts)))
    for i, T in enumerate(Ts):
        gt = scene[:T].unsqueeze(0)
        cup = ExperimentCUP(T=T, seed=0)
        torch.manual_seed(0)
        E = cup.measure(gt)
        f = T // 2
        for j, (lab, rec) in enumerate([("ground truth", gt),
                                        ("ADMM-TV", admm.reconstruct(E, cup, T)[..., :256]),
                                        ("T-free", tfree.reconstruct(E, cup, T)[..., :256])]):
            rec = torch.clamp(rec, 0, 1)
            ax = axs[i, j]
            ax.imshow(rec[0, f].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            sub = "" if j == 0 else f"  {float(compute_psnr(rec, gt)):.2f} dB (all {T} frames)"
            ax.set_title(f"{lab}{sub}", fontsize=10, fontweight="normal", color=INK)
            if j == 0:
                tag = " (unseen)" if T in UNSEEN else ""
                ax.set_ylabel(f"T = {T}{tag}\nframe {f + 1}", fontsize=12, color=INK,
                              fontweight="bold")
        print(f"  recon T={T} done")
    fig.suptitle(f"Reconstruction, scene '{scene_name}' — same T-free weights for every row",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, f"p09_recon_visual_{scene_name}_T8_T20_T32")


def median_scene():
    tf = mean_by(ABL / "tfree_realT_20261107.csv", by=("scene",))
    s = sorted(tf, key=tf.get)
    return s[len(s) // 2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    p02_measurement_shapes()
    p04_tfree_vs_admm()
    p05_per_scene_gain()
    p06_specialists_collapse()
    p07_cost_of_generality()
    p08_T8_honest()
    if a.recon:
        p09_recon(median_scene())

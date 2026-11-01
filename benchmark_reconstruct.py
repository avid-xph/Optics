"""Reconstruct benchmark orig scenes using model's own real_cup forward model.
Split each scene into T=8 groups, generate real_cup measurements, reconstruct.

Usage:
  python benchmark_reconstruct.py                    # default: P4 model (checkpoints/best_model.pth)
  python benchmark_reconstruct.py --ckpt <path>      # custom checkpoint
"""
import sys, os, argparse
import numpy as np
import scipy.io
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cup_simulator import CUPForwardModel
from baselines import compute_psnr, compute_ssim
from dd_madun_v3 import DD_MADUNv3

# === Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None, help="Custom checkpoint path")
args = parser.parse_args()

T_MODEL, H, W = 8, 256, 256
SHEAR = 1
NOISE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Determine checkpoint and output directory
_BASE = os.path.dirname(os.path.abspath(__file__))
if args.ckpt:
    CKPT = args.ckpt
    OUT_DIR = os.path.join(os.path.dirname(CKPT), "benchmark_viz")
else:
    CKPT = os.path.join(_BASE, "checkpoints", "best_model.pth")
    OUT_DIR = os.path.join(_BASE, "results")

BENCH_DIR = os.path.join(_BASE, "benchmark")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = ["aerial", "crash", "drop", "kobe", "runner", "traffic"]

print(f"Device: {DEVICE}  |  real_cup shear={SHEAR}")
print(f"Model : {CKPT}")

# Load model (support both raw state_dict and checkpoint dict)
model = DD_MADUNv3(T=T_MODEL, H=H, W=W, K=8, K1=3, ng=6, nf=96, nb=8).to(DEVICE)
ckpt_data = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
    model.load_state_dict(ckpt_data["model_state_dict"])
    print(f"  Loaded from checkpoint dict (epoch {ckpt_data.get('epoch', '?')})")
else:
    model.load_state_dict(ckpt_data)
model.eval()
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# real_cup forward model (matching training)
cup = CUPForwardModel(H=H, W=W, T=T_MODEL, shear_per_frame=SHEAR, real_cup=True)
cup.to(DEVICE)
Wm = W + (T_MODEL - 1) * SHEAR
print(f"  Measurement: {H}x{Wm}")

all_ps, all_ss = [], []

for fname in FILES:
    mat = scipy.io.loadmat(os.path.join(BENCH_DIR, f"{fname}.mat"))
    orig = mat["orig"].astype(np.float32)            # (H,W,T_total)
    orig_n = orig / 255.0                             # → [0,1]
    T_total = orig_n.shape[-1]
    N_groups = T_total // T_MODEL

    assert T_total % T_MODEL == 0, f"T_total={T_total} not divisible by T_MODEL"

    print(f"\n{fname.upper()} | T={T_total} ({N_groups} groups)")

    # Reconstruct group by group
    rec_all = np.zeros((T_total, H, W), dtype=np.float32)
    group_ps, group_ss = [], []

    for g in range(N_groups):
        # Extract T=8 frames from orig
        frames = torch.from_numpy(
            orig_n[:, :, g * T_MODEL:(g + 1) * T_MODEL]
        ).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,8,256,256)

        # Forward: generate measurement
        E = cup.forward(frames) + NOISE * torch.randn_like(cup.forward(frames))

        # Reconstruct
        with torch.no_grad():
            I, _ = model(E, cup.forward, cup.adjoint)
            I = torch.clamp(I, 0, 1)

        rec_all[g * T_MODEL:(g + 1) * T_MODEL] = I[0].detach().cpu().numpy()
        gp = float(compute_psnr(I, frames))
        gs = float(compute_ssim(I, frames))
        group_ps.append(gp); group_ss.append(gs)

    # Overall PSNR/SSIM
    rec_t = torch.from_numpy(rec_all).unsqueeze(0)
    orig_t = torch.from_numpy(orig_n).permute(2, 0, 1).unsqueeze(0)
    p = float(compute_psnr(rec_t, orig_t))
    s = float(compute_ssim(rec_t, orig_t))
    all_ps.append(p); all_ss.append(s)
    print(f"  Groups: PSNR={[f'{x:.1f}' for x in group_ps]}")
    print(f"  Overall: PSNR={p:.2f} dB  SSIM={s:.4f}")

    # ---- Visualization ----
    step = max(1, T_total // 8)
    show_idx = list(range(0, T_total, step))
    n_show = len(show_idx)

    fig = plt.figure(figsize=(n_show * 3 + 2, 9))
    plt.suptitle(
        f"{fname.upper()}  |  T={T_total}  |  real_cup  |  "
        f"PSNR={p:.2f}dB  SSIM={s:.4f}",
        fontsize=13, y=0.995)

    for idx, t in enumerate(show_idx):
        ax = fig.add_axes([0.03 + idx * (0.94 / n_show), 0.50,
                           0.90 / n_show, 0.44])
        ax.imshow(orig_n[:, :, t], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"GT t={t}", fontsize=9); ax.axis("off")
    for idx, t in enumerate(show_idx):
        ax = fig.add_axes([0.03 + idx * (0.94 / n_show), 0.04,
                           0.90 / n_show, 0.44])
        ax.imshow(rec_all[t], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Rec t={t}", fontsize=9); ax.axis("off")

    fig.text(0.005, 0.72, "GT", fontsize=12, color="blue", va="center")
    model_label = "DD-MADUN P4"
    fig.text(0.005, 0.26, model_label, fontsize=12, color="green", va="center")
    out_png = os.path.join(OUT_DIR, f"bench_{fname}_realcup.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_png}")

# Summary
print(f"\n{'='*55}")
print(f"  Benchmark: orig scenes via real_cup forward")
print(f"  {'Scene':<10s} | {'PSNR':>8s} | {'SSIM':>7s}")
print(f"  {'-'*10} | {'-'*8} | {'-'*7}")
for fn, p, s in zip(FILES, all_ps, all_ss):
    print(f"  {fn:<10s} | {p:8.2f} | {s:7.4f}")
print(f"  {'-'*10} | {'-'*8} | {'-'*7}")
print(f"  {'MEAN':<10s} | {np.mean(all_ps):8.2f} | {np.mean(all_ss):7.4f}")
print(f"{'='*55}")
print(f"Output: {OUT_DIR}")

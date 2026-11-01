"""
DD-MADUN v3 P4 Training: B1 (real video data) + B2 (FFL) + B6/B7 (VGG + training strategy)

Key changes from v3:
  - B1: Mixed real video (DAVIS 2017) + enhanced synthetic training data
  - B2: Focal Frequency Loss (spatial FFT domain, ICCV'21)
  - B6: 150 epochs, staged LR, enhanced augmentation (rotation + brightness)
  - B7: VGG16 perceptual loss

Usage:
  python dd_madun_v3_train_p4.py [--resume <path>] [--batch_size 4] [--epochs 150]
"""
import sys, os, time, json, copy, math, argparse
import numpy as np
import torch
import torch.optim as optim

# Add madun path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MADUN_DIR = os.path.join(SCRIPT_DIR, "..")
from cup_simulator import CUPForwardModel
from baselines import twist_reconstruct, admm_l1_reconstruct, compute_psnr, compute_ssim
from dataset import create_complex_dataset

# P4 imports
from video_dataset import create_mixed_dataset
from dd_madun_v3 import DD_MADUNv3, DDMADUNLossv3, build_dct_matrix

# === Argument parsing ===
parser = argparse.ArgumentParser(description="DD-MADUN v3 P4 Training")
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_train", type=int, default=2000)
parser.add_argument("--n_test", type=int, default=20)
parser.add_argument("--real_ratio", type=float, default=0.7, help="Fraction of real video data")
parser.add_argument("--ffl_weight", type=float, default=0.10)
parser.add_argument("--vgg_weight", type=float, default=0.02)
parser.add_argument("--out_dir", type=str, default="output_v3_p4")
parser.add_argument("--davis_root", type=str, default="DAVIS")
# === Grayscale experimental-style mask (Plan A) ===
parser.add_argument("--gray_mask", action="store_true",
                    help="Use experimental-style grayscale masks (mean~0.23, ~43%% near-dark)")
parser.add_argument("--gray_p", type=float, default=0.30,
                    help="Bright fraction of base binary pattern before blur")
parser.add_argument("--gray_blur", type=float, default=2.0,
                    help="Gaussian blur sigma (optical PSF) applied to base pattern")
parser.add_argument("--gray_skew", type=float, default=1.5,
                    help="Power applied after blur to skew histogram toward dark")
parser.add_argument("--gray_seed", type=int, default=12345,
                    help="Seed for the fixed eval mask (reproducibility)")
parser.add_argument("--continue_train", action="store_true",
                    help="Continue training from checkpoint epoch (default: fine-tune, start_epoch=1)")
# === Plan B: mask-weighted Data Consistency ===
parser.add_argument("--mask_dc", action="store_true",
                    help="Plan B: gate data-fidelity gradient by mask_w (dark->denoiser)")
parser.add_argument("--mask_dc_thr", type=float, default=0.05,
                    help="Near-dark threshold for mask_w (sigmoid center)")
parser.add_argument("--mask_dc_soft", type=float, default=0.05,
                    help="Soft width of mask_w sigmoid")
args = parser.parse_args()

# === P4 Training Config ===
C = {
    "T": 8, "H": 256, "W": 256,
    "K": 8, "K1": 3,
    "n_gd_steps": 6,
    "denoiser_feats": 96,
    "denoiser_blocks": 8,
    "n_train": args.n_train,
    "n_test": args.n_test,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "warmup_epochs": 10,
    "weight_decay": 1e-5,
    "noise_std": 0.01,
    "noise_scale": 1.0,
    # Loss weights
    "alpha_stage": 0.15,
    "gamma_freq": 0.15,
    "ssim_weight": 0.20,
    "edge_weight": 0.05,
    "ffl_weight": args.ffl_weight,     # B2: Focal Frequency Loss
    "vgg_weight": args.vgg_weight,     # B7: VGG perceptual loss
    # Training strategy
    "ema_decay": 0.999,
    "augment_flip": True,
    "augment_rotate": True,
    "augment_brightness": True,
    "shear_per_frame": 1,
    "real_ratio": args.real_ratio,
    "device": "cuda",
    "output_dir": args.out_dir,
    "davis_root": args.davis_root,
    "seed": 42,
    "n_objects": 7,
    "resume": args.resume,
    # Grayscale experimental-style mask (Plan A)
    "gray_mask": args.gray_mask,
    "gray_p": args.gray_p,
    "gray_blur": args.gray_blur,
    "gray_skew": args.gray_skew,
    "gray_seed": args.gray_seed,
    # Plan B: mask-weighted DC
    "mask_dc": args.mask_dc,
    "mask_dc_thr": args.mask_dc_thr,
    "mask_dc_soft": args.mask_dc_soft,
}

os.makedirs(C["output_dir"], exist_ok=True)
torch.manual_seed(C["seed"])
np.random.seed(C["seed"])

# === Grayscale experimental-style mask sampler (Plan A) ===
def _build_exp_cdf(mask_path):
    """Build empirical CDF + bin centers from the experimental mask.
    Returns (cdf, centers, stats_dict) where stats_dict holds real pixel fractions.
    """
    exp = np.load(mask_path)
    vals = exp.flatten().astype(np.float32)
    hist, edges = np.histogram(vals, bins=256, range=(0, 1))
    cdf = np.cumsum(hist) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    stats = {
        "mean": float(vals.mean()),
        "frac_near_dark": float((vals < 0.05).mean()),
        "frac_bright": float((vals > 0.5).mean()),
    }
    return cdf.astype(np.float32), centers.astype(np.float32), stats

def sample_grayscale_mask(H, W, device, exp_cdf=None, exp_centers=None,
                          blur=2.0, seed=None):
    """Generate a grayscale encoding mask by histogram-matching a smooth random
    field to the experimental mask's empirical distribution.

    This yields masks whose statistics EXACTLY match the experimental mask
    (mean~0.23, ~43% near-dark, ~17% bright) while providing infinite spatial
    diversity via different random smooth fields (Gaussian-blurred binary base).
    """
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        base = torch.bernoulli(torch.full((H, W), 0.5), generator=g).float()
    else:
        base = torch.bernoulli(torch.full((H, W), 0.5)).float()
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(base.numpy(), sigma=blur)
    fmin, fmax = field.min(), field.max()
    field = (field - fmin) / (fmax - fmin + 1e-8)
    flat = field.flatten()
    order = np.argsort(flat)
    n = len(flat)
    # Inverse-CDF sampling: map uniform quantiles -> experimental values
    target_sorted = np.interp(np.linspace(0, 1, n), exp_cdf, exp_centers)
    matched = np.empty(n, dtype=np.float32)
    matched[order] = target_sorted
    M = matched.reshape(H, W)
    return torch.from_numpy(M).to(device)


def make_mask_w(mask, thr, soft):
    """Build the (1,1,H,W) Plan-B weight map from a mask.

    W = sigmoid((M - thr) / soft): ~1 in bright regions, ~0 in near-dark
    regions. Detached so the mask itself is never trained.
    """
    if mask.dim() == 2:
        M = mask.view(1, 1, mask.shape[0], mask.shape[1])
    else:
        M = mask.view(1, 1, mask.shape[-2], mask.shape[-1])
    return torch.sigmoid((M - thr) / max(soft, 1e-4)).detach()

print("=" * 70)
print("  DD-MADUN v3: P4 Training (B1+B2+B6+B7)")
print("  Config: T=%d, H=%d, W=%d, K=%d, K1=%d" % (C["T"],C["H"],C["W"],C["K"],C["K1"]))
print("  Denoiser: feats=%d, blocks=%d, GD steps=%d" % (C["denoiser_feats"],C["denoiser_blocks"],C["n_gd_steps"]))
print("  Loss: alpha=%.2f, gamma=%.2f, SSIM=%.2f, Edge=%.2f, FFL=%.2f, VGG=%.3f" % (
    C["alpha_stage"],C["gamma_freq"],C["ssim_weight"],C["edge_weight"],
    C["ffl_weight"],C["vgg_weight"]))
print("  Data: %d samples, real_ratio=%.1f, %d epochs, device=%s" % (
    C["n_train"],C["real_ratio"],C["epochs"],C["device"]))
if C["gray_mask"]:
    print("  GrayMask: ON (histogram-matched to experimental mask, blur=%.1f seed=%d)" % (
        C["gray_blur"], C["gray_seed"]))
if C["mask_dc"]:
    print("  MaskDC (Plan B): ON (thr=%.3f soft=%.3f) — dark regions gated to denoiser" % (
        C["mask_dc_thr"], C["mask_dc_soft"]))
if C["resume"]:
    print("  Resume: %s" % C["resume"])
print("=" * 70)

# === Grayscale mask: load experimental CDF for histogram matching (Plan A) ===
EXP_MASK_PATH = os.path.join(MADUN_DIR, "实验数据", "7-28 10hz 3vpp", "encoding_mask_normalized.npy")
if C["gray_mask"]:
    assert os.path.exists(EXP_MASK_PATH), f"Experimental mask not found: {EXP_MASK_PATH}"
    exp_cdf, exp_centers, exp_stats = _build_exp_cdf(EXP_MASK_PATH)
    C["exp_cdf"] = exp_cdf
    C["exp_centers"] = exp_centers
    print(f"  [GrayMask] Loaded experimental CDF from {EXP_MASK_PATH}")
    print(f"  [GrayMask] stats: mean={exp_stats['mean']:.3f}, "
          f"{100*exp_stats['frac_near_dark']:.0f}% near-dark(<0.05), "
          f"{100*exp_stats['frac_bright']:.0f}% bright(>0.5)")

# === 1. Dataset (B1: mixed real video + synthetic) ===
print("\n[1/7] Creating mixed dataset (B1)...")
train_data, test_data = create_mixed_dataset(
    n_train=C["n_train"], n_test=C["n_test"],
    T=C["T"], H=C["H"], W=C["W"],
    real_ratio=C["real_ratio"],
    davis_root=C["davis_root"],
    seed=C["seed"],
    n_objects=C["n_objects"])
print(f"  Train: {train_data.shape}, Test: {test_data.shape}")

cup = CUPForwardModel(H=C["H"], W=C["W"], T=C["T"], shear_per_frame=C["shear_per_frame"])
cup.to(C["device"])

# === 2. Model ===
print("\n[2/7] Building DD-MADUNv3...")
model = DD_MADUNv3(T=C["T"], H=C["H"], W=C["W"],
    K=C["K"], K1=C["K1"], ng=C["n_gd_steps"],
    nf=C["denoiser_feats"], nb=C["denoiser_blocks"])
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

dct_mat = build_dct_matrix(C["T"])
criterion = DDMADUNLossv3(
    as_=C["alpha_stage"], gf=C["gamma_freq"],
    sw=C["ssim_weight"], ew=C["edge_weight"], dm=dct_mat,
    ffl_w=C["ffl_weight"], vgg_w=C["vgg_weight"],
    device=C["device"])
print(f"  FFL: {'ON' if C['ffl_weight']>0 else 'OFF'}, VGG: {'ON' if C['vgg_weight']>0 else 'OFF'}")

# === 3. Optimizer + Staged LR (B6) ===
optimizer = optim.Adam(model.parameters(), lr=C["lr"], weight_decay=C["weight_decay"])

def lr_lambda(epoch):
    """Staged cosine schedule: warmup -> high LR -> decay -> low LR fine-tune"""
    if epoch < C["warmup_epochs"]:
        return (epoch + 1) / C["warmup_epochs"]
    else:
        # Cosine annealing for the rest
        p = (epoch - C["warmup_epochs"]) / max(1, C["epochs"] - C["warmup_epochs"])
        return 0.5 * (1 + math.cos(math.pi * p))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# === Resume ===
start_epoch = 1
if C["resume"] and os.path.exists(C["resume"]):
    print(f"\n  Loading checkpoint: {C['resume']}")
    ckpt = torch.load(C["resume"], map_location=C["device"], weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    if args.continue_train:
        # Continue training: keep optimizer + epoch counter + scheduler state
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"  Continued from epoch {start_epoch}")
    else:
        # Fine-tune: load weights only, fresh epoch count + optimizer + scheduler
        start_epoch = 1
        print(f"  Fine-tune mode: loaded weights, training fresh for {C['epochs']} epochs")

# === 4. EMA ===
ema_model = copy.deepcopy(model)
model = model.to(C["device"])
ema_model = ema_model.to(C["device"])
for p in ema_model.parameters(): p.requires_grad_(False)
def update_ema(ema, model, decay):
    with torch.no_grad():
        for ep, mp in zip(ema.parameters(), model.parameters()):
            ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)

# === 5. Enhanced Data augmentation (B6) ===
def augment(gt):
    """Enhanced: flip + rotate 90 + brightness jitter."""
    B = gt.shape[0]
    for b in range(B):
        if np.random.rand() > 0.5: gt[b] = torch.flip(gt[b], dims=[1])
        if np.random.rand() > 0.5: gt[b] = torch.flip(gt[b], dims=[2])
        if np.random.rand() > 0.5:  # 90-degree rotation
            k = np.random.randint(1, 4)
            gt[b] = torch.rot90(gt[b], k=k, dims=[1, 2])
        if np.random.rand() > 0.5:  # brightness/contrast jitter
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-0.05, 0.05)
            gt[b] = torch.clamp(gt[b] * alpha + beta, 0, 1)
    return gt

# === 6. Training ===
print(f"\n[3/7] Training on {C['device']} (epochs {start_epoch}-{C['epochs']})...")
n_batches = C["n_train"] // C["batch_size"]
history = {"epoch": [], "loss": [], "psnr": [], "ssim": [],
           "ema_psnr": [], "ema_ssim": [], "lr": []}
best_ema_psnr = 0.0
best_epoch = 0
t_start = time.time()

for epoch in range(start_epoch, C["epochs"] + 1):
    model.train()
    perm = torch.randperm(C["n_train"])
    epoch_loss = 0.0
    epoch_loss_detail = {}
    for b in range(n_batches):
        idx = perm[b * C["batch_size"]:(b + 1) * C["batch_size"]]
        gt = train_data[idx].clone().to(C["device"])
        if C["augment_flip"]: gt = augment(gt)
        if C["gray_mask"]:
            cup.set_mask(sample_grayscale_mask(C["H"], C["W"], C["device"],
                                               exp_cdf=C["exp_cdf"], exp_centers=C["exp_centers"],
                                               blur=C["gray_blur"]))
        mask_w = make_mask_w(cup.mask, C["mask_dc_thr"], C["mask_dc_soft"]) if C["mask_dc"] else None
        E = cup.forward(gt); E = E + C["noise_std"] * torch.randn_like(E)
        I_pred, intermediates = model(E, cup.forward, cup.adjoint, mask_w)
        loss, ld = criterion(I_pred, gt, intermediates)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); update_ema(ema_model, model, C["ema_decay"])
        epoch_loss += loss.item()
        for k, v in ld.items():
            epoch_loss_detail[k] = epoch_loss_detail.get(k, 0) + v / n_batches
    scheduler.step()
    avg_loss = epoch_loss / n_batches

    # Evaluation every 5 epochs or last
    if epoch == start_epoch or epoch % 5 == 0 or epoch == C["epochs"]:
        model.eval(); ema_model.eval()
        if C["gray_mask"]:
            cup.set_mask(sample_grayscale_mask(C["H"], C["W"], C["device"],
                                               exp_cdf=C["exp_cdf"], exp_centers=C["exp_centers"],
                                               blur=C["gray_blur"], seed=C["gray_seed"]))
        mask_w = make_mask_w(cup.mask, C["mask_dc_thr"], C["mask_dc_soft"]) if C["mask_dc"] else None
        pv, sv, epv, esv = [], [], [], []
        with torch.no_grad():
            for i in range(C["n_test"]):
                gt_i = test_data[i:i+1].to(C["device"])
                E_i = cup.forward(gt_i); E_i = E_i + C["noise_std"] * torch.randn_like(E_i)
                I_p, _ = model(E_i, cup.forward, cup.adjoint, mask_w)
                I_p = torch.clamp(I_p, 0, 1)
                pv.append(compute_psnr(I_p, gt_i)); sv.append(compute_ssim(I_p, gt_i))
                I_e, _ = ema_model(E_i, cup.forward, cup.adjoint, mask_w)
                I_e = torch.clamp(I_e, 0, 1)
                epv.append(compute_psnr(I_e, gt_i)); esv.append(compute_ssim(I_e, gt_i))
        ap = float(np.mean(pv)); aps = float(np.mean(sv))
        ea = float(np.mean(epv)); eas = float(np.mean(esv))
        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch); history["loss"].append(avg_loss)
        history["psnr"].append(ap); history["ssim"].append(aps)
        history["ema_psnr"].append(ea); history["ema_ssim"].append(eas)
        history["lr"].append(lr_now)
        elapsed = time.time() - t_start
        loss_str = " ".join(f"{k}={v:.4f}" for k, v in epoch_loss_detail.items() if k != "total")
        print(f"  E{epoch:3d}/{C['epochs']} | L={avg_loss:.4f} | P={ap:.2f} S={aps:.4f} | "
              f"EMA={ea:.2f}/{eas:.4f} | LR={lr_now:.2e} | {elapsed:.0f}s")
        if loss_str:
            print(f"    [{loss_str}]")

        # Save best EMA model
        if ea > best_ema_psnr:
            best_ema_psnr = ea; best_epoch = epoch
            best_path = os.path.join(C["output_dir"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_model.state_dict(),
                "ema_psnr": ea, "ema_ssim": eas,
            }, best_path)
            print(f"    ** New best EMA PSNR={ea:.2f} (epoch {epoch})")

    # Save checkpoint every 20 epochs
    if epoch % 20 == 0 or epoch == C["epochs"]:
        ckpt_path = os.path.join(C["output_dir"], "checkpoint.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "config": C,
        }, ckpt_path)

# === 7. Save final models ===
train_time = time.time() - t_start
model_path = os.path.join(C["output_dir"], "dd_madun_v3_p4_model.pth")
ema_path = os.path.join(C["output_dir"], "dd_madun_v3_p4_ema.pth")
torch.save(model.state_dict(), model_path)
torch.save(ema_model.state_dict(), ema_path)
print(f"\n[4/7] Model saved: {model_path}")
print(f"  Best EMA PSNR: {best_ema_psnr:.2f} at epoch {best_epoch}")
print(f"  Training time: {train_time:.0f}s ({train_time/3600:.1f}h)")

# === 8. Final evaluation ===
print("[5/7] Final evaluation...")
model.eval(); ema_model.eval()
if C["gray_mask"]:
    cup.set_mask(sample_grayscale_mask(C["H"], C["W"], C["device"],
                                       exp_cdf=C["exp_cdf"], exp_centers=C["exp_centers"],
                                       blur=C["gray_blur"], seed=C["gray_seed"]))
mask_w = make_mask_w(cup.mask, C["mask_dc_thr"], C["mask_dc_soft"]) if C["mask_dc"] else None
fp, fs, efp, efs = [], [], [], []
with torch.no_grad():
    for i in range(C["n_test"]):
        gt_i = test_data[i:i+1].to(C["device"])
        E_i = cup.forward(gt_i); E_i = E_i + C["noise_std"] * torch.randn_like(E_i)
        I_p, _ = model(E_i, cup.forward, cup.adjoint, mask_w)
        I_p = torch.clamp(I_p, 0, 1)
        fp.append(float(compute_psnr(I_p, gt_i))); fs.append(float(compute_ssim(I_p, gt_i)))
        I_e, _ = ema_model(E_i, cup.forward, cup.adjoint, mask_w)
        I_e = torch.clamp(I_e, 0, 1)
        efp.append(float(compute_psnr(I_e, gt_i))); efs.append(float(compute_ssim(I_e, gt_i)))
        print(f"  S{i}: Std={fp[-1]:.2f}/{fs[-1]:.4f} EMA={efp[-1]:.2f}/{efs[-1]:.4f}")

use_ema = np.mean(efp) > np.mean(fp)
dd_psnr = np.mean(efp) if use_ema else np.mean(fp)
dd_ssim = np.mean(efs) if use_ema else np.mean(fs)
print(f"\n  DD-MADUNv3 P4: PSNR={dd_psnr:.2f} dB, SSIM={dd_ssim:.4f} ({'EMA' if use_ema else 'Std'})")

# === 9. Save history ===
hist_path = os.path.join(C["output_dir"], "history.json")
_config_safe = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in C.items()}
with open(hist_path, "w") as f:
    json.dump({"history": history, "config": _config_safe,
               "final_psnr": dd_psnr, "final_ssim": dd_ssim,
               "best_ema_psnr": best_ema_psnr, "best_epoch": best_epoch,
               "train_time_s": train_time}, f, indent=2)
print(f"\n[6/7] History: {hist_path}")

# === 10. Print learned parameters ===
print("[7/7] Learned parameters:")
params = model.get_p()
print(f"  sigma: {params['sg']}")
print(f"  rho_f: {params['rf']}")
print(f"  rho_s: {params['rs']}")
print(f"\n  Output: {os.path.abspath(C['output_dir'])}")

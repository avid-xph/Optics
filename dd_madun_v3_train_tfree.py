"""
DD-MADUN v3 -- T-FREE training (multi-T sampling)

Differences vs dd_madun_v3_train_p4.py:
  * model is the T-free variant (frame-wise denoiser, dynamic DCT matrix)
  * each training batch samples a different number of frames T
  * batch size adapts to T so that (B * T) stays within a fixed budget
  * optional gradient checkpointing and bf16 autocast (needed for large T)

Usage:
  # reproduce the old setting first (single T=8) -- do this before multi-T
  python dd_madun_v3_train_tfree.py --T_fixed 8 --epochs 150

  # then multi-T
  python dd_madun_v3_train_tfree.py --T_list 8 12 16 20 24 28 32 --epochs 150 \
      --ckpt --amp --davis_root <DAVIS2017>
"""
import sys, os, time, json, copy, math, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from cup_simulator import CUPForwardModel
from baselines import compute_psnr, compute_ssim
from video_dataset import create_mixed_dataset
from dd_madun_v3_tfree import DD_MADUNv3, DDMADUNLossv3, build_dct_matrix
from warm_start import warm_start_from_legacy

# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DD-MADUN v3 T-free training")
parser.add_argument("--T_list", type=int, nargs="+",
                    default=[8, 12, 16, 20, 24, 28, 32],
                    help="frame counts to sample during training")
parser.add_argument("--T_fixed", type=int, default=0,
                    help="if >0, train on this single T only (use for the sanity run)")
parser.add_argument("--temporal", type=str, default="none", choices=["none", "gru"],
                    help="'none'=A2 frame-wise (fast); 'gru'=A1 GRU along time")
parser.add_argument("--H", type=int, default=256)
parser.add_argument("--W", type=int, default=256)
parser.add_argument("--shear", type=int, default=1, help="pixels of shear per frame")
parser.add_argument("--K", type=int, default=8)
parser.add_argument("--K1", type=int, default=3)
parser.add_argument("--ng", type=int, default=6, help="gradient steps in the DC block")
parser.add_argument("--nf", type=int, default=96)
parser.add_argument("--nb", type=int, default=8)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--n_train", type=int, default=2000, help="total samples split across all T")
parser.add_argument("--n_test", type=int, default=8)
parser.add_argument("--real_ratio", type=float, default=0.7)
parser.add_argument("--davis_root", type=str, default="DAVIS")
parser.add_argument("--frames_per_batch", type=int, default=32,
                    help="B*T budget; B = max(1, budget//T). 32 fits most GPUs")
parser.add_argument("--noise_std", type=float, default=0.01)
parser.add_argument("--motion_blur_prob", type=float, default=0.0,
                    help="probability per training clip of applying a directional "
                         "motion-blur kernel (0 = disabled, matches old behavior); "
                         "applied both at dataset build time (video_dataset.py) and "
                         "as live per-batch augmentation below")
parser.add_argument("--motion_blur_len_min", type=int, default=3,
                    help="min motion-blur kernel length in pixels")
parser.add_argument("--motion_blur_len_max", type=int, default=15,
                    help="max motion-blur kernel length in pixels")
parser.add_argument("--ffl_weight", type=float, default=0.10)
parser.add_argument("--vgg_weight", type=float, default=0.02)
parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--profile_steps", type=int, default=0,
                    help="print detailed per-phase timing (data-aug / CUP forward / "
                         "model forward+loss / backward / optimizer step, each "
                         "torch.cuda.synchronize()'d so the numbers are trustworthy) "
                         "for the first N global training steps, then stop. Use this "
                         "to see exactly where a slow step's time is going instead of "
                         "waiting for a full epoch to print. 0 = disabled (default)")
parser.add_argument("--log_every", type=int, default=0,
                    help="print a one-line progress update every N training steps "
                         "(step index, T, running loss, elapsed) so you can see "
                         "training is actually progressing without waiting for an "
                         "epoch boundary. 0 = disabled (default, matches old behavior)")
parser.add_argument("--n_objects", type=int, default=7)
parser.add_argument("--out_dir", type=str, default="output_tfree")
parser.add_argument("--device", type=str, default="cuda")
# --- memory / speed ---
parser.add_argument("--ckpt", action="store_true",
                    help="per-unfolding-stage gradient checkpointing (K=8 recompute "
                         "segments during backward). Recommended default for large T: "
                         "meaningfully cuts peak memory (no longer all K2 denoiser "
                         "stages resident at once) at a modest recompute cost. Start "
                         "with this alone before adding --ckpt_blocks")
parser.add_argument("--ckpt_blocks", action="store_true",
                    help="ALSO checkpoint each CBAMResBlock inside every denoiser "
                         "(nested inside --ckpt's per-stage segments). Much more "
                         "aggressive memory savings but noticeably slower (extra "
                         "recompute per block, on top of the per-stage recompute) -- "
                         "only add this if --ckpt alone still runs out of memory")
parser.add_argument("--amp", action="store_true", help="bf16 autocast")
# --- init ---
parser.add_argument("--resume", type=str, default=None, help="T-free checkpoint to resume from")
parser.add_argument("--warm_start", type=str, default=None,
                    help="legacy T-bound checkpoint; adapts head/tail shapes by averaging")
parser.add_argument("--legacy_T", type=int, default=8, help="T the --warm_start model was trained on")
args = parser.parse_args()

T_LIST = [args.T_fixed] if args.T_fixed > 0 else sorted(set(args.T_list))
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)
DEV = args.device if torch.cuda.is_available() else "cpu"
# torch.autocast raises if device_type="cuda" while we actually fell back to CPU
AMP_DEVICE = "cuda" if DEV.startswith("cuda") else "cpu"

print("=" * 72)
print("  DD-MADUN v3  T-FREE training")
print(f"  T list      : {T_LIST}")
print(f"  temporal    : {args.temporal}")
print(f"  model       : K={args.K} K1={args.K1} ng={args.ng} nf={args.nf} nb={args.nb}")
print(f"  spatial     : {args.H}x{args.W}   shear={args.shear}   device={DEV}")
print(f"  B*T budget  : {args.frames_per_batch}   ckpt={args.ckpt}   ckpt_blocks={args.ckpt_blocks}   amp={args.amp}")
print("=" * 72)


# ----------------------------------------------------------------------------
def batch_size_for(T):
    """Keep B*T roughly constant so memory and gradient noise stay stable."""
    return max(1, args.frames_per_batch // T)


def _motion_blur_kernel_torch(length, angle_deg, device):
    """Build a normalized 2D linear motion-blur kernel (a straight line of
    `length` pixels through the center, rotated to `angle_deg`) as a torch
    tensor on `device`. GPU-native counterpart of video_dataset.py's
    scipy-based _motion_blur_kernel, used for live per-batch augmentation."""
    length = max(1, int(length))
    k = torch.zeros(length, length, device=device)
    center = length // 2
    k[center, :] = 1.0
    if length > 1:
        theta = math.radians(angle_deg)
        rot = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                            [math.sin(theta), math.cos(theta), 0.0]],
                           dtype=torch.float32, device=device).unsqueeze(0)
        grid = F.affine_grid(rot, [1, 1, length, length], align_corners=False)
        k = F.grid_sample(k.view(1, 1, length, length), grid,
                          align_corners=False, mode='bilinear').squeeze(0).squeeze(0)
    k = torch.clamp(k, min=0)
    s = k.sum()
    if s > 1e-8:
        k = k / s
    else:
        k = torch.zeros(length, length, device=device)
        k[center, center] = 1.0
    return k


def apply_motion_blur_torch(clip, len_min=3, len_max=15):
    """Apply ONE random directional motion-blur kernel to every frame of a
    clip. clip: (T, H, W) tensor on any device, values in [0, 1]. Same kernel
    used across all T frames, simulating one consistent blur direction for
    this clip's capture window (mirrors video_dataset.py's apply_motion_blur,
    but stays on-GPU so it can run cheaply every batch)."""
    T, H, W = clip.shape
    length = np.random.randint(len_min, len_max + 1)
    angle = np.random.uniform(0, 180)
    k = _motion_blur_kernel_torch(length, angle, clip.device)
    pad = length // 2
    x = clip.unsqueeze(0)  # (1, T, H, W): treat T as "channels" for grouped conv
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    kernel = k.view(1, 1, length, length).expand(T, 1, length, length)
    out = F.conv2d(x, kernel, groups=T)
    return torch.clamp(out.squeeze(0), 0, 1)


def augment(gt, motion_blur_prob=0.0, motion_blur_len_range=(3, 15)):
    """Flip / rotate 90 / brightness jitter (same policy as the original P4 script),
    plus optional directional motion blur applied live each time a batch is drawn
    (so the same underlying clip sees a different blur direction/strength across
    epochs, rather than one blur baked in permanently at dataset-build time)."""
    B = gt.shape[0]
    len_min, len_max = motion_blur_len_range
    for b in range(B):
        if np.random.rand() > 0.5:
            gt[b] = torch.flip(gt[b], dims=[1])
        if np.random.rand() > 0.5:
            gt[b] = torch.flip(gt[b], dims=[2])
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            gt[b] = torch.rot90(gt[b], k=k, dims=[1, 2])
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-0.05, 0.05)
            gt[b] = torch.clamp(gt[b] * alpha + beta, 0, 1)
        if motion_blur_prob > 0 and np.random.rand() < motion_blur_prob:
            gt[b] = apply_motion_blur_torch(gt[b], len_min, len_max)
    return gt


class MultiTSampler:
    """Holds one training tensor per T and hands out batches round-robin."""

    def __init__(self, data_by_T):
        self.data = data_by_T
        self.perm = {T: torch.randperm(len(d)) for T, d in data_by_T.items()}
        self.ptr = {T: 0 for T in data_by_T}

    def take(self, T, B):
        d = self.data[T]
        n = len(d)
        idx = []
        for _ in range(B):
            if self.ptr[T] >= n:
                self.perm[T] = torch.randperm(n)
                self.ptr[T] = 0
            idx.append(int(self.perm[T][self.ptr[T]]))
            self.ptr[T] += 1
        return d[idx].clone().to(DEV)


# ----------------------------------------------------------------------------
# 1. data
print("\n[1/6] Building datasets (one per T) ...")
data_by_T = {}
n_per_T = max(1, args.n_train // len(T_LIST))
for T in T_LIST:
    tr, te = create_mixed_dataset(
        n_train=n_per_T, n_test=1, T=T, H=args.H, W=args.W,
        real_ratio=args.real_ratio, davis_root=args.davis_root,
        seed=args.seed + T, n_objects=args.n_objects,
        motion_blur_prob=args.motion_blur_prob)
    data_by_T[T] = tr
    print(f"    T={T:2d}: train={tuple(tr.shape)}  (B={batch_size_for(T)})")

# fixed test set on the smallest T (comparable to the legacy baseline)
T_REF = T_LIST[0]
_, test_data = create_mixed_dataset(
    n_train=1, n_test=args.n_test, T=T_REF, H=args.H, W=args.W,
    real_ratio=0.0, davis_root=args.davis_root,
    seed=args.seed + 999, n_objects=args.n_objects)
test_data = test_data.to(DEV)
sampler = MultiTSampler(data_by_T)

# 2. model
print("\n[2/6] Building model ...")
model = DD_MADUNv3(T=T_LIST[0], H=args.H, W=args.W, K=args.K, K1=args.K1,
                   ng=args.ng, nf=args.nf, nb=args.nb, temporal=args.temporal,
                   use_checkpoint=args.ckpt, ckpt_blocks=args.ckpt_blocks)
if args.warm_start:
    model = warm_start_from_legacy(model, args.warm_start, args.legacy_T)
if args.resume and os.path.exists(args.resume):
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    print(f"    resumed from {args.resume}")
model = model.to(DEV)
print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

criterion = DDMADUNLossv3(as_=0.15, gf=0.15, sw=0.20, ew=0.05,
                          dm=build_dct_matrix(T_LIST[0]),
                          ffl_w=args.ffl_weight, vgg_w=args.vgg_weight,
                          device=DEV)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def lr_lambda(epoch):
    if epoch < args.warmup_epochs:
        return (epoch + 1) / args.warmup_epochs
    p = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * p))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

ema_model = copy.deepcopy(model).to(DEV)
for p in ema_model.parameters():
    p.requires_grad_(False)


def update_ema(ema, m, decay):
    with torch.no_grad():
        for ep, mp in zip(ema.parameters(), m.parameters()):
            ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)


def make_cup(T):
    return CUPForwardModel(H=args.H, W=args.W, T=T, shear_per_frame=args.shear,
                           real_cup=True, device=DEV).to(DEV)


def run_model(m, E, cup, mask_w=None):
    # NOTE: fine-grained gradient checkpointing (per unfolding stage, and per
    # CBAMResBlock inside each denoiser) now happens INSIDE the model itself,
    # controlled by the use_checkpoint flag passed to DD_MADUNv3(...) above.
    # Wrapping the whole model call in a single outer checkpoint here (the old
    # behavior) forced a full recompute of the ENTIRE forward graph in one
    # shot during backward, which needs just as much peak memory as no
    # checkpointing at all -- it bought nothing and cost 2x compute.
    return m(E, cup.forward, cup.adjoint, mask_w)


# 3. train
print(f"\n[3/6] Training ({args.epochs} epochs) ...")
n_batches = max(1, args.n_train // max(1, args.frames_per_batch // T_LIST[0]))
best_ema = 0.0
best_epoch = 0
history = []
t0 = time.time()

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


global_step = 0
for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss = 0.0
    detail = {}
    for b in range(n_batches):
        do_profile = args.profile_steps > 0 and global_step < args.profile_steps
        t_step0 = time.time()

        T = T_LIST[0] if len(T_LIST) == 1 else random.choice(T_LIST)
        B = batch_size_for(T)
        gt = augment(sampler.take(T, B), motion_blur_prob=args.motion_blur_prob,
                    motion_blur_len_range=(args.motion_blur_len_min, args.motion_blur_len_max))
        if do_profile: _sync(); t_data = time.time()

        cup = make_cup(T)
        E_clean = cup.forward(gt)
        E = E_clean + args.noise_std * torch.randn_like(E_clean)
        if do_profile: _sync(); t_cupfwd = time.time()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=AMP_DEVICE, dtype=torch.bfloat16, enabled=args.amp):
            out, inter = run_model(model, E, cup, None)
            loss, ld = criterion(out, gt, inter)
        if do_profile: _sync(); t_modelfwd = time.time()

        loss.backward()
        if do_profile: _sync(); t_backward = time.time()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema_model, model, args.ema_decay)
        if do_profile: _sync(); t_opt = time.time()

        if do_profile:
            print(f"  [profile] step{global_step:4d} T={T:2d} B={B} | "
                  f"data={t_data-t_step0:6.2f}s  cup_fwd={t_cupfwd-t_data:6.2f}s  "
                  f"model_fwd+loss={t_modelfwd-t_cupfwd:6.2f}s  "
                  f"backward={t_backward-t_modelfwd:6.2f}s  "
                  f"opt+ema={t_opt-t_backward:6.2f}s  "
                  f"total={t_opt-t_step0:6.2f}s", flush=True)
        elif args.log_every > 0 and global_step % args.log_every == 0:
            print(f"  [step {global_step:5d}] epoch {epoch}/{args.epochs} b {b}/{n_batches} "
                  f"T={T:2d} B={B} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

        epoch_loss += float(loss.item())
        for k, v in ld.items():
            detail[k] = detail.get(k, 0.0) + v / n_batches
        global_step += 1
    scheduler.step()

    if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
        model.eval(); ema_model.eval()
        cup = make_cup(T_REF)
        pv, sv, epv, esv = [], [], [], []
        with torch.no_grad():
            for i in range(len(test_data)):
                gti = test_data[i:i + 1]
                Ei_clean = cup.forward(gti)
                Ei = Ei_clean + args.noise_std * torch.randn_like(Ei_clean)
                for m_, pl, sl in ((model, pv, sv), (ema_model, epv, esv)):
                    o, _ = run_model(m_, Ei, cup, None)
                    o = torch.clamp(o.float(), 0, 1)
                    pl.append(compute_psnr(o, gti)); sl.append(compute_ssim(o, gti))
        ap, aps = float(np.mean(pv)), float(np.mean(sv))
        ea, eas = float(np.mean(epv)), float(np.mean(esv))
        history.append(dict(epoch=epoch, loss=epoch_loss / n_batches,
                            psnr=ap, ssim=aps, ema_psnr=ea, ema_ssim=eas))
        print(f"  E{epoch:3d}/{args.epochs} L={epoch_loss/n_batches:.4f} "
              f"P={ap:.2f}/{aps:.4f} EMA={ea:.2f}/{eas:.4f} "
              f"LR={optimizer.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s")
        if ea > best_ema:
            best_ema, best_epoch = ea, epoch
            torch.save(dict(epoch=epoch, model_state_dict=ema_model.state_dict(),
                            ema_psnr=ea, ema_ssim=eas, T_list=T_LIST,
                            temporal=args.temporal,
                            cfg=dict(K=args.K, K1=args.K1, ng=args.ng,
                                     nf=args.nf, nb=args.nb, H=args.H, W=args.W)),
                       os.path.join(args.out_dir, "best_model.pth"))
            print(f"    ** new best EMA PSNR={ea:.2f}")
    if epoch % 20 == 0 or epoch == args.epochs:
        # cfg is stored so benchmark_reconstruct.py / convert_to_tfree.py can
        # rebuild the exact architecture instead of assuming the defaults.
        torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                        ema_model_state_dict=ema_model.state_dict(),
                        temporal=args.temporal, T_list=T_LIST,
                        cfg=dict(K=args.K, K1=args.K1, ng=args.ng,
                                 nf=args.nf, nb=args.nb,
                                 temporal=args.temporal,
                                 H=args.H, W=args.W)),
                   os.path.join(args.out_dir, "checkpoint.pth"))

# 4. per-T evaluation
print("\n[4/6] Evaluating on every T ...")
model.eval(); ema_model.eval()
per_T = {}
with torch.no_grad():
    for T in T_LIST:
        cup = make_cup(T)
        ps, ss = [], []
        for i in range(min(4, len(test_data))):
            g0 = test_data[i:i + 1]
            # reuse / tile the reference clip to build a T-frame target
            if T_REF >= T:
                gt = g0[:, :T].clone()
            else:
                rep = math.ceil(T / T_REF)
                gt = g0.repeat(1, rep, 1, 1)[:, :T].clone()
            Ec = cup.forward(gt)
            E = Ec + args.noise_std * torch.randn_like(Ec)
            o, _ = run_model(ema_model, E, cup, None)
            o = torch.clamp(o.float(), 0, 1)
            ps.append(compute_psnr(o, gt)); ss.append(compute_ssim(o, gt))
        per_T[T] = (float(np.mean(ps)), float(np.mean(ss)))
        print(f"    T={T:2d}  PSNR={per_T[T][0]:.2f}  SSIM={per_T[T][1]:.4f}")

# 5. save
torch.save(ema_model.state_dict(), os.path.join(args.out_dir, "final_ema.pth"))
with open(os.path.join(args.out_dir, "history.json"), "w") as f:
    json.dump(dict(history=history, per_T=per_T, best_ema=best_ema,
                   best_epoch=best_epoch, T_list=T_LIST,
                   temporal=args.temporal), f, indent=2)

print("\n[5/6] Done.")
print(f"  best EMA PSNR={best_ema:.2f} @ epoch {best_epoch}")
print(f"  time: {(time.time()-t0)/3600:.2f} h")
print(f"  output: {os.path.abspath(args.out_dir)}")

"""Forward-only check of the PR #7 DC-adjoint scaling (T_REF/T).

No training, no backward: rebuild the warm-start T-free model from the legacy
T=8 checkpoint, push one synthetic measurement through it at T=8/16/24/32, and
report the reconstruction PSNR with the scaling on (T_REF=8) and off (T_REF=T).

  python diag_dc_scale.py

A healthy T-invariant model gives a finite PSNR at every T. Before PR #7 the
T=24/32 loss blew up to 1e6-1e9, which shows up here as PSNR -> -inf / NaN.
"""
import torch
import dd_madun_v3_tfree as M
from cup_simulator import CUPForwardModel
from warm_start import warm_start_from_legacy
from ckpt_utils import resolve_model_cfg

CKPT = "checkpoints/best_model.pth"
H = W = 64

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = resolve_model_cfg(ck)
print(f"cfg: {cfg}\n")

torch.manual_seed(0)
print(f"{'T':>3}  {'scaled PSNR':>12} {'max|x|':>9}   {'off PSNR':>12} {'max|x|':>9}")
for T in (8, 16, 24, 32):
    model = M.DD_MADUNv3(T=T, H=H, W=W, K=cfg["K"], K1=cfg["K1"], ng=cfg["ng"],
                         nf=cfg["nf"], nb=cfg["nb"], temporal=cfg["temporal"])
    warm_start_from_legacy(model, CKPT, legacy_T=8)
    model = model.cuda().eval()
    cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=1, device="cuda")
    gt = torch.rand(1, T, H, W, device="cuda")
    row = []
    with torch.no_grad():
        E = cup.forward(gt)
        for ref in (8, T):                      # T_REF=8 -> scaling on; T_REF=T -> off
            M.T_REF = ref
            out = model(E, cup.forward, cup.adjoint)
            out = out[0] if isinstance(out, tuple) else out
            mse = ((out - gt) ** 2).mean().item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-30))).item()
            row.append((psnr, out.abs().max().item()))
    M.T_REF = 8
    print(f"{T:3d}  {row[0][0]:12.2f} {row[0][1]:9.3f}   {row[1][0]:12.2f} {row[1][1]:9.3f}")

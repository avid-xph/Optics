"""A/B: should the PR #7 DC scaling also shrink the init x0 = adj(E)?

PR #7 scales the whole adjoint by s = T_REF/T to keep ||A^T A|| (and hence the
learned DC step sizes / momentum) at the T_REF level. It does that in two places:

  * the data-term GRADIENT inside DataConsistencyMomentum, adj(fwd(xh)-E) -- this
    is the term that actually grows with T and causes the divergence;
  * the INIT x0 = adj(E) in DD_MADUNv3.forward.

The second one is questionable: ||adj(E)|| does NOT grow with T (the adjoint
copies one measurement back into T frames, it does not sum T frames), so
shrinking x0 by T_REF/T starts the unfolding from an x that is systematically
too small at large T. Only the gradient term needs the fix.

This script measures both variants, forward-only (no training, no backward),
and also reports ||A^T A|| and ||adj(E)|| vs T so the claim above is checkable.

  python diag_dc_init.py
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


def op_norm(cup, T, iters=30):
    """||A^T A|| by power iteration."""
    x = torch.randn(1, T, H, W, device="cuda")
    lam = 1.0
    for _ in range(iters):
        x = cup.adjoint(cup.forward(x))
        lam = x.norm().item()
        x = x / (lam + 1e-12)
    return lam


for seed in (0, 1):
  torch.manual_seed(seed)
  print(f"\n--- seed {seed} ---")
  # r = ||x0|| / ||gt|| : how badly the un-normalized adjoint overshoots the
  # ground truth. The legacy model was TRAINED with r at T=8; a warm-started
  # model behaves most like it was trained when r stays near that value.
  print(f"{'T':>3} {'||A^T A||':>10} {'r=||x0||/||gt||':>17} | "
        f"{'A: scale x0':>22} | {'B: keep x0':>22}")
  print(f"{'':>3} {'':>10} {'A':>8} {'B':>8} | "
        f"{'PSNR':>10} {'max|x|':>11} | {'PSNR':>10} {'max|x|':>11}")
  for T in (8, 16, 24, 32):
    model = M.DD_MADUNv3(T=T, H=H, W=W, K=cfg["K"], K1=cfg["K1"], ng=cfg["ng"],
                         nf=cfg["nf"], nb=cfg["nb"], temporal=cfg["temporal"])
    warm_start_from_legacy(model, CKPT, legacy_T=8)
    model = model.cuda().eval()
    cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=1, device="cuda")
    gt = torch.rand(1, T, H, W, device="cuda")

    lam = op_norm(cup, T)
    s = M.T_REF / T
    rows = []
    with torch.no_grad():
        E = cup.forward(gt)
        x0 = cup.adjoint(E)
        rB = (x0.norm() / gt.norm()).item()
        rA = rB * s
        for scale_x0 in (True, False):
            M.SCALE_X0 = scale_x0
            out = model(E, cup.forward, cup.adjoint)
            out = out[0] if isinstance(out, tuple) else out
            mse = ((out - gt) ** 2).mean().item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-30))).item()
            rows.append((psnr, out.abs().max().item()))
    M.SCALE_X0 = True
    print(f"{T:3d} {lam:10.2f} {rA:8.2f} {rB:8.2f} | "
          f"{rows[0][0]:10.2f} {rows[0][1]:11.3f} | "
          f"{rows[1][0]:10.2f} {rows[1][1]:11.3f}")

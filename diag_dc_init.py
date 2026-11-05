"""A/B: should the PR #7 DC scaling also shrink the init x0 = adj(E)?

PR #7 scales the whole adjoint by s = T_REF/T to keep ||A^T A|| (and hence the
learned DC step sizes / momentum) at the T_REF level. It does that in two places:

  * the data-term GRADIENT inside DataConsistencyMomentum, adj(fwd(xh)-E);
  * the INIT x0 = adj(E) in DD_MADUNv3.forward.

Both grow ~linearly with T: E is the sum of T sheared frames, so ||E|| ~ T;
the adjoint copies E back into T frames, giving each frame ~T x the single-frame
magnitude. Measured ||A^T A|| on 256x256 (x=rand): T=8/16/24/32 = 2.23/4.17/6.07/7.93,
roughly linear. Scaling x0 by T_REF/T keeps the init at the T=8 (legacy) level.

This script measures both variants (scale x0 vs keep x0), forward-only, and
reports ||A^T A|| and ||adj(E)|| vs T so the claim is checkable.

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
        rB = (x0.norm() / gt.norm()).item()   # unscaled (variant B)
        rA = rB * s                            # scaled (variant A)

        # Variant A: production code (x0 always scaled by s in forward).
        outA = model(E, cup.forward, cup.adjoint)
        outA = outA[0] if isinstance(outA, tuple) else outA
        mseA = ((outA - gt) ** 2).mean().item()
        psnrA = 10 * torch.log10(torch.tensor(1.0 / max(mseA, 1e-30))).item()
        rows.append((psnrA, outA.abs().max().item()))

        # Variant B: no x0 scaling, data-term gradient still scaled.
        # forward() computes x0 = adj(E) as its FIRST adjoint call and then
        # multiplies by s; every later call goes through adj*s (DC gradient).
        # Dividing only that first call by s cancels the init scaling and
        # leaves the real forward untouched.
        calls = [0]

        def adj_B(r):
            calls[0] += 1
            out = cup.adjoint(r)
            return out / s if calls[0] == 1 else out

        outB = model(E, cup.forward, adj_B)
        outB = outB[0] if isinstance(outB, tuple) else outB
        mseB = ((outB - gt) ** 2).mean().item()
        psnrB = 10 * torch.log10(torch.tensor(1.0 / max(mseB, 1e-30))).item()
        rows.append((psnrB, outB.abs().max().item()))

    print(f"{T:3d} {lam:10.2f} {rA:8.2f} {rB:8.2f} | "
          f"{rows[0][0]:10.2f} {rows[0][1]:11.3f} | "
          f"{rows[1][0]:10.2f} {rows[1][1]:11.3f}")

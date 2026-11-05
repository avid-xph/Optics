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

        # Variant B: no x0 scaling.  We achieve this by passing a pre-scaled
        # measurement E*B_scale to the model and an adjoint that is pre-divided
        # by s, so the model's internal adj(E)*s cancels out for x0 but NOT for
        # the data-term gradient (which is correct — we only want to test the
        # init).  Simplest correct approach: directly swap the model's forward
        # by temporarily replacing it.
        orig_fwd = M.DD_MADUNv3.forward

        def _fwd_noscale(self, E, fwd, adj, mask_w=None):
            B = E.shape[0]
            x = adj(E)                    # unscaled init
            T = x.shape[1]
            s = M.T_REF / T
            if s != 1.0:
                adj = lambda r, _adj=adj: _adj(r) * s   # gradient still scaled
                # NOTE: deliberately skip x=x*s (variant B only)
            dm = self._get_dm(T, x.device)
            if mask_w is not None:
                xb = x * mask_w
                msum = mask_w.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)
                mmean = xb.sum(dim=[2, 3], keepdim=True) / msum
            else:
                mmean = None
            xs = x * 0.5 + 0.5 if x.min() < 0 else x
            xt = M.dct_along_time(xs, dm)
            for i in range(self.K):
                xt = self.blocks[i](xt)
                if i < self.K - self.K1:
                    xd = M.idct_along_time(xt, dm)
                    xd = self.dc(xd, E, fwd, adj, mask_w, mmean)
                    xt = M.dct_along_time(xd, dm)
            out = M.idct_along_time(xt, dm)
            return out

        M.DD_MADUNv3.forward = _fwd_noscale
        outB = model(E, cup.forward, cup.adjoint)
        outB = outB[0] if isinstance(outB, tuple) else outB
        mseB = ((outB - gt) ** 2).mean().item()
        psnrB = 10 * torch.log10(torch.tensor(1.0 / max(mseB, 1e-30))).item()
        rows.append((psnrB, outB.abs().max().item()))
        M.DD_MADUNv3.forward = orig_fwd   # restore

    print(f"{T:3d} {lam:10.2f} {rA:8.2f} {rB:8.2f} | "
          f"{rows[0][0]:10.2f} {rows[0][1]:11.3f} | "
          f"{rows[1][0]:10.2f} {rows[1][1]:11.3f}")

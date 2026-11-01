"""Quick correctness checks for dd_madun_tfree (CPU, small sizes, ~1 min).

  python check_tfree.py

1. Dynamic DCT: orthonormal, invertible, cached.
2. Interface: one set of weights runs for several T, with/without mask_w;
   output shapes are right and gradients reach the denoiser.
3. Equivalence: with K1=K (no denoiser) the output equals DD_MADUNv3.
Every failed check prints the measured value and what it means.
"""
import torch
from cup_simulator import CUPForwardModel
from dd_madun_v3 import DD_MADUNv3, dct_along_time, idct_along_time
from dd_madun_tfree import DD_MADUN_TFree, get_dct_matrix

torch.manual_seed(0)
H = W = 32; B = 2; K = 4
print(f"torch {torch.__version__}")

# 1. DCT
for T in [8, 12, 32]:
    dm = get_dct_matrix(T)
    e = (dm @ dm.T - torch.eye(T)).abs().max().item()
    assert e < 1e-5, f"[1] T={T}: DCT not orthonormal, max|D D^T - I| = {e:.2e}"
    x = torch.rand(B, T, H, W)
    e = (idct_along_time(dct_along_time(x, dm), dm) - x).abs().max().item()
    assert e < 1e-5, f"[1] T={T}: idct(dct(x)) != x, max err = {e:.2e}"
    assert get_dct_matrix(T) is dm, f"[1] T={T}: DCT cache miss"
print("[1/3] DCT ok")

# 2. Interface / shapes / grads, same weights for every T
model = DD_MADUN_TFree(H=H, W=W, K=K, K1=2, ng=2, nf=8, nb=2)
for T in [8, 12, 20, 32]:
    cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=1)
    gt = torch.rand(B, T, H, W)
    E = cup.forward(gt)
    for mw in [None, (cup.mask > 0.5).float().view(1, 1, H, W)]:
        x, inter = model(E, cup.forward, cup.adjoint, mw)
        tag = f"[2] T={T} mask_w={'on' if mw is not None else 'off'}"
        assert x.shape == (B, T, H, W), f"{tag}: output shape {tuple(x.shape)} != {(B, T, H, W)}"
        assert len(inter) == K, f"{tag}: len(inter)={len(inter)} != K={K}"
        assert torch.isfinite(x).all(), f"{tag}: output has NaN/Inf (max|x|={x.abs().nan_to_num().max():.2e})"
    model.zero_grad(); (x - gt).abs().mean().backward()
    g = sum(p.grad.abs().sum() for p in model.den.parameters() if p.grad is not None)
    assert g > 0, f"[2] T={T}: no gradient reaches the denoiser"
    print(f"      T={T}: shape ok, max|x|={x.abs().max():.3f}, denoiser grad sum={g:.3e}")
print("[2/3] interface ok")

# 3. Without denoiser, identical to v3
def rel_diff(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)

for T in [8, 16]:
    cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=1)
    E = cup.forward(torch.rand(B, T, H, W))
    E0 = E.clone()
    m_old = DD_MADUNv3(T=T, H=H, W=W, K=K, K1=K, ng=2)
    m_new = DD_MADUN_TFree(H=H, W=W, K=K, K1=K, ng=2)
    with torch.no_grad():
        a1, _ = m_old(E.clone(), cup.forward, cup.adjoint)
        b1, _ = m_new(E.clone(), cup.forward, cup.adjoint)
        a2, _ = m_old(E.clone(), cup.forward, cup.adjoint)
        b2, _ = m_new(E.clone(), cup.forward, cup.adjoint)
        fa, fb = cup.forward(a1), cup.forward(a1)   # same input twice
    r_v3, r_tf, r_x = rel_diff(a1, a2), rel_diff(b1, b2), rel_diff(a1, b1)
    r_fwd = rel_diff(fa, fb)
    print(f"      T={T}: v3 vs v3 = {r_v3:.2e} | tfree vs tfree = {r_tf:.2e} | v3 vs tfree = {r_x:.2e} | "
          f"cup.forward twice = {r_fwd:.2e} | E modified in place = {not torch.equal(E, E0)}")
    assert torch.isfinite(a1).all() and torch.isfinite(b1).all(), \
        f"[3] T={T}: NaN/Inf in output (unrolled iterations diverged)"
    assert r_fwd < 1e-6 and r_v3 < 1e-6, \
        (f"[3] T={T}: the SAME model gives different results on repeated runs "
         f"(cup.forward repeat diff {r_fwd:.2e}, v3 repeat diff {r_v3:.2e}). "
         f"The forward model / environment is non-deterministic, so v3-vs-tfree cannot be compared. "
         f"Check cup_simulator.py on this machine (it differs from the reference copy).")
    assert r_x < 1e-5, f"[3] T={T}: tfree differs from v3 (relative diff {r_x:.2e})"
print("[3/3] equivalence with v3 ok")
print("ALL CHECKS PASSED")

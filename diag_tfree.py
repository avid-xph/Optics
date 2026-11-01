"""Diagnose why check_tfree.py [3] fails (tfree != v3 without denoiser).

  python diag_tfree.py

Prints: which files are actually imported (path + MD5), initial values of the
learnable scalars in both models, the first unrolled stage where outputs
diverge, and a diff of the two forward() sources.
"""
import hashlib, inspect, difflib
import torch
import dd_madun_v3, dd_madun_tfree, cup_simulator
from cup_simulator import CUPForwardModel
from dd_madun_v3 import DD_MADUNv3
from dd_madun_tfree import DD_MADUN_TFree


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


print("== imported files ==")
for m in (dd_madun_v3, dd_madun_tfree, cup_simulator):
    p = inspect.getfile(m)
    print(f"  {m.__name__:16s} {md5(p)}  {p}")

torch.manual_seed(0)
H = W = 32; B = 2; K = 4; T = 8
cup = CUPForwardModel(H=H, W=W, T=T, shear_per_frame=1)
E = cup.forward(torch.rand(B, T, H, W))
m_old = DD_MADUNv3(T=T, H=H, W=W, K=K, K1=K, ng=2)
m_new = DD_MADUN_TFree(H=H, W=W, K=K, K1=K, ng=2)

print("\n== learnable scalars at init (stage 0) ==")
po, pn = m_old.get_p(), m_new.get_p()
for k in po:
    a = po[k][0] if po[k] else None; b = pn[k][0] if pn[k] else None
    flag = "" if a == b else "   <-- DIFFERENT"
    print(f"  {k}: v3={a}  tfree={b}{flag}")
print(f"  v3 has buffer dm: {'dm' in dict(m_old.named_buffers())}, "
      f"v3 dm == tfree dm: {torch.allclose(m_old.dm, m_new.get_dm(T))}")

print("\n== per-stage difference ==")
with torch.no_grad():
    a, ia = m_old(E, cup.forward, cup.adjoint)
    b, ib = m_new(E, cup.forward, cup.adjoint)
    print(f"  init adj(E) max: {cup.adjoint(E).abs().max():.4f}")
    for k, (xa, xb) in enumerate(zip(ia, ib)):
        d = (xa - xb).abs().max().item()
        print(f"  stage {k}: max|v3-tfree| = {d:.3e}  (max|v3|={xa.abs().max():.3f})"
              + ("   <-- first divergence" if d > 1e-5 and all((ia[j] - ib[j]).abs().max() <= 1e-5 for j in range(k)) else ""))

print("\n== diff of forward() source: v3 (-) vs tfree (+) ==")
src_old = inspect.getsource(DD_MADUNv3.forward).splitlines()
src_new = inspect.getsource(DD_MADUN_TFree.forward).splitlines()
for line in difflib.unified_diff(src_old, src_new, "v3.forward", "tfree.forward", lineterm="", n=1):
    print("  " + line)

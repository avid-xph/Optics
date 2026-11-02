"""迭代法的纯 torch 实现（GPU 上跑, 不经 numpy）.

与 baselines.py 的 real_cup 分支同一套算法与默认超参:
  twist    : IST + 两步加速, 步长 1/L（L 由幂迭代估计）
  admm_l1  : ADMM, x 子问题 (AᵀA + ρI) x = Aᵀy + ρ(z - u) 用 CG
  admm_tv  : ADMM + 各向异性 TV, x 子问题 (AᵀA + ρDᵀD) x = Aᵀy + ρDᵀ(z - p)
             所有帧一起解（baselines.py 是逐帧解, 每帧都算一次完整 AᵀA, O(T²)）
fwd: (B,T,H,W) -> (B,H,Wm), adj: (B,H,Wm) -> (B,T,H,W), 即 CUPForwardModel 的接口。
"""
import torch


def _dot(a, b):
    return (a * b).sum()


def _cg(op, b, x, n_iter):
    r = b - op(x)
    p = r.clone()
    rs = _dot(r, r)
    for _ in range(n_iter):
        Ap = op(p)
        alpha = rs / _dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _dot(r, r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _lipschitz(fwd, adj, shape, device, n_iter=30):
    v = torch.randn(shape, device=device)
    for _ in range(n_iter):
        v = adj(fwd(v))
        v = v / v.norm()
    return adj(fwd(v)).norm()


@torch.no_grad()
def twist(E, fwd, adj, T, n_iters=200, lambda_tv=0.01, alpha=0.9):
    x = adj(E)
    beta = 1.0 / _lipschitz(fwd, adj, x.shape, E.device)
    thresh = lambda_tv * beta
    x_prev = x
    for it in range(n_iters):
        x_half = torch.clamp(x - beta * adj(fwd(x) - E), -100, 100)
        x_new = torch.sign(x_half) * torch.clamp(x_half.abs() - thresh, min=0)
        if it > 0:
            x_new = (1 - alpha) * x_prev + alpha * x_new
        x_prev, x = x, x_new
    return torch.clamp(x, 0, 1)


@torch.no_grad()
def admm_l1(E, fwd, adj, T, n_iters=50, rho=0.1, n_inner=5):
    ATy = adj(E)
    x = ATy
    z = x.clone()
    u = torch.zeros_like(x)
    op = lambda v: adj(fwd(v)) + rho * v
    for _ in range(n_iters):
        x = _cg(op, ATy + rho * (z - u), x, n_inner)
        v = x + u
        z = torch.sign(v) * torch.clamp(v.abs() - 1.0 / rho, min=0)
        u = u + x - z
    return x


def _grad(x):
    """前向差分 (与 baselines.tv_gradient 相同, 循环边界): (...,H,W) -> (2,...,H,W)."""
    return torch.stack([x - torch.roll(x, -1, -1), x - torch.roll(x, -1, -2)])


def _div(p):
    """_grad 的伴随 Dᵀ（即 baselines.tv_divergence）."""
    return (p[0] - torch.roll(p[0], 1, -1)) + (p[1] - torch.roll(p[1], 1, -2))


@torch.no_grad()
def admm_tv(E, fwd, adj, T, n_iters=50, rho=0.5, lambda_tv=0.01, n_inner=5):
    ATy = adj(E)
    x = ATy
    z = torch.zeros((2,) + x.shape, device=x.device)
    p = torch.zeros_like(z)
    op = lambda v: adj(fwd(v)) + rho * _div(_grad(v))
    for _ in range(n_iters):
        x = torch.clamp(_cg(op, ATy + rho * _div(z - p), x, n_inner), -5.0, 5.0)
        g = _grad(x) + p
        mag = torch.sqrt(g[0] ** 2 + g[1] ** 2 + 1e-10)
        z = g * torch.clamp(1 - lambda_tv / rho / mag, min=0)
        p = g - z
    return x

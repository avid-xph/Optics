"""
Baseline reconstruction methods for CUP:
  1. TwIST (Two-step Iterative Shrinkage/Thresholding)
  2. ADMM-TV (ADMM with Total Variation regularization)
  3. ADMM-L1 (ADMM with L1 sparsity regularization)

These serve as classical baselines to compare against MADUN.
"""

import torch
import numpy as np
from scipy.sparse.linalg import cg as conjugate_gradient, LinearOperator


# ============================================================================
# Helper: real_cup operator wrapping
# ============================================================================

def _real_cup_ops(forward_fn, adjoint_fn, T, H, W):
    """Wrap the torch real_cup forward/adjoint operators into numpy A/AT
    functions that handle the wider measurement (Wm = W + (T-1)*shear).

    forward_fn / adjoint_fn may be either a CUPForwardModel instance
    (then .forward / .adjoint are used) or the bound methods directly.
    Returns (A, AT, Wm).
    """
    model = getattr(forward_fn, '__self__', forward_fn)
    fwd = forward_fn if hasattr(forward_fn, '__self__') else model.forward
    adj = adjoint_fn if hasattr(adjoint_fn, '__self__') else model.adjoint
    device = model.device
    shear = model.shear
    Wm = W + (T - 1) * shear

    def A(x):
        xt = torch.from_numpy(x.reshape(T, H, W)).float().to(device)
        E = fwd(xt.unsqueeze(0)).squeeze(0)  # (H, Wm)
        return E.detach().cpu().numpy().flatten()

    def AT(y):
        yt = torch.from_numpy(y.reshape(H, Wm)).float().to(device)
        I = adj(yt.unsqueeze(0)).squeeze(0)  # (T, H, W)
        return I.detach().cpu().numpy().flatten()

    return A, AT, Wm


# ============================================================================
# Helper: Image quality metrics
# ============================================================================

def compute_psnr(pred, gt, max_val=1.0):
    """Compute PSNR between prediction and ground truth."""
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred, gt, max_val=1.0):
    """Compute SSIM between prediction and ground truth (simplified)."""
    pred = pred.detach().cpu().numpy().squeeze()
    gt = gt.detach().cpu().numpy().squeeze()

    # Compute per-frame SSIM
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    if pred.ndim == 3:  # (T, H, W)
        ssim_vals = []
        for t in range(pred.shape[0]):
            p = pred[t].flatten()
            g = gt[t].flatten()
            mu_p = p.mean()
            mu_g = g.mean()
            sigma_p = p.var()
            sigma_g = g.var()
            sigma_pg = ((p - mu_p) * (g - mu_g)).mean()

            ssim_val = ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / \
                       ((mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p + sigma_g + C2))
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    else:  # 2D
        p = pred.flatten()
        g = gt.flatten()
        mu_p = p.mean()
        mu_g = g.mean()
        sigma_p = p.var()
        sigma_g = g.var()
        sigma_pg = ((p - mu_p) * (g - mu_g)).mean()
        return ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / \
               ((mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p + sigma_g + C2))


# ============================================================================
# TwIST (Two-step Iterative Shrinkage/Thresholding)
# ============================================================================

def twist_reconstruct(measurement, forward_fn, adjoint_fn, T, H, W,
                      n_iters=200, lambda_tv=0.01, alpha=0.9, beta=1.0,
                      verbose=False, real_cup=False):
    """
    TwIST reconstruction for CUP measurements.

    TwIST combines IST (Iterative Soft-Thresholding) with two-step acceleration.
    When real_cup=True, the operators A/AT are taken from the physically-accurate
    CUP model (wider measurement canvas), instead of the legacy circular-shift A.
    """
    if real_cup:
        A, AT, Wm = _real_cup_ops(forward_fn, adjoint_fn, T, H, W)
        E_np = measurement.detach().cpu().numpy().squeeze()  # (H, Wm)
    else:
        E_np = measurement.detach().cpu().numpy().squeeze()
        mask = forward_fn.mask.cpu().numpy() if hasattr(forward_fn, 'mask') else \
               np.random.binomial(1, 0.5, (H, W)).astype(np.float32)
        shear = getattr(forward_fn, 'shear', 2)

        def A(x):
            """Forward operator on flattened input."""
            I = x.reshape(T, H, W)
            E_out = np.zeros((H, W), dtype=np.float64)
            for t in range(T):
                encoded = I[t] * mask
                sheared = np.roll(encoded, t * shear, axis=1)
                E_out += sheared
            return E_out.flatten()

        def AT(y):
            """Adjoint operator on flattened input."""
            E_in = y.reshape(H, W)
            I_out = np.zeros((T, H, W), dtype=np.float64)
            for t in range(T):
                unshifted = np.roll(E_in, -t * shear, axis=1)
                I_out[t] = unshifted * mask
            return I_out.flatten()

    n = T * H * W
    x = AT(E_np.flatten())  # Initialize

    # Estimate Lipschitz constant L = ||A^T A||_2 via power iteration so the
    # IST step beta = 1/L is stable (critical for the ill-posed shear=1 case,
    # where a fixed beta=1.0 diverges to noise).
    if real_cup:
        v = np.random.randn(n)
        for _ in range(30):
            v = AT(A(v))
            nv = np.linalg.norm(v)
            if nv > 0:
                v /= nv
        L = np.linalg.norm(AT(A(v))) / max(np.linalg.norm(v), 1e-12)
        beta = 1.0 / max(L, 1e-6)
        if verbose:
            print(f"  TwIST: estimated L={L:.3f}, beta={beta:.5f}")

    x_prev = x.copy()
    for it in range(n_iters):
        # Gradient step
        grad = AT(A(x) - E_np.flatten())
        x_half = x - beta * grad
        x_half = np.clip(x_half, -100, 100)  # Prevent overflow

        # Soft-thresholding (proximal for L1)
        thresh = lambda_tv * beta
        x_new = np.sign(x_half) * np.maximum(np.abs(x_half) - thresh, 0)

        # Two-step update (TwIST acceleration)
        if it > 0:
            x_new = (1 - alpha) * x_prev + alpha * x_new

        x_prev = x
        x = x_new

    x = np.clip(x, 0, 1)  # Final clipping to valid range
    return torch.from_numpy(x.reshape(1, T, H, W).astype(np.float32))


# ============================================================================
# ADMM with L1 sparsity (closer comparison to MADUN's sparse stages)
# ============================================================================

def admm_l1_reconstruct(measurement, forward_fn, adjoint_fn, T, H, W,
                        n_iters=50, rho=0.1, n_inner=5, verbose=False,
                        real_cup=False):
    """
    ADMM with L1 sparsity regularization.

    min_I (1/2)*||A(I) - E||^2 + lambda*||I||_1

    When real_cup=True, the operators A/AT are taken from the physically-accurate
    CUP model (wider measurement canvas), instead of the legacy circular-shift A.
    """
    if real_cup:
        A, AT, Wm = _real_cup_ops(forward_fn, adjoint_fn, T, H, W)
        E = measurement.squeeze().detach().cpu().numpy()  # (H, Wm)
    else:
        E = measurement.squeeze().detach().cpu().numpy()
        mask = forward_fn.mask.cpu().numpy() if hasattr(forward_fn, 'mask') else \
               np.random.binomial(1, 0.5, (H, W)).astype(np.float32)
        shear = getattr(forward_fn, 'shear', 2)

        def A(x):
            I = x.reshape(T, H, W)
            E_out = np.zeros((H, W), dtype=np.float64)
            for t in range(T):
                encoded = I[t] * mask
                sheared = np.roll(encoded, t * shear, axis=1)
                E_out += sheared
            return E_out.flatten()

        def AT(y):
            E_in = y.reshape(H, W)
            I_out = np.zeros((T, H, W), dtype=np.float64)
            for t in range(T):
                unshifted = np.roll(E_in, -t * shear, axis=1)
                I_out[t] = unshifted * mask
            return I_out.flatten()

    n = T * H * W
    I_cur = AT(E.flatten())
    z = I_cur.copy()
    u = np.zeros(n)

    for it in range(n_iters):
        # I-update: solve (A^T A + rho I) I = A^T E + rho(z - u) via CG
        b = AT(E.flatten()) + rho * (z - u)

        def ATA_plus_rhoI(v):
            return AT(A(v)) + rho * v

        A_op = LinearOperator((n, n), matvec=ATA_plus_rhoI, dtype=np.float64)
        I_cur, _ = conjugate_gradient(
            A_op, b,
            x0=I_cur, maxiter=n_inner, rtol=1e-4
        )

        # z-update: soft-thresholding
        v = I_cur + u
        z = np.sign(v) * np.maximum(np.abs(v) - 1.0 / rho, 0)

        # u-update
        u = u + I_cur - z

        if verbose and it % 20 == 0:
            residual = np.linalg.norm(A(I_cur) - E.flatten())
            print(f"  ADMM-L1 iter {it}: residual={residual:.4f}")

    return torch.from_numpy(I_cur.reshape(1, T, H, W).astype(np.float32))


# ============================================================================
# ADMM with Total Variation (TV) regularization
# ============================================================================

def tv_gradient(x_2d):
    """Compute TV gradient for a 2D image."""
    dx = x_2d - np.roll(x_2d, -1, axis=1)
    dy = x_2d - np.roll(x_2d, -1, axis=0)
    return np.stack([dx, dy], axis=0)


def tv_divergence(p):
    """Divergence of a 2D vector field."""
    px, py = p[0], p[1]
    div_x = px - np.roll(px, 1, axis=1)
    div_y = py - np.roll(py, 1, axis=0)
    return div_x + div_y


def admm_tv_reconstruct(measurement, forward_fn, adjoint_fn, T, H, W,
                        n_iters=50, rho=0.5, lambda_tv=0.01, n_inner=5,
                        verbose=False, real_cup=False):
    """
    ADMM with isotropic Total Variation regularization.

    min_I (1/2)*||A(I) - E||^2 + lambda*TV(I)

    When real_cup=True, the operators A/AT are taken from the physically-accurate
    CUP model (wider measurement canvas), instead of the legacy circular-shift A.
    """
    if real_cup:
        A, AT, Wm = _real_cup_ops(forward_fn, adjoint_fn, T, H, W)
        E = measurement.squeeze().detach().cpu().numpy()  # (H, Wm)
    else:
        E = measurement.squeeze().detach().cpu().numpy()
        mask = forward_fn.mask.cpu().numpy() if hasattr(forward_fn, 'mask') else \
               np.random.binomial(1, 0.5, (H, W)).astype(np.float32)
        shear = getattr(forward_fn, 'shear', 2)

        def A(x):
            I = x.reshape(T, H, W)
            E_out = np.zeros((H, W), dtype=np.float64)
            for t in range(T):
                encoded = I[t] * mask
                sheared = np.roll(encoded, t * shear, axis=1)
                E_out += sheared
            return E_out.flatten()

        def AT(y):
            E_in = y.reshape(H, W)
            I_out = np.zeros((T, H, W), dtype=np.float64)
            for t in range(T):
                unshifted = np.roll(E_in, -t * shear, axis=1)
                I_out[t] = unshifted * mask
            return I_out.flatten()

    n = T * H * W
    I_cur = AT(E.flatten())

    # Initialize dual variables for TV (per-frame)
    p = [np.zeros((2, H, W), dtype=np.float64) for _ in range(T)]
    z = [np.zeros((2, H, W), dtype=np.float64) for _ in range(T)]

    for it in range(n_iters):
        # I-update: per-frame CG solve
        I_mat = I_cur.reshape(T, H, W)

        for t in range(T):
            b_frame = AT(E.flatten())[t * H * W:(t + 1) * H * W].reshape(H, W)
            b_frame = b_frame + rho * (tv_divergence(z[t]) - tv_divergence(p[t]))

            def ATA_t(v):
                v_flat = np.zeros(n, dtype=np.float64)
                v_flat[t * H * W:(t + 1) * H * W] = v.flatten()
                result = AT(A(v_flat))
                return result[t * H * W:(t + 1) * H * W].reshape(H, W) + rho * v.flatten().reshape(H, W)

            x0 = I_mat[t].flatten()
            def mv_fn(v):
                return ATA_t(v.reshape(H, W)).flatten()
            A_t = LinearOperator((H*W, H*W), matvec=mv_fn, dtype=np.float64)
            frame_sol, _ = conjugate_gradient(
                A_t,
                b_frame.flatten(),
                x0=x0, maxiter=n_inner, rtol=1e-4
            )
            # Clip to a safe range: CUP intensities are in [0,1], and the
            # ill-posed shear=1 operator makes CG diverge to +/-inf otherwise.
            frame_sol = np.clip(frame_sol, -5.0, 5.0)
            I_mat[t] = frame_sol.reshape(H, W)

        # z-update: TV proximal (soft-thresholding of gradient)
        for t in range(T):
            grad_I = tv_gradient(I_mat[t])
            combined = grad_I + p[t]
            norm_combined = np.sqrt(combined[0] ** 2 + combined[1] ** 2 + 1e-10)
            z[t] = combined * np.maximum(1 - lambda_tv / rho / norm_combined, 0)

        # p-update
        for t in range(T):
            p[t] = p[t] + tv_gradient(I_mat[t]) - z[t]

        I_cur = I_mat.flatten()

        if verbose and it % 20 == 0:
            residual = np.linalg.norm(A(I_cur) - E.flatten())
            print(f"  ADMM-TV iter {it}: residual={residual:.4f}")

    return torch.from_numpy(I_cur.reshape(1, T, H, W).astype(np.float32))


# ============================================================================
# Back-projection baseline (simplest)
# ============================================================================

def bp_reconstruct(measurement, adjoint_fn):
    """Simple back-projection reconstruction (A^T E)."""
    return adjoint_fn(measurement)

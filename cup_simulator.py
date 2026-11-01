"""
CUP (Compressed Ultrafast Photography) Forward Model Simulator.

Forward operator: E(x,y) = integrate_t [S_t(M(x,y) * I(x,y,t))]
  M: spatial encoding mask (random binary)
  S_t: shearing operator (horizontal shift proportional to t)
  integrate_t: temporal integration (sum over frames)

The compressed measurement E is a 2D image that encodes the 3D (x,y,t) data cube.
"""

import torch
import torch.nn.functional as F
import numpy as np


class CUPForwardModel:
    """CUP forward model: maps 3D data cube I(x,y,t) to 2D measurement E(x,y)."""

    def __init__(self, H=64, W=64, T=8, shear_per_frame=2, mask_type='random', device='cpu', real_cup=True, mask=None):
        """
        Args:
            H, W: spatial dimensions
            T: number of temporal frames
            shear_per_frame: pixels to shear per frame (forward model sensitivity)
            mask_type: 'random' or 'hadamard' (ignored if mask is provided)
            real_cup: if True, use the physically-accurate CUP forward where each
                frame is sheared right by t*shear pixels and summed onto a wider
                canvas (measurement width = W + (T-1)*shear). If False, use the
                simplified circular-shift (torch.roll) model where the canvas
                stays W and pixels wrap around.
            mask: optional pre-computed encoding mask of shape (H, W). If given,
                it overrides mask_type (enables grayscale experimental-style masks).
        """
        self.H = H
        self.W = W
        self.T = T
        self.shear = shear_per_frame
        self.device = device
        self.real_cup = real_cup

        # Create encoding mask
        if mask is not None:
            self.mask = mask.to(device).float()
        elif mask_type == 'random':
            self.mask = torch.bernoulli(torch.full((H, W), 0.5))
        elif mask_type == 'hadamard':
            self.mask = self._hadamard_mask(H, W)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
        self.mask = self.mask.to(device)

    def set_mask(self, mask):
        """Replace the encoding mask (e.g. with an experimental-style grayscale mask).

        Args:
            mask: tensor of shape (H, W) with values in [0, 1].
        """
        assert mask.shape == (self.H, self.W), \
            f"mask shape {tuple(mask.shape)} != required ({self.H}, {self.W})"
        self.mask = mask.to(self.device).float()

    def _hadamard_mask(self, H, W):
        """Create a pseudo-Hadamard pattern mask."""
        mask = np.ones((H, W))
        for i in range(H):
            for j in range(W):
                if (i % 4 < 2) ^ (j % 4 < 2):
                    mask[i, j] = 0
        return torch.from_numpy(mask).float()

    def forward(self, I):
        """
        Forward operator: I(x,y,t) -> E(x,y)

        Args:
            I: 4D tensor (B, T, H, W) ground truth data cube

        Returns:
            E: 3D tensor (B, H, Wm) compressed measurement, where
               Wm = W + (T-1)*shear  (physically-accurate, real_cup=True)
               or  Wm = W            (simplified circular model, real_cup=False)
        """
        B, T, H, W = I.shape
        assert H == self.H and W == self.W and T == self.T, \
            f"Expected ({self.T},{self.H},{self.W}), got ({T},{H},{W})"

        mask = self.mask.to(I.device)

        if self.real_cup:
            # ---- Physically-accurate CUP (streak-camera shear) ----
            # Frame t is shifted right by t*shear pixels and summed onto a
            # wider canvas. No wraparound -> measurement width grows.
            Wm = W + (T - 1) * self.shear
            E = torch.zeros(B, H, Wm, device=I.device)
            for t in range(T):
                encoded = I[:, t] * mask                 # (B, H, W)
                shift = t * self.shear
                E[:, :, shift:shift + W] += encoded      # place at sheared position
            return E
        else:
            # ---- Simplified circular-shift model (legacy) ----
            E = torch.zeros(B, H, W, device=I.device)
            for t in range(T):
                encoded = I[:, t] * mask
                shift_pixels = t * self.shear
                sheared = torch.roll(encoded, shifts=shift_pixels, dims=-1)
                E += sheared
            return E

    def adjoint(self, E):
        """
        Adjoint (transpose) operator: E(x,y) -> I_est(x,y,t)
        Used for initialization and gradient-based updates.

        Args:
            E: 3D tensor (B, H, Wm) compressed measurement

        Returns:
            I_est: 4D tensor (B, T, H, W) back-projected estimate
        """
        B, H, Wm = E.shape
        I_est = torch.zeros(B, self.T, H, self.W, device=E.device)
        mask = self.mask.to(E.device)

        if self.real_cup:
            # Inverse of the (non-circular) shear-and-place forward:
            # take the [shift : shift+W] slice of E for frame t, then *mask.
            for t in range(self.T):
                shift = t * self.shear
                I_est[:, t] = E[:, :, shift:shift + self.W] * mask
        else:
            for t in range(self.T):
                shift_pixels = t * self.shear
                unshifted = torch.roll(E, shifts=-shift_pixels, dims=-1)
                I_est[:, t] = unshifted * mask

        return I_est

    def ATA_plus_rhoI(self, I, rho):
        """
        Compute (A^T A + rho*I)(I) using the operator form.
        Necessary for the data-consistency step in the unfolding network.

        Args:
            I: 4D tensor (B, T, H, W)
            rho: scalar penalty parameter

        Returns:
            result: 4D tensor (B, T, H, W)
        """
        # A^T(A(I))
        E = self.forward(I)
        ATA_I = self.adjoint(E)
        return ATA_I + rho * I

    def gradient_data_fidelity(self, I, E):
        """
        Gradient of (1/2)||A(I) - E||_2^2 w.r.t. I
        = A^T(A(I) - E)

        Args:
            I: 4D tensor (B, T, H, W) current estimate
            E: 3D tensor (B, H, W) measurement

        Returns:
            grad: 4D tensor (B, T, H, W) gradient
        """
        residual = self.forward(I) - E
        return self.adjoint(residual)

    def to(self, device):
        self.device = device
        self.mask = self.mask.to(device)
        return self


def shift2d(x, dx, dy):
    """Shift a 2D tensor by (dx, dy) pixels using circular shift."""
    x = torch.roll(x, shifts=dx, dims=-1)  # horizontal
    x = torch.roll(x, shifts=dy, dims=-2)  # vertical
    return x

"""Simplified Mamba-style SSM layer for comparison with STU.

This implements a selective state space model (S4/Mamba-style) with the same
interface as STULayer but using learned A, B, C matrices instead of Hankel
spectral filters. This serves as a "vanilla SSM" baseline to test whether
STU's Hankel structure provides benefits beyond generic recurrence.

Key differences from STU:
- Uses parameterized state matrices (A, B, C) instead of Hankel eigenvectors
- Recurrence is computed via parallel scan (or equivalently, FFT convolution)
- No spectral filtering / eigendecomposition structure
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hippo_initializer(state_dim: int) -> torch.Tensor:
    """HiPPO-LegS initialization for A matrix (diagonal approximation).

    This is the standard S4/Mamba initialization that approximates the
    continuous-time HiPPO operator with a diagonal state matrix.
    """
    A = -torch.arange(1, state_dim + 1, dtype=torch.float32)
    return A


class MambaLayer(nn.Module):
    """Simplified Mamba/S4-style SSM layer.

    Processes [B, L, D_in] -> [B, L, D_out] using a selective state space model.
    Uses diagonal state matrix with HiPPO initialization and FFT-based convolution.

    This is intentionally simplified to match the STU layer's interface and
    parameter count for fair comparison. The key architectural difference is:
    - STU: Hankel eigenvector filters (structured, from control theory)
    - Mamba: Learned diagonal SSM (generic recurrence, HiPPO init)
    """

    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        input_dim: int,
        output_dim: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.state_dim = num_filters  # Use same K as STU for fair comparison
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Diagonal state matrix A (log-parameterized for stability)
        # HiPPO initialization: A_n = -(n + 1)
        A_init = _hippo_initializer(self.state_dim)
        self.log_A = nn.Parameter(torch.log(-A_init).to(dtype))  # [N]

        # Input projection B: [N, D_in]
        scale = (self.state_dim * input_dim) ** -0.5
        self.B = nn.Parameter(torch.randn(self.state_dim, input_dim, dtype=dtype) * scale)

        # Output projection C: [N, D_out]
        self.C = nn.Parameter(torch.randn(self.state_dim, output_dim, dtype=dtype) * scale)

        # Skip connection (D matrix in SSM notation): [D_in, D_out]
        self.D = nn.Parameter(torch.randn(input_dim, output_dim, dtype=dtype) * (input_dim ** -0.5))

        # Discretization step size (learnable, per-state)
        self.log_dt = nn.Parameter(torch.zeros(self.state_dim, dtype=dtype))

    def _compute_kernel(self, L: int) -> torch.Tensor:
        """Compute the SSM convolution kernel of length L.

        Uses ZOH (zero-order hold) discretization:
            A_bar = exp(A * dt)
            B_bar = (A_bar - I) * A^{-1} * B

        Returns kernel: [L, D_in, D_out]
        """
        dt = F.softplus(self.log_dt)  # [N]
        A = -torch.exp(self.log_A)  # [N], negative real
        A_bar = torch.exp(A * dt)  # [N], discrete A

        # B_bar = (A_bar - I) / A * B = (exp(A*dt) - 1) / A * B
        # For numerical stability when A is close to 0:
        B_bar_scale = (A_bar - 1.0) / (A + 1e-8)  # [N]

        # Compute kernel: k_l = C^T * A_bar^l * B_bar for l = 0, ..., L-1
        # Powers of A_bar: [L, N]
        powers = A_bar.unsqueeze(0) ** torch.arange(L, device=A.device, dtype=A.dtype).unsqueeze(1)  # [L, N]

        # Scale by B_bar_scale
        powers = powers * B_bar_scale.unsqueeze(0)  # [L, N]

        # kernel[l] = sum_n C[n, :] * powers[l, n] * B[n, :]^T
        # = einsum('n o, l n, n i -> l i o')
        kernel = torch.einsum('no, ln, ni -> lio', self.C, powers, self.B)  # [L, D_in, D_out]

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D_in] -> [B, L, D_out]"""
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, L, D_in = x.shape
        x = x.to(self.B.dtype)

        # Compute convolution kernel
        kernel = self._compute_kernel(L)  # [L, D_in, D_out]

        # FFT-based convolution: for each (d_in, d_out) pair
        # y[:, :, d_out] = sum_{d_in} conv(x[:, :, d_in], kernel[:, d_in, d_out])
        n_fft = 1 << math.ceil(math.log2(2 * L - 1))

        # x: [B, L, D_in] -> FFT along L
        x_f = torch.fft.rfft(x, n=n_fft, dim=1)  # [B, n_fft//2+1, D_in]
        k_f = torch.fft.rfft(kernel, n=n_fft, dim=0)  # [n_fft//2+1, D_in, D_out]

        # Convolve via multiplication in frequency domain
        # y_f[b, f, o] = sum_i x_f[b, f, i] * k_f[f, i, o]
        y_f = torch.einsum('bfi, fio -> bfo', x_f, k_f)

        # Inverse FFT
        y = torch.fft.irfft(y_f, n=n_fft, dim=1)[:, :L, :]  # [B, L, D_out]

        # Add skip connection
        y = y + x @ self.D  # [B, L, D_out]

        return y

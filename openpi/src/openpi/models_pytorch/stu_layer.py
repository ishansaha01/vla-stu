"""Spectral Transfer Unit (STU) layer for PyTorch.

Adapted from STUZero (EVAR Lab, Tsinghua University) for integration
into the openpi VLA action prediction heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    if not round_up:
        return 1 << math.floor(math.log2(x))
    else:
        return 1 << math.ceil(math.log2(x))


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k**0.25
    return phi_k.to(dtype=dtype)


def convolve(
    u: torch.Tensor,
    v: torch.Tensor,
    n: int,
    use_approx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    _, K = v.shape
    sgn = sgn.unsqueeze(-1)
    v = v.view(1, -1, K, 1, 1).to(torch.float32).contiguous()

    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)
    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn
    return U_plus, U_minus


class STULayer(nn.Module):
    """Spectral Transfer Unit layer for sequence-to-sequence processing.

    Processes [B, L, D_in] -> [B, L, D_out] using spectral filtering
    via Hankel matrix eigendecomposition.
    """

    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        input_dim: int,
        output_dim: int,
        use_hankel_L: bool = False,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        # K cannot exceed sequence length
        num_filters = min(num_filters, seq_len)
        self.num_filters = num_filters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_hankel_L = use_hankel_L
        self.zero_init = zero_init
        self.init_scale = init_scale

        # Spectral filters: [L, K]
        phi = get_spectral_filters(seq_len, num_filters, use_hankel_L, dtype=dtype)
        self.register_buffer("phi", phi, persistent=False)

        # FFT length
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)

        # Learnable projections. Three init regimes:
        #   zero_init=True            -> exact zeros (model = baseline at step 0)
        #   init_scale=1.0 (default)  -> standard randn * (K*D_in)^{-1/2}
        #   init_scale=alpha < 1.0    -> randn * alpha * (K*D_in)^{-1/2}
        #                                ("small init"; small but non-zero
        #                                 starting contribution)
        if zero_init:
            self.M_phi_plus = nn.Parameter(
                torch.zeros(num_filters, input_dim, output_dim, dtype=dtype)
            )
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.zeros(num_filters, input_dim, output_dim, dtype=dtype)
                )
        else:
            scale = init_scale * (num_filters * input_dim) ** -0.5
            self.M_phi_plus = nn.Parameter(
                torch.randn(num_filters, input_dim, output_dim, dtype=dtype) * scale
            )
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.randn(num_filters, input_dim, output_dim, dtype=dtype) * scale
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D_in] -> [B, L, D_out]"""
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.to(self.M_phi_plus.dtype)
        U_plus, U_minus = convolve(x, self.phi, self.n, use_approx=False)

        spectral_plus = torch.einsum("blki,kio->blo", U_plus, self.M_phi_plus)

        if self.use_hankel_L:
            return spectral_plus

        spectral_minus = torch.einsum("blki,kio->blo", U_minus, self.M_phi_minus)
        return spectral_plus + spectral_minus

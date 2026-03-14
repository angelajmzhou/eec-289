"""
SMT.py
======
Sparse Manifold Transform (SMT) building blocks:
  - FISTA sparse coding   → ``compute_smt_embeddings``
  - Hessian-informed dictionary update → ``quadratic_basis_update``

Based on: "Scaling Non-Parametric Sampling with Representation"
          (Lu et al., arXiv 2510.22196) and the accompanying
          SMT (2).ipynb reference implementation.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# FISTA sparse-coding (positive codes, all-PyTorch, device-agnostic)
# ---------------------------------------------------------------------------

def compute_smt_embeddings(
    I: torch.Tensor,
    basis: torch.Tensor,
    lambd: float = 0.1,
    num_iter: int = 50,
    eta: float | None = None,
) -> torch.Tensor:
    """
    FISTA sparse-coding: ``argmin_a  0.5||basis·a - I||² + λ||a||₁``.

    Device-agnostic replacement for the original CUDA-only ``FISTA``
    in SMT (2).ipynb.  Operates entirely on whatever device ``I`` and
    ``basis`` live on.

    Args:
        I:        (features, batch_size)  — input observations.
        basis:    (features, M)           — dictionary columns.
        lambd:    L1 regularisation weight.
        num_iter: FISTA iterations.
        eta:      Step size. If ``None``, computed as ``1 / λ_max(BᵀB)``.

    Returns:
        ahat: (M, batch_size) sparse codes (non-negative).
    """
    M         = basis.shape[1]
    batch_size = I.shape[1]
    dev        = I.device
    dtype      = I.dtype

    if eta is None:
        # Lipschitz constant of gradient of 0.5||Bx - y||²  is σ_max(BᵀB)
        BtB = basis.t().mm(basis)
        L   = torch.linalg.eigvalsh(BtB).max().item()
        eta = float(1.0 / (L + 1e-8))

    tk_n = 1.0
    tk   = 1.0

    ahat   = torch.zeros(M, batch_size, device=dev, dtype=dtype)
    ahat_y = torch.zeros(M, batch_size, device=dev, dtype=dtype)

    for _ in range(num_iter):
        tk     = tk_n
        tk_n   = (1 + np.sqrt(1 + 4 * tk ** 2)) / 2.0
        ahat_pre = ahat.clone()

        Res    = I - basis.mm(ahat_y)
        ahat_y = ahat_y + eta * basis.t().mm(Res)
        ahat   = (ahat_y - eta * lambd).clamp(min=0.0)
        momentum = (tk - 1) / tk_n
        ahat_y = ahat + momentum * (ahat - ahat_pre)

    return ahat  # (M, batch_size)


# ---------------------------------------------------------------------------
# Hessian-informed (quadratic) dictionary update
# ---------------------------------------------------------------------------

def quadratic_basis_update(
    basis: torch.Tensor,
    Res: torch.Tensor,
    ahat: torch.Tensor,
    lowest_activation: float = 1e-6,
    step_size: float = 0.002,
    constraint: str = "L2",
    nonneg: bool = False,
) -> torch.Tensor:
    """
    Hessian-informed dictionary update via diagonal approximation.

    The diagonal of the Hessian is estimated as the per-atom mean activation
    energy ``diag_H ≈ ahat.pow(2).mean(dim=1)``, which is the curvature along
    each basis column.  ``lowest_activation`` regularises its inverse.

    Args:
        basis:             (features, M) — current dictionary.
        Res:               (features, batch) — reconstruction residuals.
        ahat:              (M, batch) — sparse codes.
        lowest_activation: lower-clamp on the Hessian diagonal (stability).
        step_size:         gradient step multiplier.
        constraint:        ``"L2"`` normalises columns to unit norm after update.
        nonneg:            If True, clamp basis to non-negative values.

    Returns:
        Updated basis (features, M).
    """
    # Diagonal of Hessian per basis column: (M,) → (1, M) for broadcasting
    hessian_diag = ahat.pow(2).mean(dim=1).unsqueeze(0)  # (1, M)

    # Gradient: (features, M)
    d_basis = step_size * Res.mm(ahat.t()) / ahat.shape[1]
    # Scale by inverse diagonal Hessian
    d_basis = d_basis / (hessian_diag + lowest_activation)

    basis = basis + d_basis

    if nonneg:
        basis = basis.clamp(min=0.0)
    if constraint == "L2":
        basis = basis / (basis.norm(2, dim=0, keepdim=True) + 1e-8)

    return basis

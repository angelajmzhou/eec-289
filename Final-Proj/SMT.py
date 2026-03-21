"""FISTA sparse coding for SMT embeddings"""

from __future__ import annotations

import torch


def _soft_threshold(x: torch.Tensor, thr: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(x.abs() - thr, min=0.0)


@torch.no_grad()
def compute_smt_embeddings(
    I: torch.Tensor,
    basis: torch.Tensor,
    lambd: float = 0.1,
    num_iter: int = 50,
    eta: float | None = None,
    BtB: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 sparse codes: I is (F, batch), basis is (F, d), returns (d, batch)."""
    if I.ndim != 2 or basis.ndim != 2:
        raise ValueError(f"I and basis must be 2D; got {I.shape=} {basis.shape=}")
    F, B = I.shape
    Fb, d = basis.shape
    if Fb != F:
        raise ValueError(f"Feature dim mismatch: I has {F}, basis has {Fb}")

    Phi = basis
    if BtB is None:
        BtB = Phi.t().mm(Phi)
    BtI = Phi.t().mm(I)

    if eta is None:
        L = torch.linalg.eigvalsh(BtB).max().item()
        eta = float(1.0 / (L + 1e-8))
    else:
        eta = float(eta)

    thr = float(lambd) * eta
    A = torch.zeros((d, B), device=I.device, dtype=I.dtype)
    Y = A.clone()
    t = 1.0

    for _ in range(int(num_iter)):
        A_prev = A
        grad = BtB.mm(Y) - BtI
        A = _soft_threshold(Y - eta * grad, thr)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Y = A + ((t - 1.0) / t_next) * (A - A_prev)
        t = t_next

    return A

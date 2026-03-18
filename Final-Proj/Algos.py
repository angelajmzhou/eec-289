"""
Algos.py
========
Synthesis algorithms for non-parametric audio texture / speech generation
following the three-pillar strategy from Lu et al. (arXiv 2510.22196).

Public API (re-exported via ``from Algos import *``)
-----------------------------------------------------
set_seed                   – reproducible random state
choose_seed_patch_index    – pick an initial patch for the synthesis canvas
choose_frontier_frame      – pick the next frontier frame to fill
masked_ssd                 – sum-of-squared differences over known positions only
sample_from_candidates     – weighted / uniform sample from candidate pool
synthesize_audio_mfcc      – three-pillar non-parametric MFCC synthesis
synthesize_audio_ablation  – same, with per-pillar ablation flags
reconstruct_audio          – patch stitching or Griffin-Lim inversion
neural_vocoder_reconstruct – WaveRNN neural vocoder reconstruction
compute_smt_embeddings     – re-exported from SMT (FISTA sparse coding)
quadratic_basis_update     – re-exported from SMT (Hessian-informed dict update)
"""

from __future__ import annotations

import random
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from tqdm import tqdm

from SMT import compute_smt_embeddings, quadratic_basis_update  # noqa: F401 – re-export

# ---------------------------------------------------------------------------
# Global device (used by callers that build tensors outside synthesize_*)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_seed_patch_index(
    patches: torch.Tensor,
    seed_fg_min: int = 0,
    seed_fg_max: int | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> int:
    """
    Return a patch index to use as the synthesis seed.

    Picks the highest-variance patch among all admissible candidates.
    Two independent filter criteria can be combined:

    * **Index bounds** (``seed_fg_min`` / ``seed_fg_max``): restrict the search
      to a contiguous slice of the patch array — used by the utterance-based
      seeding logic in :func:`speech_synth.synthesize_speech`.
    * **Energy bounds** (``energy_min`` / ``energy_max``): restrict to patches
      whose L1 energy (``patches.abs().sum()``) falls in the given range —
      used when the caller passes a percentile energy window so the seed is
      neither silence nor an impulse.

    Both criteria are applied as an AND.  If the intersection is empty the
    function falls back gracefully (progressively relaxes to index-only, then
    to the full pool).

    Args:
        patches:     (N, C, F, W) source patch pool.
        seed_fg_min: lower index bound (inclusive).
        seed_fg_max: upper index bound (exclusive).  Defaults to N.
        energy_min:  lower L1 energy bound (inclusive).  ``None`` = no lower cap.
        energy_max:  upper L1 energy bound (inclusive).  ``None`` = no upper cap.
    """
    N   = patches.shape[0]
    lo  = min(int(seed_fg_min), N - 1)
    hi  = N if seed_fg_max is None else min(int(seed_fg_max), N)
    hi  = max(lo + 1, hi)

    # Index-range mask
    idx_mask = torch.zeros(N, dtype=torch.bool, device=patches.device)
    idx_mask[lo:hi] = True

    if energy_min is not None or energy_max is not None:
        # L1 energy per patch — matches the notebook's abs().sum() metric
        patch_energy = patches.abs().sum(dim=(-1, -2, -3))  # (N,)
        e_mask = torch.ones(N, dtype=torch.bool, device=patches.device)
        if energy_min is not None:
            e_mask &= patch_energy >= float(energy_min)
        if energy_max is not None:
            e_mask &= patch_energy <= float(energy_max)
        valid = idx_mask & e_mask
        # Graceful fallback: relax energy filter, then relax index filter
        if not valid.any():
            valid = idx_mask
        if not valid.any():
            valid = e_mask
        if not valid.any():
            valid = torch.ones(N, dtype=torch.bool, device=patches.device)
    else:
        valid = idx_mask

    valid_idx  = valid.nonzero().flatten()             # (M,)
    candidates = patches[valid_idx]                    # (M, C, F, W)
    var        = candidates.var(dim=(-1, -2, -3))      # (M,) — variance per patch
    best_local = int(var.argmax())
    return int(valid_idx[best_local])


def choose_frontier_frame(
    frontier: torch.Tensor,
    known: torch.Tensor,
    W_t: int,
    frontier_kernel: torch.Tensor | None = None,
) -> int:
    """
    Choose the next frontier frame to synthesize.

    Selects the frontier frame that already has the most known neighbours
    within a half-patch-width window, maximising the useful context available
    for Pillar-1 SSD matching.

    The selection is fully vectorised: a single 1-D convolution counts the
    known neighbours in the (2*pad+1)-wide window around every position in
    one pass, replacing the previous O(|frontier|) Python for-loop.

    Args:
        frontier:        1-D bool tensor — frames adjacent to known needing filling.
        known:           1-D bool tensor — already-filled frames.
        W_t:             patch width (frames).
        frontier_kernel: optional pre-allocated all-ones conv kernel
                         shape (1, 1, 2*(W_t//2)+1).  When provided the kernel
                         allocation (500+ calls × torch.ones) is eliminated.
    Returns:
        Integer frame index to fill next.
    """
    frontier_indices = torch.nonzero(frontier).flatten()
    if frontier_indices.numel() == 1:
        return frontier_indices[0].item()

    pad = W_t // 2
    if frontier_kernel is None:
        frontier_kernel = torch.ones(1, 1, 2 * pad + 1,
                                     device=known.device, dtype=torch.float32)
    nb_count = F.conv1d(known.float().view(1, 1, -1), frontier_kernel, padding=pad).view(-1)

    # Mask non-frontier positions to -1 so argmax always lands on the frontier.
    frontier_nb = torch.where(frontier, nb_count, nb_count.new_full((), -1.0))
    return int(frontier_nb.argmax())


def masked_ssd(
    tgt_flat: torch.Tensor,
    patches_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    gauss_weights: torch.Tensor | None = None,
    patches_sq: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-patch Gaussian-weighted SSD restricted to *known* positions.

    Implements d_SSD(ω(p), ω') = Σ_x G(x) [ω(p;x) - ω'(x)]² from the paper
    (§2.4), where G(x) is a 1-D Gaussian over the temporal (W_t) axis,
    broadcast across all MFCC feature coefficients.  When *gauss_weights* is
    None, falls back to uniform weighting (original behaviour).

    When *patches_sq* (= ``patches_flat.pow(2)``, precomputed once outside the
    synthesis loop) is supplied together with *gauss_weights*, the function uses
    the squared-distance identity

        ‖p_i − t‖²_W = (p_i²)ᵀw − 2·p_iᵀ(w⊙t) + tᵀ(w⊙t)

    which requires only two GEMVs and avoids allocating the (N, feat)
    difference matrix on every call.

    Args:
        tgt_flat:      (feat,)   — flattened target patch (unknown frames zeroed).
        patches_flat:  (N, feat) — all source patches, flattened.
        mask_flat:     (feat,)   — bool, True where the target frame is known.
        gauss_weights: (feat,)   — optional Gaussian weight per feature position.
        patches_sq:    (N, feat) — optional precomputed ``patches_flat.pow(2)``.
    Returns:
        (N,) tensor of masked SSD values, normalised by sum of active weights.
    """
    if gauss_weights is not None:
        w     = gauss_weights * mask_flat.float()   # (feat,)
        w_sum = w.sum().clamp(min=1e-8)
        if patches_sq is not None:
            # Expanded identity — no (N, feat) temp allocation.
            wt      = w * tgt_flat                  # (feat,)
            t_sq_w  = tgt_flat.dot(wt)              # scalar
            cross   = patches_flat.mv(wt)           # (N,)
            p_sq_w  = patches_sq.mv(w)              # (N,)
            return (p_sq_w - 2.0 * cross + t_sq_w) / w_sum
        # Fallback: allocate diff explicitly (backward-compatible path).
        diff = patches_flat - tgt_flat.unsqueeze(0)         # (N, feat)
        return (diff * diff * w.unsqueeze(0)).sum(dim=1) / w_sum

    # Uniform weighting — also benefits from patches_sq when available.
    n_known = mask_flat.float().sum().clamp(min=1.0)
    if patches_sq is not None:
        w     = mask_flat.float()                   # (feat,) — 0/1
        wt    = w * tgt_flat
        t_sq_w = tgt_flat.dot(wt)
        cross  = patches_flat.mv(wt)
        p_sq_w = patches_sq.mv(w)
        return (p_sq_w - 2.0 * cross + t_sq_w) / n_known
    mask_f = mask_flat.float().unsqueeze(0)                 # (1, feat)
    diff   = patches_flat - tgt_flat.unsqueeze(0)           # (N, feat)
    return (diff * diff * mask_f).sum(dim=1) / n_known

# ---------------------------------------------------------------------------
# Updated Sampling Logic
# ---------------------------------------------------------------------------

def sample_from_candidates(
    cand_idx: torch.Tensor,
    dist_ssd: torch.Tensor,
    dmin_ssd: torch.Tensor,
    weighted: bool = False,
    h_mult: float = 0.3,
    patch_var: torch.Tensor | None = None,
    var_mean: torch.Tensor | None = None,
    var_std: torch.Tensor | None = None,
    var_osc_enabled: bool = False,
    var_osc_strength: float = 0.5,
    phi: float | None = None,
    recent_mean_var: float | None = None,
) -> int:
    """
    Sample one patch index from the candidate pool.

    ``weighted=False`` (default, matches paper §2.2): places equal probability
    mass on every admissible candidate — the empirical conditional distribution
    f_p(x|ω(p)) = 1/Z Σ_{ω'∈Ω} 1{x = c(ω')}.
    ``weighted=True``: soft-min temperature sampling biased toward low-SSD
    candidates (may improve perceptual smoothness for audio at the cost of
    paper fidelity).
    """
    if cand_idx.numel() == 0:
        return -1
    if cand_idx.numel() == 1:
        return cand_idx[0].item()

    def _apply_var_osc_weights(
        base_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Only apply when explicitly enabled and all required statistics are present.
        if (
            not var_osc_enabled
            or patch_var is None
            or var_mean is None
            or var_std is None
            or phi is None
        ):
            return base_probs

        if var_std.item() <= 0:
            return base_probs

        v_cand = patch_var[cand_idx]
        z_cand = (v_cand - var_mean) / (var_std + 1e-6)

        if recent_mean_var is None:
            delta = 0.0
        else:
            delta = float((recent_mean_var - var_mean.item()) / (var_std.item() + 1e-6))

        amp = 1.0 / (1.0 + abs(delta))

        # Use a bounded exponent for numerical stability.
        sin_phi = math.sin(phi)
        exponent = var_osc_strength * amp * sin_phi
        exponent = max(min(exponent, 5.0), -5.0)

        pref = torch.exp(exponent * z_cand)
        base_probs = base_probs * pref
        s = base_probs.sum()
        if s <= 0:
            # Fallback to original probabilities if everything underflows.
            return base_probs
        return base_probs / (s + 1e-8)

    if weighted:
        cand_dists = dist_ssd[cand_idx]
        # Temperature is adaptive but floored at the std-dev of candidate distances
        # to prevent collapse when dmin_ssd ≈ 0 (sampler becoming purely greedy).
        dstd = cand_dists.std().clamp(min=1e-4)
        temperature = h_mult * dmin_ssd.clamp(min=dstd)
        # Probabilities proportional to exp(-dist / temp)
        probs = torch.exp(-(cand_dists - dmin_ssd) / temperature.clamp(min=1e-6))
        probs = probs / (probs.sum() + 1e-8)

        probs = _apply_var_osc_weights(probs)
        
        chosen_local = torch.multinomial(probs, 1).item()
        return cand_idx[chosen_local].item()
    else:
        # Start from implicit uniform probabilities and apply variance-oscillation
        # weighting if enabled.
        probs = torch.ones(cand_idx.numel(), device=dist_ssd.device)
        probs = _apply_var_osc_weights(probs)
        chosen_local = torch.multinomial(probs, 1).item()
        return cand_idx[chosen_local].item()


# ---------------------------------------------------------------------------
# Core Synthesis (Updated for Dual Epsilon)
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_audio_mfcc(
    patches: torch.Tensor,
    smt_basis: torch.Tensor,
    out_time_steps: int,
    R_loc: float = 0.2,
    eps_ssd: float = 0.1,
    eps_smt: float = 0.1,
    seed_fg_min: int = 0,
    seed_fg_max: int | None = None,
    seed_energy_min: float | None = None,
    seed_energy_max: float | None = None,
    weighted: bool = False,
    h_mult: float = 0.3,
    use_var_osc: bool = False,
    var_osc_window: int = 100,
    var_osc_strength: float = 0.5,
    var_osc_period: float = 400.0,
):
    """
    Stochastic Three-Pillar Synthesis with independent SSD and SMT thresholds.
    """
    N, C, F_bins, W_t = patches.shape
    pad_left  = W_t // 2
    pad_right = W_t - pad_left - 1
    total_len = out_time_steps + pad_left + pad_right

    patches_flat  = patches.reshape(N, -1)
    center_frames = patches[:, :, :, pad_left]
    patches_sq    = patches_flat.pow(2)             # (N, feat) — precomputed once

    # Global per-patch variance statistics for the optional variance-oscillation
    # fourth pillar.
    patch_var = patches.var(dim=(-1, -2, -3))        # (N,)
    var_mean  = patch_var.mean()
    var_std   = patch_var.std().clamp(min=1e-6)

    # Precompute FISTA step size once; smt_basis is fixed for the entire run.
    _BtB     = smt_basis.t().mm(smt_basis)          # (d, d)
    _L_smt   = torch.linalg.eigvalsh(_BtB).max().item()
    _smt_eta = float(1.0 / (_L_smt + 1e-8))

    # Source embeddings (batch FISTA, computed once at full precision).
    I_source = patches_flat.t()
    source_embeddings = compute_smt_embeddings(
        I_source, smt_basis, lambd=0.1, num_iter=50, eta=_smt_eta, BtB=_BtB
    )  # (d, N)
    # Precompute per-source-patch squared-norm for the SSL expanded-identity trick.
    source_emb_sq = source_embeddings.pow(2).sum(dim=0)  # (N,)

    source_time_positions = torch.linspace(0, 1, steps=N, device=patches.device)

    # Canvas
    out        = torch.zeros((C, F_bins, total_len), device=patches.device)
    known      = torch.zeros(total_len, dtype=torch.bool, device=patches.device)
    valid_mask = torch.zeros(total_len, dtype=torch.bool, device=patches.device)
    valid_mask[pad_left : pad_left + out_time_steps] = True
    chosen_patch_history = torch.full((total_len, 2), -1, dtype=torch.long,
                                      device=patches.device)

    # Seed placement
    seed_idx = choose_seed_patch_index(
        patches, seed_fg_min, seed_fg_max, seed_energy_min, seed_energy_max
    )
    cx = pad_left + out_time_steps // 2
    out[:, :, cx - pad_left : cx + pad_right + 1] = patches[seed_idx]
    for i in range(W_t):
        chosen_patch_history[cx - pad_left + i, 0] = seed_idx
        chosen_patch_history[cx - pad_left + i, 1] = i
    known[cx] = True

    # Gaussian weights for d_SSD (paper §2.4)
    t_gauss       = torch.arange(W_t, device=patches.device).float() - pad_left
    sigma_g       = max(W_t / 4.0, 0.5)
    gauss_1d      = torch.exp(-0.5 * (t_gauss / sigma_g) ** 2)
    gauss_1d      = gauss_1d / gauss_1d.sum()
    gauss_weights = gauss_1d.view(1, 1, -1).expand(C, F_bins, -1).reshape(-1)  # (feat,)

    # Hoist kernel allocations out of the loop
    kernel_adj      = torch.tensor([1, 0, 1], dtype=torch.float32,
                                    device=patches.device).view(1, 1, 3)
    frontier_kernel = torch.ones(1, 1, 2 * pad_left + 1,
                                  device=patches.device, dtype=torch.float32)

    n_known = 1  # integer counter replaces known[valid_mask].sum() each step
    pbar    = tqdm(total=out_time_steps)
    pbar.update(1)

    # State for the optional variance-oscillation fourth pillar.
    recent_var: list[float] = []
    phi = 0.0

    while n_known < out_time_steps:
        nb_adj   = F.conv1d(known.float().view(1, 1, -1), kernel_adj, padding=1).view(-1)
        frontier = (~known) & valid_mask & (nb_adj > 0)
        if not frontier.any():
            break

        x = choose_frontier_frame(frontier, known, W_t, frontier_kernel)

        # PILLAR 1: Gaussian-weighted SSD over ALL source patches (paper §2.4)
        tgt_patch = out[:, :, x - pad_left : x + pad_right + 1]
        mask_1d   = known[x - pad_left : x + pad_right + 1]
        mask_flat = mask_1d.repeat(C, F_bins, 1).reshape(-1)

        dist_ssd = masked_ssd(tgt_patch.reshape(-1), patches_flat, mask_flat,
                              gauss_weights, patches_sq)
        dmin_ssd = dist_ssd.min()
        mask_ssd = dist_ssd <= (1 + eps_ssd) * dmin_ssd

        # PILLAR 2: Locality (paper §2.4)
        target_time_pos = (x - pad_left) / max(out_time_steps - 1, 1)
        dist_loc = torch.abs(source_time_positions - target_time_pos)
        mask_loc = dist_loc <= R_loc

        # PILLAR 3: High-level SMT — use 10 iters (target is near-zero early on).
        # Expanded identity avoids (d, N) temporary allocation:
        #   ‖s_i − t‖² = s_i²_sum − 2·sᵢᵀt + ‖t‖²
        tgt_masked = tgt_patch.clone()
        tgt_masked[:, :, ~mask_1d] = 0.0
        I_tgt         = tgt_masked.reshape(-1, 1)
        tgt_embedding = compute_smt_embeddings(
            I_tgt, smt_basis, lambd=0.1, num_iter=10, eta=_smt_eta, BtB=_BtB
        )                                                    # (d, 1)
        tgt_e         = tgt_embedding.squeeze(1)             # (d,)
        cross         = source_embeddings.t().mv(tgt_e)      # (N,) — one GEMV
        tgt_sq        = tgt_e.dot(tgt_e)                     # scalar
        all_dist_ssl  = source_emb_sq - 2.0 * cross + tgt_sq  # (N,)
        dmin_ssl      = all_dist_ssl.min()
        mask_ssl      = all_dist_ssl <= (1 + eps_smt) * dmin_ssl

        # Final candidate pool: P1 ∩ P2 ∩ P3  (uniform sampling per paper §2.2)
        final_cand_idx = torch.nonzero(mask_ssd & mask_loc & mask_ssl).flatten()
        if final_cand_idx.numel() == 0:
            final_cand_idx = torch.nonzero(mask_ssd & mask_loc).flatten()
            if final_cand_idx.numel() == 0:
                final_cand_idx = torch.nonzero(mask_ssd).flatten()

        if use_var_osc:
            if recent_var:
                recent_mean_var = float(sum(recent_var) / len(recent_var))
            else:
                recent_mean_var = float(var_mean.item())
            chosen_idx = sample_from_candidates(
                final_cand_idx,
                dist_ssd,
                dmin_ssd,
                weighted,
                h_mult,
                patch_var=patch_var,
                var_mean=var_mean,
                var_std=var_std,
                var_osc_enabled=True,
                var_osc_strength=var_osc_strength,
                phi=phi,
                recent_mean_var=recent_mean_var,
            )
        else:
            chosen_idx = sample_from_candidates(
                final_cand_idx, dist_ssd, dmin_ssd, weighted, h_mult
            )

        out[:, :, x] = center_frames[chosen_idx]
        known[x]     = True
        chosen_patch_history[x, 0] = chosen_idx
        chosen_patch_history[x, 1] = pad_left

        # Update variance-oscillation state.
        if use_var_osc:
            recent_var.append(float(patch_var[chosen_idx].item()))
            if len(recent_var) > var_osc_window:
                recent_var = recent_var[-var_osc_window:]
            if var_osc_period > 0:
                phi += 2.0 * math.pi / max(var_osc_period, 1.0)

        n_known += 1
        pbar.update(1)

    pbar.close()
    return (out[:, :, pad_left : pad_left + out_time_steps],
            chosen_patch_history[pad_left : pad_left + out_time_steps])


# ---------------------------------------------------------------------------
# Ablation variant — selectively disable pillars 2 and/or 3
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_audio_ablation(
    patches: torch.Tensor,
    smt_basis: torch.Tensor,
    out_time_steps: int,
    R_loc: float = 0.2,
    eps: float = 0.1,
    seed_fg_min: int = 0,
    seed_fg_max: int | None = None,
    seed_energy_min: float | None = None,
    seed_energy_max: float | None = None,
    weighted: bool = True,
    h_mult: float = 0.3,
    use_loc: bool = True,
    use_smt: bool = True,
    use_var_osc: bool = False,
    var_osc_window: int = 100,
    var_osc_strength: float = 0.5,
    var_osc_period: float = 400.0,
):
    """
    Drop-in replacement for ``synthesize_audio_mfcc`` with per-pillar ablation.

    Setting ``use_loc=False`` replaces the locality gate (Pillar 2) with an
    all-True mask (all source positions accepted).  Setting ``use_smt=False``
    replaces the SSL gate (Pillar 3) with an all-True mask.  Pillar 1 (SSD)
    is always active.

    All other arguments and return values are identical to
    ``synthesize_audio_mfcc``.
    """
    N, C, F_bins, W_t = patches.shape
    pad_left  = W_t // 2
    pad_right = W_t - pad_left - 1
    total_len = out_time_steps + pad_left + pad_right

    patches_flat  = patches.reshape(N, -1)
    center_frames = patches[:, :, :, pad_left]
    patches_sq    = patches_flat.pow(2)             # (N, feat) — precomputed once

    # Global per-patch variance statistics for the optional variance-oscillation
    # fourth pillar.
    patch_var = patches.var(dim=(-1, -2, -3))        # (N,)
    var_mean  = patch_var.mean()
    var_std   = patch_var.std().clamp(min=1e-6)

    _BtB     = smt_basis.t().mm(smt_basis)
    _L_smt   = torch.linalg.eigvalsh(_BtB).max().item()
    _smt_eta = float(1.0 / (_L_smt + 1e-8))

    I_source          = patches_flat.t()
    source_embeddings = compute_smt_embeddings(
        I_source, smt_basis, lambd=0.1, num_iter=50, eta=_smt_eta, BtB=_BtB
    )  # (d, N)
    source_emb_sq     = source_embeddings.pow(2).sum(dim=0)  # (N,)

    source_time_positions = torch.linspace(0, 1, steps=N, device=patches.device)

    # Gaussian weights for d_SSD (paper §2.4)
    t_gauss      = torch.arange(W_t, device=patches.device).float() - pad_left
    sigma_g      = max(W_t / 4.0, 0.5)
    gauss_1d     = torch.exp(-0.5 * (t_gauss / sigma_g) ** 2)
    gauss_1d     = gauss_1d / gauss_1d.sum()
    gauss_weights = gauss_1d.view(1, 1, -1).expand(C, F_bins, -1).reshape(-1)

    out        = torch.zeros((C, F_bins, total_len), device=patches.device)
    known      = torch.zeros(total_len, dtype=torch.bool, device=patches.device)
    valid_mask = torch.zeros(total_len, dtype=torch.bool, device=patches.device)
    valid_mask[pad_left : pad_left + out_time_steps] = True
    chosen_patch_history = torch.full((total_len, 2), -1, dtype=torch.long,
                                      device=patches.device)

    seed_idx = choose_seed_patch_index(
        patches, seed_fg_min, seed_fg_max, seed_energy_min, seed_energy_max
    )
    cx = pad_left + out_time_steps // 2
    out[:, :, cx - pad_left : cx + pad_right + 1] = patches[seed_idx]
    for _i in range(W_t):
        chosen_patch_history[cx - pad_left + _i, 0] = seed_idx
        chosen_patch_history[cx - pad_left + _i, 1] = _i
    known[cx] = True

    # Hoist kernel allocations
    kernel_adj      = torch.tensor([1, 0, 1], dtype=torch.float32,
                                    device=patches.device).view(1, 1, 3)
    frontier_kernel = torch.ones(1, 1, 2 * pad_left + 1,
                                  device=patches.device, dtype=torch.float32)

    # Static all-True masks (for disabled pillars)
    _all_true = torch.ones(N, dtype=torch.bool, device=patches.device)

    n_known = 1

    # State for the optional variance-oscillation fourth pillar.
    recent_var: list[float] = []
    phi = 0.0

    while n_known < out_time_steps:
        nb_adj   = F.conv1d(known.float().view(1, 1, -1), kernel_adj, padding=1).view(-1)
        frontier = (~known) & valid_mask & (nb_adj > 0)
        if not frontier.any():
            break
        x = choose_frontier_frame(frontier, known, W_t, frontier_kernel)

        tgt_patch = out[:, :, x - pad_left : x + pad_right + 1]
        mask_1d   = known[x - pad_left : x + pad_right + 1]
        mask_flat = mask_1d.repeat(C, F_bins, 1).reshape(-1)

        # Pillar 1: Gaussian-weighted SSD (paper §2.4)
        dist_ssd = masked_ssd(tgt_patch.reshape(-1), patches_flat, mask_flat,
                              gauss_weights, patches_sq)
        dmin_ssd = dist_ssd.min()
        mask_ssd = dist_ssd <= (1 + eps) * dmin_ssd

        # Pillar 2: Locality (optional)
        if use_loc:
            target_time_pos = (x - pad_left) / max(out_time_steps - 1, 1)
            dist_loc = torch.abs(source_time_positions - target_time_pos)
            mask_loc = dist_loc <= R_loc
        else:
            mask_loc = _all_true

        # Pillar 3: SMT (optional) — same expanded-identity trick as main function
        if use_smt:
            tgt_masked = tgt_patch.clone()
            tgt_masked[:, :, ~mask_1d] = 0.0
            I_tgt = tgt_masked.reshape(-1, 1)
            tgt_embedding = compute_smt_embeddings(
                I_tgt, smt_basis, lambd=0.1, num_iter=10, eta=_smt_eta, BtB=_BtB
            )                                                # (d, 1)
            tgt_e    = tgt_embedding.squeeze(1)              # (d,)
            cross    = source_embeddings.t().mv(tgt_e)       # (N,)
            tgt_sq   = tgt_e.dot(tgt_e)                      # scalar
            dist_ssl = source_emb_sq - 2.0 * cross + tgt_sq  # (N,)
            dmin_ssl = dist_ssl.min()
            mask_ssl = dist_ssl <= (1 + eps) * dmin_ssl
        else:
            mask_ssl = _all_true

        combined_mask = mask_ssd & mask_loc & mask_ssl
        cand_idx      = torch.nonzero(combined_mask).flatten()
        if cand_idx.numel() == 0:
            cand_idx = torch.nonzero(mask_ssd & mask_loc).flatten()
            if cand_idx.numel() == 0:
                cand_idx = torch.nonzero(mask_ssd).flatten()

        if use_var_osc:
            if recent_var:
                recent_mean_var = float(sum(recent_var) / len(recent_var))
            else:
                recent_mean_var = float(var_mean.item())
            chosen_idx = sample_from_candidates(
                cand_idx,
                dist_ssd,
                dmin_ssd,
                weighted,
                h_mult,
                patch_var=patch_var,
                var_mean=var_mean,
                var_std=var_std,
                var_osc_enabled=True,
                var_osc_strength=var_osc_strength,
                phi=phi,
                recent_mean_var=recent_mean_var,
            )
        else:
            chosen_idx = sample_from_candidates(cand_idx, dist_ssd, dmin_ssd, weighted, h_mult)

        out[:, :, x]               = center_frames[chosen_idx]
        known[x]                   = True
        chosen_patch_history[x, 0] = chosen_idx
        chosen_patch_history[x, 1] = pad_left

        # Update variance-oscillation state.
        if use_var_osc:
            recent_var.append(float(patch_var[chosen_idx].item()))
            if len(recent_var) > var_osc_window:
                recent_var = recent_var[-var_osc_window:]
            if var_osc_period > 0:
                phi += 2.0 * math.pi / max(var_osc_period, 1.0)

        n_known += 1

    return (out[:, :, pad_left : pad_left + out_time_steps],
            chosen_patch_history[pad_left : pad_left + out_time_steps])


# ---------------------------------------------------------------------------
# Audio reconstruction
# ---------------------------------------------------------------------------

def reconstruct_audio(
    method: str,
    synth_mfcc: torch.Tensor = None,
    chosen_indices: torch.Tensor = None,
    source_audio: np.ndarray = None,
    hop_length: int = 512,
    window_t: int = None,
    sr: int = 22050,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Reconstruct a waveform from synthesis outputs.

    Two methods are supported:

    **'stitch'** — patch stitching (requires *chosen_indices* and *source_audio*).
        For each output frame the corresponding HOP_LENGTH-sample chunk is copied
        from the original source waveform and concatenated.  Fast and artefact-free
        provided the source sample rate is known.

    **'griffin_lim'** — iterative spectrogram inversion (requires *synth_mfcc*).
        Inverts MFCCs via librosa's Griffin-Lim implementation.  Slower but does
        not require the original audio.

    Args:
        method:         'stitch' or 'griffin_lim'.
        synth_mfcc:     (1, n_mfcc, T_out) — for griffin_lim path.
        chosen_indices: (T_out, 2)  — [patch_idx, frame_in_patch] for stitch path.
        source_audio:   1-D numpy waveform for stitch path.
        hop_length:     MFCC hop length in samples.
        window_t:       patch width (frames) — unused but kept for API compatibility.
        sr:             sample rate (for griffin_lim only).
        n_mels:         mel bands (for griffin_lim only).
    Returns:
        1-D float32 numpy waveform.
    """
    if method == "stitch":
        if chosen_indices is None or source_audio is None:
            raise ValueError("'stitch' requires chosen_indices and source_audio.")

        stitched_audio = []
        for item in chosen_indices:
            patch_idx = item[0].item()
            frame_within_patch = item[1].item()

            if patch_idx == -1:
                stitched_audio.append(np.zeros(hop_length))
                continue

            absolute_frame_idx = patch_idx + frame_within_patch
            start_sample = absolute_frame_idx * hop_length
            end_sample   = start_sample + hop_length
            audio_chunk  = source_audio[start_sample:end_sample]

            if len(audio_chunk) < hop_length:
                audio_chunk = np.pad(audio_chunk, (0, hop_length - len(audio_chunk)))

            stitched_audio.append(audio_chunk)

        return np.concatenate(stitched_audio)

    elif method == "griffin_lim":
        if synth_mfcc is None:
            raise ValueError("'griffin_lim' requires synth_mfcc.")

        mfcc_np = synth_mfcc.squeeze(0).cpu().numpy()
        print("Running Griffin-Lim reconstruction (this may take a moment)...")
        audio_recon = librosa.feature.inverse.mfcc_to_audio(
            mfcc_np,
            n_mels=n_mels,
            hop_length=hop_length,
            sr=sr,
        )
        return audio_recon

    else:
        raise ValueError("method must be 'stitch' or 'griffin_lim'.")


# ---------------------------------------------------------------------------
# Neural Vocoder (WaveRNN)
# ---------------------------------------------------------------------------

def neural_vocoder_reconstruct(
    audio_np: np.ndarray,
    sr: int,
    output_path: Path | None = None,
    target_sr: int = 22050,
    n_mels: int = 80,
):
    """
    Re-synthesize *audio_np* through a WaveRNN neural vocoder.

    Steps:
        1. Resample to 22050 Hz (WaveRNN model rate).
        2. Compute 80-bin log-mel spectrogram with WaveRNN's window parameters.
        3. Run ``torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH`` vocoder.
        4. Return the new waveform as a numpy array at 22050 Hz.

    The vocoder model (~50 MB) is downloaded once and cached by torchaudio.

    Args:
        audio_np:    Input waveform (float32 numpy array).
        sr:          Sample rate of *audio_np*.
        output_path: If given, save the vocoder output to this path.
        target_sr:   Ignored — WaveRNN always runs at 22050 Hz.
        n_mels:      Mel bands (must be 80 for this WaveRNN checkpoint).
    Returns:
        (audio_out, vocoder_sr) — numpy waveform and its sample rate.
    """
    import torchaudio
    from torchaudio.pipelines import TACOTRON2_WAVERNN_PHONE_LJSPEECH as BUNDLE

    print("Loading WaveRNN vocoder (downloads ~50 MB on first run)...")
    vocoder    = BUNDLE.get_vocoder().to(device)
    vocoder.eval()
    vocoder_sr = BUNDLE.sample_rate  # 22050

    wf = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)  # (1, T)
    if sr != vocoder_sr:
        wf = torchaudio.functional.resample(wf, sr, vocoder_sr)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=vocoder_sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=n_mels,
        f_min=40.0,
        f_max=vocoder_sr // 2,
        power=1.0,  # amplitude spectrogram (WaveRNN convention)
    ).to(device)

    with torch.no_grad():
        mel   = mel_transform(wf.to(device))       # (1, n_mels, T)
        mel   = torch.log(mel.clamp(min=1e-5))
        mel_t = mel.transpose(1, 2)                 # (1, T, 80)
        output_wf, _lengths = vocoder(mel_t, [mel_t.shape[1]])

    audio_out = output_wf[0].squeeze(0).cpu().numpy()

    if output_path is not None:
        sf.write(str(output_path), audio_out, vocoder_sr)
        print(f"Neural vocoder output saved to: {output_path}")

    return audio_out, vocoder_sr

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

from SMT import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0) -> None:
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
    """Pick seed patch: index range plus optional L1 energy window; prefer high variance."""
    N   = patches.shape[0]
    lo  = min(int(seed_fg_min), N - 1)
    hi  = N if seed_fg_max is None else min(int(seed_fg_max), N)
    hi  = max(lo + 1, hi)

    idx_mask = torch.zeros(N, dtype=torch.bool, device=patches.device)
    idx_mask[lo:hi] = True

    if energy_min is not None or energy_max is not None:
        patch_energy = patches.abs().sum(dim=(-1, -2, -3))  # (N,)
        e_mask = torch.ones(N, dtype=torch.bool, device=patches.device)
        if energy_min is not None:
            e_mask &= patch_energy >= float(energy_min)
        if energy_max is not None:
            e_mask &= patch_energy <= float(energy_max)
        valid = idx_mask & e_mask
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
    """Next frontier index; pass frontier_kernel to avoid reallocating each step."""
    frontier_indices = torch.nonzero(frontier).flatten()
    if frontier_indices.numel() == 1:
        return frontier_indices[0].item()

    pad = W_t // 2
    if frontier_kernel is None:
        frontier_kernel = torch.ones(1, 1, 2 * pad + 1,
                                     device=known.device, dtype=torch.float32)
    nb_count = F.conv1d(known.float().view(1, 1, -1), frontier_kernel, padding=pad).view(-1)

    frontier_nb = torch.where(frontier, nb_count, nb_count.new_full((), -1.0))
    return int(frontier_nb.argmax())


def masked_ssd(
    tgt_flat: torch.Tensor,
    patches_flat: torch.Tensor,
    mask_flat: torch.Tensor,
    gauss_weights: torch.Tensor | None = None,
    patches_sq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian-weighted SSD to all patches; optional patches_sq for the expanded norm."""
    if gauss_weights is not None:
        w     = gauss_weights * mask_flat.float()   # (feat,)
        w_sum = w.sum().clamp(min=1e-8)
        if patches_sq is not None:
            wt      = w * tgt_flat                  # (feat,)
            t_sq_w  = tgt_flat.dot(wt)              # scalar
            cross   = patches_flat.mv(wt)           # (N,)
            p_sq_w  = patches_sq.mv(w)              # (N,)
            return (p_sq_w - 2.0 * cross + t_sq_w) / w_sum
        diff = patches_flat - tgt_flat.unsqueeze(0)         # (N, feat)
        return (diff * diff * w.unsqueeze(0)).sum(dim=1) / w_sum

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
) -> int:
    """
    Sample one patch index from the candidate pool.
    Bias toward low-SSD with h_mult
    """
    if cand_idx.numel() == 0:
        return -1
    if cand_idx.numel() == 1:
        return cand_idx[0].item()

    def _apply_var_osc_weights(
        base_probs: torch.Tensor,
    ) -> torch.Tensor:
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

        sin_phi = math.sin(phi)
        exponent = var_osc_strength * sin_phi
        exponent = max(min(exponent, 5.0), -5.0)

        pref = torch.exp(exponent * z_cand)
        base_probs = base_probs * pref
        s = base_probs.sum()
        if s <= 0:
            return base_probs
        return base_probs / (s + 1e-8)

    if weighted:
        cand_dists = dist_ssd[cand_idx]
        dstd = cand_dists.std().clamp(min=1e-4)
        temperature = h_mult * dmin_ssd.clamp(min=dstd)
        probs = torch.exp(-(cand_dists - dmin_ssd) / temperature.clamp(min=1e-6))
        probs = probs / (probs.sum() + 1e-8)
        probs = _apply_var_osc_weights(probs)
        chosen_local = torch.multinomial(probs, 1).item()
        return cand_idx[chosen_local].item()
    else:
        probs = torch.ones(cand_idx.numel(), device=dist_ssd.device)
        probs = _apply_var_osc_weights(probs)
        chosen_local = torch.multinomial(probs, 1).item()
        return cand_idx[chosen_local].item()



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
    var_osc_strength: float = 0.5,
    var_osc_period: float = 400.0,
    return_debug: bool = False,
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


    patch_var = patches.var(dim=(-1, -2, -3))        # (N,)
    var_mean  = patch_var.mean()
    var_std   = patch_var.std().clamp(min=1e-6)

    _BtB     = smt_basis.t().mm(smt_basis)          # (d, d)
    _L_smt   = torch.linalg.eigvalsh(_BtB).max().item()
    _smt_eta = float(1.0 / (_L_smt + 1e-8))
    I_source = patches_flat.t()
    source_embeddings = compute_smt_embeddings(
        I_source, smt_basis, lambd=0.1, num_iter=50, eta=_smt_eta, BtB=_BtB
    )  # (d, N)
    source_emb_sq = source_embeddings.pow(2).sum(dim=0)  # (N,)

    source_time_positions = torch.linspace(0, 1, steps=N, device=patches.device)

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
    for i in range(W_t):
        chosen_patch_history[cx - pad_left + i, 0] = seed_idx
        chosen_patch_history[cx - pad_left + i, 1] = i
    known[cx] = True

    t_gauss       = torch.arange(W_t, device=patches.device).float() - pad_left
    sigma_g       = max(W_t / 4.0, 0.5)
    gauss_1d      = torch.exp(-0.5 * (t_gauss / sigma_g) ** 2)
    gauss_1d      = gauss_1d / gauss_1d.sum()
    gauss_weights = gauss_1d.view(1, 1, -1).expand(C, F_bins, -1).reshape(-1)  # (feat,)
    kernel_adj      = torch.tensor([1, 0, 1], dtype=torch.float32,
                                    device=patches.device).view(1, 1, 3)
    frontier_kernel = torch.ones(1, 1, 2 * pad_left + 1,
                                  device=patches.device, dtype=torch.float32)

    n_known = 1  
    pbar    = tqdm(total=out_time_steps)
    pbar.update(1)

    phi = 0.0
    empty_smt_steps = 0

    while n_known < out_time_steps:
        nb_adj   = F.conv1d(known.float().view(1, 1, -1), kernel_adj, padding=1).view(-1)
        frontier = (~known) & valid_mask & (nb_adj > 0)
        if not frontier.any():
            break

        x = choose_frontier_frame(frontier, known, W_t, frontier_kernel)

        tgt_patch = out[:, :, x - pad_left : x + pad_right + 1]
        mask_1d   = known[x - pad_left : x + pad_right + 1]
        mask_flat = mask_1d.repeat(C, F_bins, 1).reshape(-1)
        #SSD
        dist_ssd = masked_ssd(tgt_patch.reshape(-1), patches_flat, mask_flat,
                              gauss_weights, patches_sq)
        dmin_ssd = dist_ssd.min()
        mask_ssd = dist_ssd <= (1 + eps_ssd) * dmin_ssd

        #position
        target_time_pos = (x - pad_left) / max(out_time_steps - 1, 1)
        dist_loc = torch.abs(source_time_positions - target_time_pos)
        mask_loc = dist_loc <= R_loc

        # SMT 
        tgt_masked = tgt_patch.clone()
        tgt_masked[:, :, ~mask_1d] = 0.0
        I_tgt         = tgt_masked.reshape(-1, 1)
        tgt_embedding = compute_smt_embeddings(
            I_tgt, smt_basis, lambd=0.1, num_iter=10, eta=_smt_eta, BtB=_BtB
        )                                                 
        tgt_e         = tgt_embedding.squeeze(1)             
        cross         = source_embeddings.t().mv(tgt_e)     
        tgt_sq        = tgt_e.dot(tgt_e)                   
        all_dist_ssl  = source_emb_sq - 2.0 * cross + tgt_sq 
        dmin_ssl      = all_dist_ssl.min()
        mask_ssl      = all_dist_ssl <= (1 + eps_smt) * dmin_ssl

        cand_smt = mask_ssd & mask_loc & mask_ssl
        final_cand_idx = torch.nonzero(cand_smt).flatten()
        if final_cand_idx.numel() == 0:
            if (mask_ssd & mask_loc).any(): #fallback
                empty_smt_steps += 1
            final_cand_idx = torch.nonzero(mask_ssd & mask_loc).flatten()
            if final_cand_idx.numel() == 0:
                final_cand_idx = torch.nonzero(mask_ssd).flatten()

        if use_var_osc:
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
            )
        else:
            chosen_idx = sample_from_candidates(
                final_cand_idx, dist_ssd, dmin_ssd, weighted, h_mult
            )

        out[:, :, x] = center_frames[chosen_idx]
        known[x]     = True
        chosen_patch_history[x, 0] = chosen_idx
        chosen_patch_history[x, 1] = pad_left

        if use_var_osc and var_osc_period > 0:
            phi += 2.0 * math.pi / max(var_osc_period, 1.0)

        n_known += 1
        pbar.update(1)

    pbar.close()
    out_mfcc = out[:, :, pad_left : pad_left + out_time_steps]
    history  = chosen_patch_history[pad_left : pad_left + out_time_steps]
    if return_debug:
        return out_mfcc, history, {"empty_smt_steps": int(empty_smt_steps)}
    return out_mfcc, history



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
    """Stitch hop windows from source or Griffin–Lim from MFCC."""
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



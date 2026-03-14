
# Imports
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def compute_audio_frontier(known: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    known, valid: (H, W) bool where H=MFCC_bins, W=Time
    frontier: unknown & valid & adjacent-to-known (strictly in time)
    """
    assert known.shape == valid.shape
    H, W = known.shape
    known_f = known.float().view(1, 1, H, W)
    # look only at temporal neighbors as we always use whole frequency band
    k = torch.tensor([[[[1, 0, 1]]]], device=known.device).float() # (out_channels, in_channels, height, width)
    # Count known neighbors along the time axis
    # Padding=(0, 1) pads only the width (time) dimension
    nb = F.conv2d(known_f, k, padding=(0, 1)).view(H, W)
    # Frontier = unknown frames that are valid and have a known neighbor in time
    frontier = (~known) & valid & (nb > 0)
    return frontier
    
def choose_seed_patch_index(patches: torch.Tensor, seed_fg_min: int = 0) -> int:
    if seed_fg_min > 0:
        # Sum over all dims except the batch (N) -> C, F, T
        fg = patches.abs().sum(dim=[1, 2, 3])
        valid_indices = torch.nonzero(fg >= seed_fg_min).flatten()
        if valid_indices.numel() > 0:
            return valid_indices[random.randint(0, valid_indices.shape[0] - 1)].item()
    return random.randint(0, patches.shape[0] - 1)

def choose_frontier_frame(frontier: torch.Tensor, known: torch.Tensor, window_t: int) -> int:
    """
    Finds the time-index (x) on the frontier with the most known temporal neighbors. It there is a tie, pick randomly.
    frontier/known: (Time,) bool
    """
    if not frontier.any(): return None
    
    # 1. Prepare the 1D kernel (Size: Out_channels, In_channels, Width)
    # A window of 'ones' counts how many neighbors are 'True' (1.0)
    kernel = torch.ones((1, 1, window_t), device=known.device)
    known_f = known.float().view(1, 1, -1)

    pad = window_t // 2
    nb = F.conv1d(known_f, kernel, padding="same").view(-1) #look @ last dim (t)
    frontier_counts = torch.where(frontier, nb, -1.0) #(t), non-frontier = -1

    max_val = torch.max(frontier_counts)
    candidate_indices = torch.where(frontier_counts == max_val)[0]

    choice = torch.randint(0, len(candidate_indices), (1,)).item()
    return int(candidate_indices[choice])
    
def sample_from_candidates(
    cand_idx: torch.Tensor,   # (M,)
    dist: torch.Tensor,       # (N,)
    dmin: torch.Tensor,       # scalar
    weighted: bool = True,
    h_mult: float = 0.3,
) -> int:
    if weighted:
        cand_dists = dist[cand_idx]
        probs = torch.exp(-(cand_dists - dmin) / (h_mult * dmin + 1e-6))
        idx = torch.multinomial(probs, num_samples=1).item()
        chosen_idx = cand_idx[idx].item()
    else:
        # Uniform random selection
        random_pos = torch.randint(0, len(cand_idx), (1,)).item()
        chosen_idx = cand_idx[random_pos].item()
    return int(chosen_idx)

def masked_ssd(tgt_patch: torch.Tensor, patches_flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    tgt_patch: (1, F, T)
    patches_flat: (N, 1, F*T)
    mask: (F*T) bool
    """
    # Flatten target and compare only at masked (known) positions
    tgt_flat = tgt_patch.reshape(1, -1) # (1, F*T)
    # SSD over the known spectral-temporal bins
    diff = (patches_flat[:, mask] - tgt_flat[:, mask]) #sub target from all patches
    return diff.square().sum(dim=-1) #return squared diff

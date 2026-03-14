import numpy as np
from scipy.special import rel_entr
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Metric 1: KL divergence — MFCC distribution match
# Compare per-coefficient distributions of output center-frames vs source frames.
# Lower is better (generated distributes like the source).
# ─────────────────────────────────────────────────────────────────────────────
def kl_mfcc_divergence(
    out_mfcc: torch.Tensor,   # (1, F_bins, T_out)
    source_patches: torch.Tensor,  # (N, 1, F_bins, W_t)
    n_bins: int = 30,
) -> float:
    # Source: center frames of all source patches → (N, F_bins)
    W_t = source_patches.shape[3]
    pad_left = W_t // 2
    src_frames = source_patches[:, 0, :, pad_left].cpu().numpy()  # (N, F)

    # Output: frames along time axis → (T_out, F_bins)
    out_frames = out_mfcc[0].T.cpu().numpy()  # (T_out, F)

    kl_sum = 0.0
    n_coeff = src_frames.shape[1]
    for c in range(n_coeff):
        mn = min(src_frames[:, c].min(), out_frames[:, c].min())
        mx = max(src_frames[:, c].max(), out_frames[:, c].max()) + 1e-8
        bins = np.linspace(mn, mx, n_bins + 1)

        p, _ = np.histogram(src_frames[:, c], bins=bins, density=True)
        q, _ = np.histogram(out_frames[:, c], bins=bins, density=True)

        # Smooth to avoid log(0)
        p = p + 1e-9;  p /= p.sum()
        q = q + 1e-9;  q /= q.sum()

        kl_sum += rel_entr(p, q).sum()

    return float(kl_sum / n_coeff)   # mean KL across MFCC coefficients


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2: Patch diversity
# Fraction of unique source patches selected. 0 = same patch repeated; 1 = all unique.
# ─────────────────────────────────────────────────────────────────────────────
def patch_diversity(history_indices: torch.Tensor) -> float:
    valid = history_indices[:, 0]
    valid = valid[valid >= 0]
    if valid.numel() == 0:
        return 0.0
    return float(valid.unique().numel() / valid.numel())


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3: Temporal continuity
# Mean cosine similarity between consecutive output frames.
# 1 = identical neighbours (too smooth / looping); ~0.7-0.9 is natural speech.
# ─────────────────────────────────────────────────────────────────────────────
def temporal_continuity(out_mfcc: torch.Tensor) -> float:
    frames = out_mfcc[0].T  # (T, F_bins)
    if frames.shape[0] < 2:
        return float("nan")
    a = frames[:-1]  # (T-1, F)
    b = frames[1:]   # (T-1, F)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return float(cos.mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4: Source coverage
# Fraction of the source pool that was used at least once.
# ─────────────────────────────────────────────────────────────────────────────
def source_coverage(history_indices: torch.Tensor, n_source_patches: int) -> float:
    valid = history_indices[:, 0]
    valid = valid[valid >= 0]
    if valid.numel() == 0 or n_source_patches == 0:
        return 0.0
    return float(valid.unique().numel() / n_source_patches)


# ─────────────────────────────────────────────────────────────────────────────
# Composite score (equal weights, all normalised to "higher = better").
# kl_div is inverted via 1/(1+kl).
# ─────────────────────────────────────────────────────────────────────────────
def composite_score(kl: float, diversity: float,
                    continuity: float, coverage: float) -> float:
    kl_score = 1.0 / (1.0 + kl)
    return float(np.mean([kl_score, diversity, continuity, coverage]))


def evaluate_synthesis_run(
    out_mfcc: torch.Tensor,
    history_indices: torch.Tensor,
    source_patches: torch.Tensor,
) -> dict:
    """Return a dict of all four metrics plus the composite score."""
    n_src = source_patches.shape[0]
    kl_val  = kl_mfcc_divergence(out_mfcc, source_patches)
    div_val = patch_diversity(history_indices)
    cont_val= temporal_continuity(out_mfcc)
    cov_val = source_coverage(history_indices, n_src)
    cs      = composite_score(kl_val, div_val, cont_val, cov_val)
    return {
        "kl_div":       round(kl_val,  4),
        "diversity":    round(div_val,  4),
        "continuity":   round(cont_val, 4),
        "coverage":     round(cov_val,  4),
        "composite":    round(cs,       4),
    }




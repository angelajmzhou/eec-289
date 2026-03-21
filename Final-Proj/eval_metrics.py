from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import rel_entr
import torch


def kl_mfcc_divergence(
    out_mfcc: torch.Tensor,
    source_patches: torch.Tensor,
    n_bins: int = 30,
) -> float:
    W_t = source_patches.shape[3]
    pad_left = W_t // 2
    src_frames = source_patches[:, 0, :, pad_left].cpu().numpy()
    out_frames = out_mfcc[0].T.cpu().numpy()

    kl_sum = 0.0
    n_coeff = src_frames.shape[1]
    for c in range(n_coeff):
        mn = min(src_frames[:, c].min(), out_frames[:, c].min())
        mx = max(src_frames[:, c].max(), out_frames[:, c].max()) + 1e-8
        bins = np.linspace(mn, mx, n_bins + 1)

        p, _ = np.histogram(src_frames[:, c], bins=bins, density=True)
        q, _ = np.histogram(out_frames[:, c], bins=bins, density=True)

        p = p + 1e-9
        p /= p.sum()
        q = q + 1e-9
        q /= q.sum()

        kl_sum += rel_entr(p, q).sum()

    return float(kl_sum / n_coeff)


def patch_diversity(history_indices: torch.Tensor) -> float:
    valid = history_indices[:, 0]
    valid = valid[valid >= 0]
    if valid.numel() == 0:
        return 0.0
    return float(valid.unique().numel() / valid.numel())


def temporal_continuity(out_mfcc: torch.Tensor) -> float:
    frames = out_mfcc[0].T
    if frames.shape[0] < 2:
        return float("nan")
    a = frames[:-1]
    b = frames[1:]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return float(cos.mean().item())


def source_coverage(history_indices: torch.Tensor, n_source_patches: int) -> float:
    valid = history_indices[:, 0]
    valid = valid[valid >= 0]
    if valid.numel() == 0 or n_source_patches == 0:
        return 0.0
    return float(valid.unique().numel() / n_source_patches)


def composite_score_s(
    kl: float,
    diversity: float,
    coverage: float,
    empty_smt_steps: int = 0,
    total_steps: int | None = None,
) -> tuple[float, float, float, float]:
    fidelity_factor = float(np.exp(-kl))
    variety_factor = float((diversity + coverage) / 2.0)
    if total_steps is None or total_steps <= 0:
        stability_factor = 1.0
    else:
        stability_factor = float(max(0.0, 1.0 - (empty_smt_steps / float(total_steps))))
    score_s = fidelity_factor * variety_factor * stability_factor
    return score_s, fidelity_factor, variety_factor, stability_factor


def calculate_composite_s(
    df: pd.DataFrame,
    target_secs: float = 5.0,
    source_sr: int = 22050,
) -> pd.DataFrame:
    """Rank grid rows by composite S; adds T_steps and factor columns."""
    if df.empty:
        return df

    pdf = df.copy()
    hop_samples = (pdf["hop_ms"] / 1000.0) * source_sr
    pdf["T_steps"] = np.round((target_secs * source_sr) / hop_samples).astype(int)

    if "kl_div" in pdf.columns:
        pdf["fidelity_factor"] = np.exp(-pdf["kl_div"])
    else:
        pdf["fidelity_factor"] = 0.0

    if "diversity" in pdf.columns and "coverage" in pdf.columns:
        pdf["variety_factor"] = (pdf["diversity"] + pdf["coverage"]) / 2.0
    else:
        pdf["variety_factor"] = 0.0

    if "empty_smt_steps" in pdf.columns:
        pdf["stability_factor"] = (
            1.0 - (pdf["empty_smt_steps"] / pdf["T_steps"])
        ).clip(lower=0.0)
    else:
        pdf["stability_factor"] = 1.0

    pdf["composite_score_s"] = (
        pdf["fidelity_factor"] * pdf["variety_factor"] * pdf["stability_factor"]
    )

    sort_cols = ["composite_score_s", "kl_div"] if "kl_div" in pdf.columns else ["composite_score_s"]
    sort_asc = [False, True] if "kl_div" in pdf.columns else [False]
    return pdf.sort_values(by=sort_cols, ascending=sort_asc).reset_index(drop=True)


def evaluate_synthesis_run(
    out_mfcc: torch.Tensor,
    history_indices: torch.Tensor,
    source_patches: torch.Tensor,
) -> dict:
    n_src = source_patches.shape[0]
    kl_val = kl_mfcc_divergence(out_mfcc, source_patches)
    div_val = patch_diversity(history_indices)
    cont_val = temporal_continuity(out_mfcc)
    cov_val = source_coverage(history_indices, n_src)
    score_s, fidelity_factor, variety_factor, stability_factor = composite_score_s(
        kl=kl_val,
        diversity=div_val,
        coverage=cov_val,
        empty_smt_steps=0,
        total_steps=out_mfcc.shape[-1],
    )
    return {
        "kl_div": round(kl_val, 4),
        "diversity": round(div_val, 4),
        "continuity": round(cont_val, 4),
        "coverage": round(cov_val, 4),
        "fidelity_factor": round(fidelity_factor, 4),
        "variety_factor": round(variety_factor, 4),
        "stability_factor": round(stability_factor, 4),
        "composite_score_s": round(score_s, 4),
    }

"""Grid driver: build_corpus_state → synthesize_audio_mfcc → reconstruct_audio."""

from __future__ import annotations

import numpy as np
from eval_metrics import evaluate_synthesis_run
from eval_metrics import composite_score_s


def fista_sparse_numpy(
    I: np.ndarray,
    basis: np.ndarray,
    lambd: float = 0.1,
    num_iter: int = 50,
) -> np.ndarray:
    """NumPy wrapper around SMT.compute_smt_embeddings."""
    import torch
    from SMT import compute_smt_embeddings

    I_t     = torch.tensor(I,     dtype=torch.float32)
    basis_t = torch.tensor(basis, dtype=torch.float32)
    BtB     = basis_t.t().mm(basis_t)
    L       = torch.linalg.eigvalsh(BtB).max().item()
    eta     = float(1.0 / (L + 1e-8))
    ahat    = compute_smt_embeddings(I_t, basis_t, lambd=lambd,
                                     num_iter=num_iter, eta=eta, BtB=BtB)
    return ahat.numpy()


def run_grid_config(
    audio: np.ndarray,
    source_sr: int,
    patch_ms: float,
    hop_ms: float,
    win_ms: float,
    n_mfcc: int,
    K: int,
    D: int,
    eps_ssd: float,
    eps_smt: float,
    r_loc: float,
    target_secs: float,
    recon_method: str = "stitch",
    weighted_sampling: bool = False,
    seed_fg_min: int = 0,
    seed_fg_max: int | None = None,
    seed_energy_min: float | None = None,
    seed_energy_max: float | None = None,
    use_var_osc: bool = False,
    var_osc_strength: float = 0.5,
    var_osc_period: float = 400.0,
    return_metrics: bool = False,
    corpus_state: "dict | None" = None,
) -> "np.ndarray | tuple[np.ndarray, dict] | None":
    """One synthesis config; pass corpus_state to skip rebuild when only eps/r_loc change."""
    from speech_synth import build_corpus_state
    from Algos import (
        synthesize_audio_mfcc,
        reconstruct_audio,
        set_seed,
    )

    hop_length = int(round(hop_ms / 1000.0 * source_sr))
    out_steps  = int(round(target_secs * source_sr / hop_length))

    print(
        f"\n  patch_ms={patch_ms:>6.1f}  hop_ms={hop_ms:>5.1f}  "
        f"out_steps={out_steps}"
    )

    set_seed(0)
    if corpus_state is None:
        try:
            corpus_state = build_corpus_state(
                raw_audio_list = audio,
                target_sr      = source_sr,
                n_mfcc         = n_mfcc,
                patch_ms       = patch_ms,
                hop_ms         = hop_ms,
                win_ms         = win_ms,
                smt_dict_size  = K,
                smt_embed_dim  = D,
            )
        except Exception as exc:
            print(f"  [warn] build_corpus_state failed ({exc}) — skipping config")
            return (None, {}) if return_metrics else None
    else:
        print("  [corpus] reusing pre-built corpus state")

    patches_t = corpus_state["patches_tensor"]
    smt_basis = corpus_state["smt_basis"]

    try:
        debug: dict = {}
        if return_metrics:
            out_mfcc, history, debug = synthesize_audio_mfcc(
                patches          = patches_t,
                smt_basis        = smt_basis,
                out_time_steps   = out_steps,
                R_loc            = r_loc,
                eps_ssd          = eps_ssd,
                eps_smt          = eps_smt,
                seed_fg_min      = seed_fg_min,
                seed_fg_max      = seed_fg_max,
                seed_energy_min  = seed_energy_min,
                seed_energy_max  = seed_energy_max,
                weighted         = weighted_sampling,
                use_var_osc      = use_var_osc,
                var_osc_strength = var_osc_strength,
                var_osc_period   = var_osc_period,
                return_debug     = True,
            )
        else:
            out_mfcc, history = synthesize_audio_mfcc(
                patches          = patches_t,
                smt_basis        = smt_basis,
                out_time_steps   = out_steps,
                R_loc            = r_loc,
                eps_ssd          = eps_ssd,
                eps_smt          = eps_smt,
                seed_fg_min      = seed_fg_min,
                seed_fg_max      = seed_fg_max,
                seed_energy_min  = seed_energy_min,
                seed_energy_max  = seed_energy_max,
                weighted         = weighted_sampling,
                use_var_osc      = use_var_osc,
                var_osc_strength = var_osc_strength,
                var_osc_period   = var_osc_period,
            )
    except Exception as exc:
        print(f"  [warn] synthesis failed ({exc}) — skipping config")
        return (None, {}) if return_metrics else None

    metrics: dict = {}
    if return_metrics:
        try:
            metrics = evaluate_synthesis_run(
                out_mfcc        = out_mfcc,
                history_indices = history,
                source_patches  = patches_t,
            )
        except Exception as exc:
            print(f"  [warn] metrics computation failed ({exc})")
        metrics["empty_smt_steps"] = int(debug.get("empty_smt_steps", 0))
        score_s, fidelity_factor, variety_factor, stability_factor = composite_score_s(
            kl=float(metrics.get("kl_div", 0.0)),
            diversity=float(metrics.get("diversity", 0.0)),
            coverage=float(metrics.get("coverage", 0.0)),
            empty_smt_steps=int(metrics["empty_smt_steps"]),
            total_steps=int(out_mfcc.shape[-1]),
        )
        metrics["fidelity_factor"] = round(fidelity_factor, 4)
        metrics["variety_factor"] = round(variety_factor, 4)
        metrics["stability_factor"] = round(stability_factor, 4)
        metrics["composite_score_s"] = round(score_s, 4)

    try:
        if recon_method == "stitch":
            synth_audio = reconstruct_audio(
                method         = "stitch",
                chosen_indices = history,
                source_audio   = corpus_state["corpus_audio_cat"],
                hop_length     = corpus_state["hop_length"],
                window_t       = corpus_state["W_t"],
            )
        else:
            synth_audio = reconstruct_audio(
                method     = "griffin_lim",
                synth_mfcc = out_mfcc,
                hop_length = corpus_state["hop_length"],
                sr         = source_sr,
                n_mels     = 128,
            )
    except Exception as exc:
        print(f"  [warn] reconstruction failed ({exc}); returning silence")
        synth_audio = np.zeros(int(target_secs * source_sr), dtype=np.float32)

    return (synth_audio, metrics) if return_metrics else synth_audio

"""
grid_search.py
==============
Grid search over patch size × hop size configurations for the three-pillar
non-parametric audio synthesiser.

Each configuration delegates entirely to the canonical pipeline:

    build_corpus_state  (speech_synth.py)
        ZCA whitening → K-means dictionary → SMT spectral decomp
        Uses ``preprocess_patches`` (audio_extract.py) which applies
        per-feature standardisation + ZCA + L2 normalisation — the full
        preprocessing stack — and respects utterance boundaries when
        computing the slowness matrix.

    synthesize_audio_mfcc / synthesize_audio_ablation  (Algos.py)
        Frontier-based three-pillar synthesis with Gaussian-weighted SSD,
        locality gate, and FISTA SMT embedding (SMT.py).

    reconstruct_audio  (Algos.py)
        Patch stitching or Griffin-Lim inversion.

This guarantees the grid search and the main notebook exercise **exactly
the same code path** — no duplicate implementations.

Public API
----------
run_grid_config  – synthesize one (patch_ms, hop_ms) configuration
"""

from __future__ import annotations

import numpy as np
from eval_metrics import evaluate_synthesis_run


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

def fista_sparse_numpy(
    I: np.ndarray,
    basis: np.ndarray,
    lambd: float = 0.1,
    num_iter: int = 50,
) -> np.ndarray:
    """
    NumPy-compatible FISTA sparse-coding shim.

    Previously this was a standalone NumPy implementation in ``grid_search.py``.
    It now delegates to the optimised :func:`SMT.compute_smt_embeddings`
    (PyTorch, ``@torch.no_grad()``, pre-allocated buffers) so that callers
    that import this symbol continue to work.

    Args:
        I:        (features, batch_size) float32 array — input observations.
        basis:    (features, M) float32 array — dictionary columns.
        lambd:    L1 regularisation weight.
        num_iter: FISTA iterations.

    Returns:
        ahat: (M, batch_size) float32 numpy array of sparse codes.
    """
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
    var_osc_window: int = 100,
    var_osc_strength: float = 0.5,
    var_osc_period: float = 400.0,
    return_metrics: bool = False,
    corpus_state: "dict | None" = None,
) -> "np.ndarray | tuple[np.ndarray, dict] | None":
    """
    Run the full three-pillar synthesis for one ``(patch_ms, hop_ms)`` config.

    Delegates entirely to :func:`speech_synth.build_corpus_state`,
    :func:`Algos.synthesize_audio_mfcc` (or
    :func:`Algos.synthesize_audio_ablation` for the ``'euclidean'`` path),
    and :func:`Algos.reconstruct_audio`.  The grid search therefore tests
    the **exact same synthesis algorithm** as the main notebook.

    Preprocessing (ZCA whitening, K-means, spectral decomp) is handled by
    ``build_corpus_state``, which calls ``preprocess_patches`` from
    ``audio_extract.py``.  This includes per-feature standardisation,
    ZCA whitening, and L2 normalisation — the full preprocessing stack used
    by the main pipeline.  Utterance boundaries are passed correctly to the
    slowness matrix so transitions at the single audio clip's boundary are
    not penalised.

    **Performance tip — sharing ``corpus_state`` across configs:**
    When sweeping only ``eps_ssd``, ``eps_smt``, or ``r_loc`` while keeping
    ``(patch_ms, hop_ms)`` fixed, building the corpus (MFCC + ZCA + K-means +
    spectral decomp) once and passing the result as ``corpus_state`` avoids
    rebuilding it for every combination::

        state = build_corpus_state(audio, target_sr=sr, patch_ms=25, hop_ms=10, ...)
        for eps in [0.1, 0.3, 0.5]:
            run_grid_config(..., eps_ssd=eps, corpus_state=state)

    Args:
        audio:            Source waveform (float32, mono, at *source_sr* Hz).
                          Ignored when *corpus_state* is provided.
        source_sr:        Sample rate of *audio*.
        patch_ms:         Patch width in milliseconds.
        hop_ms:           MFCC frame hop in milliseconds.
        win_ms:           MFCC analysis window in milliseconds.
        n_mfcc:           Number of MFCC coefficients.
        K:                K-means dictionary size (``smt_dict_size``).
        D:                SMT embedding dimension (``smt_embed_dim``).
        eps_ssd:          Adaptive pool width for Pillar 1 (SSD gate).
        eps_smt:          Adaptive pool width for Pillar 3 (SMT gate).
        r_loc:            Locality radius for Pillar 2.
        target_secs:      Desired output duration in seconds.
        recon_method:     ``'stitch'`` or ``'griffin_lim'``.
        weighted_sampling: Uniform (False) or softmin (True) sampling.
        seed_fg_min:      Lower bound on admissible seed patch index (inclusive).
        seed_fg_max:      Upper bound on admissible seed patch index (exclusive).
        seed_energy_min:  Optional lower L1 energy bound for seed selection.
        seed_energy_max:  Optional upper L1 energy bound for seed selection.
        use_var_osc:      Enable optional variance-oscillation fourth pillar.
        var_osc_window:   Sliding window length (frames) for recent-variance history.
        var_osc_strength: Strength of variance oscillation bias.
        var_osc_period:   Approximate oscillation period in frames.
        return_metrics:   Return ``(waveform, metrics_dict)`` when True.
        corpus_state:     Optional pre-built corpus dict from
                          :func:`speech_synth.build_corpus_state`.  When
                          supplied, ``build_corpus_state`` is skipped entirely,
                          making this call cheap when only sampling parameters
                          (``eps_*``, ``r_loc``) vary.

    Returns:
        Synthesised waveform (float32 numpy array), or
        ``(waveform, metrics_dict)`` if ``return_metrics=True``.
        Returns ``None`` / ``(None, {})`` on failure.
    """
    from speech_synth import build_corpus_state
    from Algos import (
        synthesize_audio_mfcc,
        synthesize_audio_ablation,
        reconstruct_audio,
        set_seed,
    )

    hop_length = int(round(hop_ms / 1000.0 * source_sr))
    out_steps  = int(round(target_secs * source_sr / hop_length))

    print(
        f"\n  patch_ms={patch_ms:>6.1f}  hop_ms={hop_ms:>5.1f}  "
        f"out_steps={out_steps}"
    )

    # ── Build corpus state (skipped if pre-built state is provided) ───────────
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

    patches_t = corpus_state["patches_tensor"]  # (N, 1, n_mfcc, W_t) on device
    smt_basis = corpus_state["smt_basis"]       # (flat_dim, d) on device

    # ── Synthesis ─────────────────────────────────────────────────────────────
    try:
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
            var_osc_window   = var_osc_window,
            var_osc_strength = var_osc_strength,
            var_osc_period   = var_osc_period,
        )
    except Exception as exc:
        print(f"  [warn] synthesis failed ({exc}) — skipping config")
        return (None, {}) if return_metrics else None

    # ── Optional metrics ──────────────────────────────────────────────────────
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

    # ── Audio reconstruction ──────────────────────────────────────────────────
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

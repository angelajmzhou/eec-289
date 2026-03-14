"""
grid_search.py
==============
Self-contained grid search over patch size × hop size configurations for the
three-pillar non-parametric audio synthesiser.

Each configuration runs the full pipeline from scratch:
    ZCA whitening → K-means dictionary → SMT spectral decomp (slowness criterion)
    → three-pillar synthesis loop → MFCC inversion (Griffin-Lim)

The SMT basis is built via the generalised eigenproblem M u = λ V u
(audio_extract.spectral_decomp), matching the formulation in SMT.py /
Final.ipynb.  Formulation 2 (count-weighted centroid covariance / PCA) has
been removed.

Public API
----------
fista_sparse_numpy  – numpy FISTA (no GPU needed, used inside the pipeline)
run_grid_config     – synthesize one (patch_ms, hop_ms) configuration
"""

from __future__ import annotations

import numpy as np
import librosa
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from audio_extract import spectral_decomp
from eval_metrics import evaluate_synthesis_run


# ---------------------------------------------------------------------------
# FISTA (numpy-only, no torch requirement in this module)
# ---------------------------------------------------------------------------

def fista_sparse_numpy(
    y: np.ndarray,
    D: np.ndarray,
    lam: float = 0.1,
    n_iter: int = 50,
) -> np.ndarray:
    """
    FISTA for the non-negative Lasso: ``argmin_{x≥0} 0.5||Dx-y||² + λ||x||₁``.

    Matches ``compute_smt_embeddings`` in SMT.py (non-negative codes as per the
    paper).  Pure-numpy, no GPU required.

    Args:
        y:      (m,)    — observation vector.
        D:      (m, k)  — dictionary matrix (SMT basis columns).
        lam:    L1 regularisation weight.
        n_iter: number of FISTA iterations.
    Returns:
        x: (k,) non-negative sparse code.
    """
    m, k = D.shape
    DtD  = D.T @ D
    Dty  = D.T @ y
    L    = float(np.linalg.norm(DtD, ord=2)) + 1e-8
    x    = np.zeros(k)
    z    = x.copy()
    t    = 1.0
    for _ in range(n_iter):
        grad  = DtD @ z - Dty
        x_new = np.maximum(z - grad / L - lam / L, 0.0)   # non-negative soft threshold
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        z     = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t  = x_new, t_new
    return x


def fista_sparse_numpy_batch(
    Y: np.ndarray,
    D: np.ndarray,
    lam: float = 0.1,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Batched FISTA for the non-negative Lasso over N observations simultaneously.

    Equivalent to calling ``fista_sparse_numpy`` once per column of Y but
    vectorised over the batch dimension for efficiency.

    Args:
        Y:      (m, N)  — N observation vectors stacked as columns.
        D:      (m, k)  — dictionary matrix (SMT basis columns).
        lam:    L1 regularisation weight.
        n_iter: number of FISTA iterations.
    Returns:
        X: (N, k) non-negative sparse codes, one row per observation.
    """
    m, k = D.shape
    N    = Y.shape[1]
    DtD  = D.T @ D           # (k, k)
    DtY  = D.T @ Y           # (k, N)
    L    = float(np.linalg.norm(DtD, ord=2)) + 1e-8
    X    = np.zeros((k, N), dtype=np.float32)
    Z    = X.copy()
    t    = 1.0
    for _ in range(n_iter):
        grad  = DtD @ Z - DtY                              # (k, N)
        X_new = np.maximum(Z - grad / L - lam / L, 0.0)   # non-negative soft threshold
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        Z     = X_new + ((t - 1) / t_new) * (X_new - X)
        X, t  = X_new, t_new
    return X.T  # (N, k)


# ---------------------------------------------------------------------------
# Full three-pillar pipeline for a single grid configuration
# ---------------------------------------------------------------------------
def sample_from_candidates(cand_idx, dist, dmin, weighted=False, h_mult=0.3):
    if len(cand_idx) == 0: return -1
    if len(cand_idx) == 1: return cand_idx[0]
    if not weighted:
        return np.random.choice(cand_idx)
    
    # Softmin sampling
    temp = h_mult * (dmin + 1e-8)
    probs = np.exp(-(dist - dmin) / temp)
    probs /= (np.sum(probs) + 1e-8)
    return np.random.choice(cand_idx, p=probs)
def run_grid_config(
    audio: np.ndarray,
    source_sr: int,
    patch_ms: float,
    hop_ms: float,
    win_ms: float,
    n_mfcc: int,
    K: int,
    D: int,
    eps_ssd: float,       # Pillar 1 Epsilon
    eps_smt: float,       # Pillar 3 Epsilon
    r_loc: float,         # Pillar 2 Radius
    target_secs: float,
    recon_method: str = "griffin_lim",
    pillar3_method: str = "smt",
    weighted_sampling: bool = False,
    return_metrics: bool = False,
) -> "np.ndarray | tuple[np.ndarray, dict] | None":
    """
    Run the full three-pillar synthesis for one ``(patch_ms, hop_ms)`` config.

    All pipeline stages (ZCA, K-means, spectral decomp, FISTA) are recomputed
    from scratch so each configuration is independent.

    Args:
        audio:        Source waveform (float32, mono, at *source_sr* Hz).
        source_sr:    Sample rate of *audio*.
        patch_ms:     Patch width in milliseconds.
        hop_ms:       MFCC frame hop in milliseconds.
        win_ms:       MFCC analysis window in milliseconds.
        n_mfcc:       Number of MFCC coefficients.
        K:            K-means dictionary size.
        D:            SMT embedding dimension (number of spectral eigenvectors).
        eps:          Adaptive pool width (R = (1+eps)*min_dist).
        r_loc:        Locality radius for Pillar 2.
        target_secs:  Desired output duration in seconds.
        recon_method:    'stitch' copies hop_length-sample chunks from *audio*
                         (preserves natural timbre; recommended for speech).
                         'griffin_lim' inverts MFCCs via Griffin-Lim (works without
                         source audio but sounds phaseless/buzzy; fine for textures).
        pillar3_method:  Controls the high-level statistic used in Pillar 3.
                         'smt'       – SMT spectral embedding + FISTA sparse code
                                       distance (default; matches the paper).
                         'euclidean' – plain L2 distance in ZCA-whitened patch
                                       space; faster and requires no eigdecomp.
        return_metrics:  When True, returns a ``(audio, metrics_dict)`` tuple where
                         ``metrics_dict`` contains kl_div, diversity, continuity,
                         coverage, and composite scores.  When False (default),
                         returns only the synthesised waveform (backward-compatible).

    Returns:
        * ``return_metrics=False`` (default): synthesised waveform (float32 numpy
          array) or ``None`` if the config is degenerate.
        * ``return_metrics=True``: ``(waveform, metrics_dict)`` tuple, or
          ``(None, {})`` if the config is degenerate.
    """
    # ── Step 0: derive frame-level parameters ─────────────────────────────────
    hop_length = int(round(hop_ms / 1000.0 * source_sr))
    win_length = int(round(win_ms / 1000.0 * source_sr))
    W_t        = max(1, round((patch_ms / 1000.0) * source_sr / hop_length))
    out_steps  = int(round(target_secs * source_sr / hop_length))

    # n_fft must be large enough to support n_mfcc mel bins — a 32-sample FFT
    # with n_mfcc=40 is ill-conditioned.  Clamp to at least 4× hop_length and
    # a 256-point minimum so MFCC computation is numerically valid regardless of
    # how short win_ms or hop_ms are.
    n_fft_analysis = max(win_length, hop_length * 4, 256)

    print(
        f"\n  patch_ms={patch_ms:>6.1f}  hop_ms={hop_ms:>5.1f}  "
        f"W_t={W_t}  flat_dim={n_mfcc * W_t}  out_steps={out_steps}"
    )

    # ── Step 1: extract MFCC features ─────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y          = audio.astype(np.float32),
        sr         = source_sr,
        n_mfcc     = n_mfcc,
        hop_length = hop_length,
        n_fft      = n_fft_analysis,
    ).T  # (T_frames, n_mfcc)

    T_frames = mfcc.shape[0]
    if T_frames < W_t + out_steps:
        repeats  = (W_t + out_steps) // T_frames + 1
        mfcc     = np.tile(mfcc, (repeats, 1))[: W_t + out_steps]
        T_frames = mfcc.shape[0]

    # ── Step 2: extract time patches ─────────────────────────────────────────
    patches = np.stack(
        [mfcc[i : i + W_t] for i in range(T_frames - W_t + 1)],
        axis=0,
    )  # (N_patches, W_t, n_mfcc)
    N_patches = patches.shape[0]

    MAX_SYNTH_PATCHES = 50_000
    # Track mapping from (possibly subsampled) local index → original MFCC frame index
    # so stitch reconstruction can copy audio from the correct source position.
    global_frame_map = np.arange(N_patches)   # identity before any subsampling
    if N_patches > MAX_SYNTH_PATCHES:
        rng_sub = np.random.default_rng(42)
        keep      = rng_sub.choice(N_patches, MAX_SYNTH_PATCHES, replace=False)
        keep.sort()
        patches          = patches[keep]
        global_frame_map = keep          # local idx i → original MFCC frame keep[i]
        N_patches        = MAX_SYNTH_PATCHES
        print(f"  [subsample] capped source pool to {MAX_SYNTH_PATCHES:,} patches")

    patches_flat = patches.reshape(N_patches, -1).astype(np.float32)
    del patches  # free the 3D float64 source array; patches_flat is an independent copy

    # ── Step 3: ZCA whitening ─────────────────────────────────────────────────
    mu    = patches_flat.mean(axis=0, keepdims=True)
    X_c   = patches_flat - mu
    cov   = (X_c.T @ X_c) / (N_patches - 1)
    U, S_zca, _ = np.linalg.svd(cov)
    W_zca = U @ np.diag(1.0 / np.sqrt(S_zca + 1e-6)) @ U.T
    patches_w = (X_c @ W_zca.T).astype(np.float32)
    del cov, U, S_zca, X_c

    # ── Step 4: K-means dictionary ────────────────────────────────────────────
    km = MiniBatchKMeans(
        n_clusters   = K,
        random_state = 42,
        max_iter     = 300,
        batch_size   = min(512, N_patches),
    )
    km.fit(patches_w)
    centroids = km.cluster_centers_  # (K, flat_dim)

    # ── Step 5: SMT spectral decomposition (skipped for 'euclidean' mode) ──────
    # Uses Formulation 1: generalised eigenproblem M u = λ V u where M is the
    # slowness (temporal-penalty) matrix and V is the cluster-usage covariance.
    # Eigenvectors with the SMALLEST λ capture the slowest-varying directions —
    # the SMT manifold basis.  Matches audio_extract.spectral_decomp / SMT.py.
    basis   = None
    src_emb = None
    if pillar3_method == 'smt':
        labels = km.predict(patches_w)
        D_eff  = min(D, K - 1)
        try:
            _, evecs = spectral_decomp(labels, [(0, N_patches)], K)
            # Take the D_eff smallest non-trivial eigenvectors (skip index 0)
            P      = evecs[:, 1 : D_eff + 1].T          # (D_eff, K)
            # Project L2-normalised cluster centres into SMT spectral space
            cc_norm = normalize(centroids, axis=1, norm='l2')  # (K, flat_dim)
            basis   = normalize(cc_norm.T @ P.T, axis=0, norm='l2').astype(np.float32)
            # basis shape: (flat_dim, D_eff)
        except Exception as exc:
            print(f"  [warn] spectral decomp failed ({exc}) — skipping config")
            return None

    # ── Step 6: SMT embeddings for all source patches via FISTA ──────────────
    # All source patches are FISTA-encoded against the SMT basis — consistent
    # with compute_smt_embeddings() in SMT.py and the synthesis loop below.
    # Y shape: (flat_dim, N_patches) → src_emb shape: (N_patches, D_eff)
    if pillar3_method == 'smt':
        src_emb = fista_sparse_numpy_batch(
            Y      = patches_w.T.astype(np.float32),
            D      = basis,
            lam    = 0.1,
            n_iter = 50,
        )  # (N_patches, D_eff)
    src_ssd = patches_flat            # view — no extra copy

    synth_frames: list[np.ndarray] = []
    chosen_audio_frames: list[int] = []
    chosen_local_indices: list[int] = []

    # ── Seed selection: highest-variance patch in the source pool ─────────────
    # Matches choose_seed_patch_index() in Algos.py.  Using mfcc[:W_t] (the raw
    # first frames) risks starting from silence or a fade-in; the highest-variance
    # patch is the most perceptually salient region to grow synthesis outward from.
    patch_variances = patches_flat.var(axis=1)          # (N_patches,)
    seed_local_idx  = int(patch_variances.argmax())
    seed_frame_idx  = global_frame_map[seed_local_idx]  # original MFCC frame
    context         = list(mfcc[seed_frame_idx : seed_frame_idx + W_t])
    # fall back to mfcc start if the slice is shorter than W_t (shouldn't happen)
    if len(context) < W_t:
        context = list(mfcc[:W_t])
    del mfcc  # no longer needed; context list holds the seed frames

    # Temporal positions of every source patch, normalised to [0, 1].
    # Matches synthesize_audio_mfcc's `source_time_positions` linspace.
    # Uses global_frame_map so subsampled pools are still correctly positioned.
    source_time_positions = global_frame_map / max(T_frames - 1, 1)  # (N_patches,)

    all_local = np.arange(N_patches)

    # Pre-allocate a reusable (N, flat_dim) buffer for the per-step SSD diff so
    # NumPy doesn't malloc/free a fresh ~160 MB array on every synthesis step.
    _diff_buf = np.empty_like(patches_flat)  # shape (N_patches, flat_dim)

    # Gaussian weights for d_SSD = Σ_x G(x)[ω(p;x)-ω'(x)]²  (paper §2.4)
    # flat dim order: (W_t, n_mfcc) so gauss repeats for each MFCC coefficient.
    pad_left_g = W_t // 2
    t_gauss    = np.arange(W_t, dtype=np.float32) - pad_left_g
    sigma_g    = max(W_t / 4.0, 0.5)
    gauss_1d   = np.exp(-0.5 * (t_gauss / sigma_g) ** 2)
    gauss_1d  /= gauss_1d.sum()
    gauss_flat = np.repeat(gauss_1d, n_mfcc)  # (W_t * n_mfcc,)

    for _step in range(out_steps):
        ctx_patch = np.array(context[-W_t:]).flatten()
        ctx_w     = (ctx_patch - mu.ravel()) @ W_zca.T

        # ── Pillar 2: Locality — temporal proximity (matches Algos.py) ────────
        # Compare normalised output-time position against each source patch's
        # original position in the corpus.  r_loc=1.0 → |dist|≤1.0 for all
        # patches in [0,1] → effectively disabled (global search).
        target_time_pos = _step / max(out_steps - 1, 1)
        dist_loc = np.abs(source_time_positions - target_time_pos)
        loc_mask = dist_loc <= r_loc
        if loc_mask.sum() < 1:
            loc_mask = np.ones(N_patches, dtype=bool)

        # ── Pillar 1: Gaussian-weighted SSD over ALL source patches ──────────
        # R_SSD(p) := (1+ε) min_{all ω''} d_SSD(ω(p),ω'')   (paper §2.4)
        ctx_ssd   = ctx_patch
        # In-place ops reuse _diff_buf — avoids allocating a fresh (N, flat_dim)
        # array (~160 MB for large configs) on every synthesis step.
        np.subtract(src_ssd, ctx_ssd, out=_diff_buf)   # (N, flat_dim) no alloc
        np.multiply(_diff_buf, _diff_buf, out=_diff_buf)  # square in-place
        ssd_dists = _diff_buf @ gauss_flat              # (N,) — single matmul

        dmin_ssd  = ssd_dists.min() + 1e-12                       # global min (paper)
        mask_ssd  = ssd_dists <= (1 + eps_ssd) * dmin_ssd         # (N,) global P1
        # P1 ∩ P2
        loc_local    = all_local[loc_mask]                         # kept for fallback
        p1p2_mask    = mask_ssd & loc_mask
        p1_indices   = all_local[p1p2_mask]
        p1_ssd_dists = ssd_dists[p1p2_mask]

        # ── Pillar 3: Semantic SMT — min over ALL source embeddings ──────────
        # R_SSL(p) := (1+ε) min_{all ω''} d_SSL(ω(p),ω'')   (paper §2.4)
        if pillar3_method == 'smt' and len(p1_indices) > 0:
            ctx_code      = fista_sparse_numpy(ctx_w, basis, lam=0.1, n_iter=50)
            all_smt_dists = np.sum((src_emb - ctx_code) ** 2, axis=1)  # (N,) global
            dmin_smt      = all_smt_dists.min() + 1e-12
            smt_mask_g    = all_smt_dists <= (1 + eps_smt) * dmin_smt   # (N,) global P3

            # Final pool: P1 ∩ P2 ∩ P3 (uniform sampling per paper §2.2)
            final_mask     = p1p2_mask & smt_mask_g
            final_cand_idx = all_local[final_mask]
            if len(final_cand_idx) == 0:
                final_mask     = p1p2_mask   # fallback: relax P3
                final_cand_idx = p1_indices

            best_local_idx = sample_from_candidates(
                cand_idx = final_cand_idx,
                dist     = ssd_dists[final_mask],
                dmin     = dmin_ssd,
                weighted = False,   # uniform per paper §2.2
                h_mult   = 0.3,
            )
        else:
            # Euclidean fallback or empty Pillar-3 / non-SMT mode
            if len(p1_indices) == 0:
                p1_indices   = loc_local
                p1_ssd_dists = ssd_dists[loc_mask]
            fallback_dists = np.sum((patches_w[p1_indices] - ctx_w) ** 2, axis=1)
            best_local_idx = p1_indices[np.argmin(fallback_dists)]

        # --- Finalize Step ---
        best_patch = patches_flat[best_local_idx]
        synth_frame = best_patch[(-n_mfcc):]
        synth_frames.append(synth_frame)
        context.append(synth_frame)
        chosen_local_indices.append(int(best_local_idx))

        if recon_method == "stitch":
            orig_patch_start = global_frame_map[best_local_idx]
            chosen_audio_frames.append(int(orig_patch_start) + W_t - 1)

    # ── Step 8: Reconstruction ────────────────────────────────────────────────
    if recon_method == "stitch":
        # Copy hop_length-sample chunks from the original waveform at each chosen
        # MFCC frame position.  Chunks are real speech audio so timbre is preserved.
        chunks: list[np.ndarray] = []
        for frame_idx in chosen_audio_frames:
            start = frame_idx * hop_length
            end   = start + hop_length
            chunk = audio[start:end] if end <= len(audio) else audio[start:]
            if len(chunk) < hop_length:
                chunk = np.pad(chunk, (0, hop_length - len(chunk)))
            chunks.append(chunk.astype(np.float32))
        return np.concatenate(chunks)

    # ── Griffin-Lim inversion (fallback / environmental textures) ────────────
    synth_mfcc = np.array(synth_frames).T  # (n_mfcc, out_steps)
    n_fft_inv  = max(n_fft_analysis, hop_length * 4, 1024)
    try:
        log_mel = librosa.feature.inverse.mfcc_to_audio(
            synth_mfcc,
            n_mels=128,
            hop_length=hop_length,
            n_fft=n_fft_inv,
            sr=source_sr,
        )
        return log_mel.astype(np.float32)
    except Exception as e:
        print(f"  [warn] MFCC inversion failed ({e}); returning silence")
        return np.zeros(int(target_secs * source_sr), dtype=np.float32)

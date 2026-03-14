"""
speech_synth.py
===============
Reusable functions for building a LibriSpeech patch corpus and running the
three-pillar non-parametric speech synthesis pipeline.

Public API
----------
load_speech_corpus   – download / read raw LibriSpeech waveforms
build_corpus_state   – extract MFCC patches → ZCA → K-means → SMT basis
synthesize_speech    – run three-pillar synthesis from a pre-built corpus state

Typical usage
-------------
    # 1. Load raw audio once (slow: disk I/O + resampling)
    corpus = load_speech_corpus(data_dir, n_utterances=30, target_sr=16_000)

    # 2. Build corpus state for a given (patch_ms, hop_ms) config (slow: ZCA + K-means)
    state = build_corpus_state(corpus, target_sr=16_000, patch_ms=25.0, hop_ms=10.0)

    # 3. Synthesize many times with different hyperparams (fast: reuses state)
    audio = synthesize_speech(state, out_steps=500, eps=0.10, r_loc=1.0)
    audio = synthesize_speech(state, out_steps=500, eps=0.20, r_loc=1.0, h_mult=0.3)

This split makes hyperparameter grid searches 10–100× faster because the
expensive corpus-building step runs only once per (patch_ms, hop_ms) config.

IMPORTANT — why MFCC is computed on the *concatenated* audio
------------------------------------------------------------
Patch stitching maps ``patch_idx * hop_length`` to a sample position in the
source waveform.  This is only correct when every patch's index matches its
position in a single contiguous array.  If MFCC is computed per-utterance and
patches are concatenated afterwards, patch 300 from utterance 3 has local frame
index 300 but points to the wrong sample in the combined audio — producing
random fragments and incomprehensible output.

Solution: concatenate all utterances (with a ``win_length``-sample silence gap
between them to avoid FFT smear at boundaries), compute one MFCC on the full
array, then extract patches from that.  ``patch_idx * hop_length`` then gives
the exact correct sample offset in ``corpus_state['corpus_audio_cat']``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import librosa
import soundfile as sf
from sklearn.preprocessing import normalize as sk_normalize

from audio_extract import (
    extract_time_patches,
    preprocess_patches,
    apply_kmeans_to_patches,
    spectral_decomp,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load raw audio
# ──────────────────────────────────────────────────────────────────────────────

def load_speech_corpus(
    data_dir: str | Path,
    n_utterances: int = 30,
    target_sr: int = 16_000,
    seed: int = 0,
) -> list[np.ndarray]:
    """
    Load *n_utterances* random utterances from LibriSpeech dev-clean.
    Downloads the dataset (~340 MB) on first call.

    Args:
        data_dir:     Root directory for LibriSpeech data.
        n_utterances: Number of utterances to sample.
        target_sr:    Target sample rate (Hz); all audio is resampled here.
        seed:         Random seed for utterance selection.

    Returns:
        List of float32 mono numpy waveforms, each at *target_sr*.
    """
    import torchaudio

    rng = random.Random(seed)
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = torchaudio.datasets.LIBRISPEECH(
        root=str(data_dir), url="dev-clean", download=True,
    )
    indices = rng.sample(range(len(ds)), min(n_utterances, len(ds)))
    print(f"Loading {len(indices)} utterances from LibriSpeech dev-clean "
          f"({len(ds)} total)...")

    audio_list: list[np.ndarray] = []
    for idx in indices:
        metadata   = ds.get_metadata(idx)
        audio_path = Path(ds._archive) / metadata[0]
        audio_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != target_sr:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=target_sr)
        audio_list.append(audio_np.astype(np.float32))

    total_secs = sum(len(a) for a in audio_list) / target_sr
    print(f"  {len(audio_list)} utterances loaded  ({total_secs:.1f} s total)")
    return audio_list


# ──────────────────────────────────────────────────────────────────────────────
# 2. Build corpus state (patches + SMT basis)
# ──────────────────────────────────────────────────────────────────────────────

def build_corpus_state(
    raw_audio_list: list[np.ndarray],
    target_sr: int = 16_000,
    n_mfcc: int = 40,
    patch_ms: float = 25.0,
    hop_ms: float = 10.0,
    win_ms: float = 20.0,
    smt_dict_size: int = 128,
    smt_embed_dim: int = 128,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Build the MFCC patch corpus and SMT basis from raw audio waveforms.

    Call once per ``(patch_ms, hop_ms)`` configuration.  The returned
    ``corpus_state`` dict is passed directly to :func:`synthesize_speech` and
    can be reused cheaply across many synthesis runs that vary only *eps*,
    *r_loc*, or *h_mult*.

    Args:
        raw_audio_list: List of float32 mono waveforms (from
                        :func:`load_speech_corpus`).
        target_sr:      Sample rate of all waveforms (Hz).
        n_mfcc:         Number of MFCC coefficients per frame.
        patch_ms:       Patch width in milliseconds.
        hop_ms:         MFCC frame hop in milliseconds.
        win_ms:         MFCC analysis window in milliseconds.
        smt_dict_size:  K-means dictionary size (K).
        smt_embed_dim:  SMT spectral embedding dimension (d).
        device:         PyTorch device; defaults to CUDA if available.

    Returns:
        ``corpus_state`` dict with keys:

        - ``patches_tensor``  : (N, 1, n_mfcc, W_t) ``torch.Tensor`` on *device*
        - ``smt_basis``       : (flat_dim, d) ``torch.Tensor`` on *device*
        - ``corpus_audio_cat``: contiguous waveform; ``patch_idx * hop_length`` gives the correct sample offset
        - ``hop_length``      : int — samples per MFCC frame
        - ``W_t``            : int — frames per patch
        - ``target_sr``      : int
        - ``n_mfcc``         : int
        - ``utt_bounds``     : list of ``(start, end)`` patch-index pairs
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hop_length = max(1, int(round(target_sr * hop_ms / 1000)))
    win_length = max(1, int(round(target_sr * win_ms / 1000)))
    # n_fft must be large enough to support n_mfcc mel bins regardless of how
    # short win_ms is (e.g. 2 ms → 32 samples, which is far too small).
    n_fft = max(win_length, hop_length * 4, 256)

    # ── Step 1: Concatenate all utterances into one waveform ──────────────────
    # IMPORTANT: We must compute a SINGLE MFCC on the full concatenated signal.
    # The stitch reconstruction formula is:
    #   start_sample = (patch_idx + frame_in_patch) * hop_length
    # This is only correct when patch_idx == MFCC frame index, which holds only
    # when the patches were extracted from one contiguous MFCC.  If we compute
    # per-utterance MFCCs and concatenate patches, patch_idx no longer maps to
    # the right sample position in corpus_audio_cat.
    silence_gap     = np.zeros(win_length, dtype=np.float32)
    audio_segs      = []
    utt_sample_starts: list[int] = []
    sample_cursor   = 0
    for audio_np in raw_audio_list:
        utt_sample_starts.append(sample_cursor)
        audio_segs.append(audio_np.astype(np.float32))
        audio_segs.append(silence_gap)
        sample_cursor += len(audio_np) + win_length

    corpus_audio_cat = np.concatenate(audio_segs)

    # ── Step 2: One MFCC on the full signal ───────────────────────────────────
    mfcc_full = librosa.feature.mfcc(
        y=corpus_audio_cat,
        sr=target_sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft,
    )  # (n_mfcc, T_total)

    all_patches = extract_time_patches(
        features=mfcc_full,
        sr=target_sr,
        hop_length=hop_length,
        patch_ms=patch_ms,
        patch_hop_frames=1,
    ).astype(np.float32)  # (N, n_mfcc, W_t)

    if all_patches.shape[0] < 2:
        raise RuntimeError(
            "Too few patches for this patch_ms/hop_ms configuration.  "
            "Try a smaller patch_ms or larger hop_ms."
        )

    W_t      = all_patches.shape[2]
    N        = all_patches.shape[0]
    flat_dim = n_mfcc * W_t
    patches_flat = all_patches.reshape(N, flat_dim)

    print(f"  {N:,} patches  (W_t={W_t}, flat_dim={flat_dim})")

    # ── Step 3: Compute utt_bounds in patch-index space ───────────────────────
    # Disables the SMT slowness penalty across utterance boundaries.
    utt_bounds: list[tuple[int, int]] = []
    for audio_np, s_start in zip(raw_audio_list, utt_sample_starts):
        frame_start = s_start // hop_length
        n_utt_frames = int(np.ceil(len(audio_np) / hop_length))
        p_start = min(frame_start, N - 1)
        p_end   = min(frame_start + max(0, n_utt_frames - W_t + 1), N)
        if p_end > p_start:
            utt_bounds.append((p_start, p_end))
    if not utt_bounds:
        utt_bounds = [(0, N)]

    # ── Step 4: ZCA whitening ─────────────────────────────────────────────────
    norm_patches, _mean, _zca = preprocess_patches(patches_flat)

    # ── Step 5: K-means dictionary ────────────────────────────────────────────
    n_clusters = min(smt_dict_size, max(1, N // 4))
    kmeans     = apply_kmeans_to_patches(norm_patches, n_clusters=n_clusters,
                                         sample_size=None)

    # ── Step 6: Spectral decomp (respects utterance boundaries) ───────────────
    eigvals, eigvecs = spectral_decomp(kmeans.labels_, utt_bounds, n_clusters)
    d        = min(smt_embed_dim, eigvecs.shape[1] - 1)
    P        = eigvecs[:, 1:d + 1].T                               # (d, K)
    cc       = sk_normalize(kmeans.cluster_centers_, axis=1, norm='l2')
    basis_np = sk_normalize(cc.T @ P.T, axis=0, norm='l2')         # (flat_dim, d)

    smt_basis      = torch.tensor(basis_np,    dtype=torch.float32, device=device)
    patches_tensor = (torch.tensor(all_patches, dtype=torch.float32, device=device)
                      .unsqueeze(1))   # (N, 1, n_mfcc, W_t)

    print(f"  SMT basis: {tuple(smt_basis.shape)}")

    return {
        "patches_tensor":  patches_tensor,
        "smt_basis":       smt_basis,
        "corpus_audio_cat": corpus_audio_cat,  # contiguous; patch_idx*hop == sample offset
        "hop_length":      hop_length,
        "W_t":             W_t,
        "target_sr":       target_sr,
        "n_mfcc":          n_mfcc,
        "utt_bounds":      utt_bounds,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 3. Synthesize speech (Updated for Dual-Epsilon & Stochastic Sampling)
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_speech(
    corpus_state: dict[str, Any],
    out_steps: int = 500,
    r_loc: float = 1.0,
    eps_ssd: float = 0.10,  # New: Pillar 1 gate
    eps_smt: float = 0.40,  # New: Pillar 3 gate
    h_mult: float = 0.3,    # Sampling temperature
    weighted: bool = True,  # Use stochastic sampler
    seed: int = 0,
) -> np.ndarray:
    """
    Synthesize speech using the Stochastic Three-Pillar sampler.

    Args:
        corpus_state: Dict from :func:`build_corpus_state`.
        out_steps:    Number of MFCC frames to synthesize.
        r_loc:        Locality radius (1.0 = global search).
        eps_ssd:      Adaptive pool width for Pillar 1 (Texture matching).
        eps_smt:      Adaptive pool width for Pillar 3 (Semantic matching).
        h_mult:       Sampling temperature; matches sample_from_candidates logic.
        weighted:     If True, uses distance-weighted softmax sampling.
        seed:         Random seed for reproducibility.
    """
    from Algos import synthesize_audio_mfcc, set_seed, reconstruct_audio

    set_seed(seed)

    # ── Resolve seed patch start point ────────────────────────────────────────
    utt_bounds = corpus_state.get("utt_bounds", [(0, corpus_state["patches_tensor"].shape[0])])
    utt_idx    = seed % len(utt_bounds)
    seed_lo, seed_hi = utt_bounds[utt_idx]

    # Note: Ensure your synthesize_audio_mfcc in Algos.py accepts eps_ssd/eps_smt
    out_mfcc, history = synthesize_audio_mfcc(
        patches        = corpus_state["patches_tensor"],
        smt_basis      = corpus_state["smt_basis"],
        out_time_steps = out_steps,
        R_loc          = r_loc,
        eps_ssd        = eps_ssd,   # Passed separately
        eps_smt        = eps_smt,   # Passed separately
        seed_fg_min    = seed_lo,
        seed_fg_max    = seed_hi,
        weighted       = weighted,
        h_mult         = h_mult,
    )

    return reconstruct_audio(
        method         = "stitch",
        chosen_indices = history,
        source_audio   = corpus_state["corpus_audio_cat"],
        hop_length     = corpus_state["hop_length"],
        window_t       = corpus_state["W_t"],
    )
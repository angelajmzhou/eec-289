"""LibriSpeech helpers: load audio, build one MFCC corpus per (patch_ms, hop_ms), synthesize."""

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


def load_speech_corpus(
    data_dir: str | Path,
    n_utterances: int = 30,
    target_sr: int = 16_000,
    seed: int = 0,
) -> list[np.ndarray]:
    """Random LibriSpeech dev-clean utterances as float32 mono at target_sr."""
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


def build_corpus_state(
    raw_audio_list: "list[np.ndarray] | np.ndarray",
    target_sr: int = 16_000,
    n_mfcc: int = 40,
    patch_ms: float = 25.0,
    hop_ms: float = 10.0,
    win_ms: float = 20.0,
    smt_dict_size: int = 128,
    smt_embed_dim: int = 128,
    device: str | None = None,
) -> dict[str, Any]:
    """One MFCC on concatenated audio, patches, ZCA, k-means, SMT basis; reuse per (patch_ms, hop_ms)."""
    if isinstance(raw_audio_list, np.ndarray):
        raw_audio_list = [raw_audio_list]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hop_length = max(1, int(round(target_sr * hop_ms / 1000)))
    win_length = max(1, int(round(target_sr * win_ms / 1000)))
    n_fft = max(win_length, hop_length * 4, 256)

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
    ).astype(np.float32)

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

    norm_patches, _mean, _zca = preprocess_patches(patches_flat)

    n_clusters = min(smt_dict_size, max(1, N // 4))
    kmeans     = apply_kmeans_to_patches(norm_patches, n_clusters=n_clusters,
                                         sample_size=None)

    eigvals, eigvecs = spectral_decomp(kmeans.labels_, utt_bounds, n_clusters)
    d        = min(smt_embed_dim, eigvecs.shape[1] - 1)
    P        = eigvecs[:, 1:d + 1].T
    cc       = sk_normalize(kmeans.cluster_centers_, axis=1, norm='l2')
    basis_np = sk_normalize(cc.T @ P.T, axis=0, norm='l2')

    smt_basis      = torch.tensor(basis_np,    dtype=torch.float32, device=device)
    patches_tensor = (torch.tensor(all_patches, dtype=torch.float32, device=device)
                      .unsqueeze(1))

    print(f"  SMT basis: {tuple(smt_basis.shape)}")

    return {
        "patches_tensor":  patches_tensor,
        "smt_basis":       smt_basis,
        "corpus_audio_cat": corpus_audio_cat,
        "hop_length":      hop_length,
        "W_t":             W_t,
        "target_sr":       target_sr,
        "n_mfcc":          n_mfcc,
        "utt_bounds":      utt_bounds,
    }

def synthesize_speech(
    corpus_state: dict[str, Any],
    out_steps: int = 500,
    r_loc: float = 1.0,
    eps_ssd: float = 0.10,
    eps_smt: float = 0.40,
    h_mult: float = 0.3,
    weighted: bool = False,
    seed: int = 0,
    seed_patch_bounds: "tuple[float, float] | None" = None,
    return_history: bool = False,
) -> "np.ndarray | tuple[np.ndarray, list[int]]":
    """Three-pillar synthesis from corpus_state; optional L1 energy window for seed."""
    from Algos import synthesize_audio_mfcc, set_seed, reconstruct_audio

    set_seed(seed)

    utt_bounds = corpus_state.get("utt_bounds",
                                  [(0, corpus_state["patches_tensor"].shape[0])])
    utt_idx          = seed % len(utt_bounds)
    seed_lo, seed_hi = utt_bounds[utt_idx]

    energy_lo: float | None = None
    energy_hi: float | None = None
    if seed_patch_bounds is not None:
        energy_lo, energy_hi = float(seed_patch_bounds[0]), float(seed_patch_bounds[1])

    out_mfcc, history = synthesize_audio_mfcc(
        patches          = corpus_state["patches_tensor"],
        smt_basis        = corpus_state["smt_basis"],
        out_time_steps   = out_steps,
        R_loc            = r_loc,
        eps_ssd          = eps_ssd,
        eps_smt          = eps_smt,
        seed_fg_min      = seed_lo,
        seed_fg_max      = seed_hi,
        seed_energy_min  = energy_lo,
        seed_energy_max  = energy_hi,
        weighted         = weighted,
        h_mult           = h_mult,
    )

    audio_out = reconstruct_audio(
        method         = "stitch",
        chosen_indices = history,
        source_audio   = corpus_state["corpus_audio_cat"],
        hop_length     = corpus_state["hop_length"],
        window_t       = corpus_state["W_t"],
    )
    if return_history:
        return audio_out, list(history)
    return audio_out
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.linalg import eigh

import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display, Audio, HTML

"""
Neural Vocoder Inversion
========================
The `reconstruct_audio("griffin_lim")` path already runs Griffin-Lim, an
iterative spectrogram inversion algorithm.  Here we demonstrate how to use
torchaudio's WaveRNN neural vocoder pipeline as a higher-quality alternative.

WaveRNN expects 80 log-mel bins at 22050 Hz.  We: 
  1) Compute a mel spectrogram from the already-stitched audio (or from any
     synthesized signal), 
  2) Normalise it to the WaveRNN's expected range,
  3) Run the vocoder forward pass.
"""

import torchaudio

def neural_vocoder_reconstruct(
    audio_np: np.ndarray,
    sr: int,
    output_path: Path | None = None,
    target_sr: int = 22050,
    n_mels:    int = 80,
):
    """
    Re-synthesize `audio_np` through a WaveRNN neural vocoder.

    Steps:
        1. Resample to 22050 Hz (WaveRNN model rate).
        2. Compute 80-bin log-mel spectrogram with WaveRNN's expected window.
        3. Run torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH vocoder.
        4. Return the new waveform as a numpy array at 22050 Hz.

    The vocoder model is downloaded once and cached by torchaudio.
    """
    from torchaudio.pipelines import TACOTRON2_WAVERNN_PHONE_LJSPEECH as BUNDLE

    # -- obtain the WaveRNN vocoder (downloads ~50 MB on first run) -----------
    print("Loading WaveRNN vocoder (downloads ~50 MB on first run)...")
    vocoder = BUNDLE.get_vocoder().to(device)
    vocoder.eval()

    vocoder_sr = BUNDLE.sample_rate         # 22050

    # -- resample source audio ------------------------------------------------
    wf = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)   # (1, T)
    if sr != vocoder_sr:
        wf = torchaudio.functional.resample(wf, sr, vocoder_sr)

    # -- extract 80-bin log-mel (WaveRNN-compatible) ---------------------------
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=vocoder_sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=n_mels,
        f_min=40.0,
        f_max=vocoder_sr // 2,
        power=1.0,           # amplitude spectrogram (WaveRNN convention)
    ).to(device)

    with torch.no_grad():
        mel = mel_transform(wf.to(device))                   # (1, n_mels, T)
        # Log + clamp to avoid -inf
        mel = torch.log(mel.clamp(min=1e-5))
        # WaveRNN expects mel shape (1, T, n_mels) — transpose last two dims
        mel_t = mel.transpose(1, 2)                          # (1, T, 80)
        output_wf, lengths = vocoder(mel_t, [mel_t.shape[1]])

    audio_out = output_wf[0].squeeze(0).cpu().numpy()

    if output_path is not None:
        sf.write(str(output_path), audio_out, vocoder_sr)
        print(f"Neural vocoder output saved to: {output_path}")

    return audio_out, vocoder_sr


# ── Demo: apply neural vocoder to the last stitched file in the previous run ──
if "audio_stitched" in dir() and audio_stitched is not None:
    print("\nRunning neural vocoder on the last stitched synthesis...")
    try:
        nv_path = OUTPUT_DIR / f"wavernn_{filepath.stem}.wav"
        nv_audio, nv_sr = neural_vocoder_reconstruct(
            audio_stitched, SR,
            output_path=nv_path,
        )
        display(HTML(f"<b>Neural Vocoder (WaveRNN) output:</b> {nv_path.name}"))
        display(Audio(nv_audio, rate=nv_sr))
    except Exception as e:
        print(f"WaveRNN demo skipped: {e}")
        print("(Ensure torchaudio can download model weights; "
              "Griffin-Lim inversion is still available via reconstruct_audio.)")
else:
    print("Run the synthesis loop above first to populate `audio_stitched`.")


def reconstruct_audio(
    method: str,
    synth_mfcc: torch.Tensor = None,
    chosen_indices: torch.Tensor = None,
    source_audio: np.ndarray = None,
    hop_length: int = 512,
    window_t: int = None,
    sr: int = 22050,
    n_mels: int = 128
):
    """
    method: 'stitch' or 'griffin_lim'
    """
    if method == "stitch":
        if chosen_indices is None or source_audio is None:
            raise ValueError("'stitch' requires chosen_indices and source_audio.")

        stitched_audio = []

        for item in chosen_indices:
            patch_idx = item[0].item()
            frame_within_patch = item[1].item()

            if patch_idx == -1:
                # Fallback just in case, though the valid block shouldn't have any left
                stitched_audio.append(np.zeros(hop_length))
                continue

            # The exact frame index in the original MFCC
            # (Assuming stride=1 during patch extraction)
            absolute_frame_idx = patch_idx + frame_within_patch

            start_sample = absolute_frame_idx * hop_length
            end_sample = start_sample + hop_length

            audio_chunk = source_audio[start_sample:end_sample]

            # Pad if we hit the very end of the file
            if len(audio_chunk) < hop_length:
                audio_chunk = np.pad(audio_chunk, (0, hop_length - len(audio_chunk)))

            stitched_audio.append(audio_chunk)

        return np.concatenate(stitched_audio)

    elif method == "griffin_lim":
        if synth_mfcc is None:
            raise ValueError("'griffin_lim' requires synth_mfcc.")

        # Convert tensor back to numpy array: shape (n_mfcc, time_steps)
        mfcc_np = synth_mfcc.squeeze(0).cpu().numpy()

        # librosa's inverse feature requires knowing how many Mel bands were
        # originally used (default is often 128)
        print("Running Griffin-Lim reconstruction (this may take a moment)...")
        audio_recon = librosa.feature.inverse.mfcc_to_audio(
            mfcc_np,
            n_mels=n_mels,
            hop_length=hop_length,
            sr=sr
        )
        return audio_recon

    else:
        raise ValueError("Method must be 'stitch' or 'griffin_lim'")

def setup_audio_dataset(base_path: Path, zip_name: str):
    """Extracts and flattens audio files from a zip if the destination directory is missing or empty."""
    audio_dir = base_path / "extracted_audio"
    zip_path = base_path / zip_name

    # Check if the directory already exists and has files
    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"Directory '{audio_dir.name}' already exists and is not empty. Skipping extraction.")
    else:
        if not zip_path.exists():
            raise FileNotFoundError(f"Could not find zip file at {zip_path}")

        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, "r") as z:
            counts = {}
            # Filter for audio files, excluding metadata folders
            valid_files = [
                x for x in z.infolist() 
                if not x.is_dir() 
                and x.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))
                and "__MACOSX" not in x.filename
            ]

            for f in valid_files:
                name = Path(f.filename).name
                counts[name] = counts.get(name, 0) + 1
                
                # Handle potential filename collisions during flattening
                if counts[name] == 1:
                    out_name = name
                else:
                    stem = Path(name).stem
                    suffix = Path(name).suffix
                    out_name = f"{stem}_{counts[name]-1}{suffix}"
                
                with z.open(f) as src, open(audio_dir / out_name, "wb") as dst:
                    dst.write(src.read())

    # Final glob to return the list of files
    files = sorted(list(audio_dir.rglob("*.[wmop][afg][vc3]*")))
    print(f"Found {len(files)} files. Preview: {[f.name for f in files[:5]]}")
    return files

# Usage

def load_audio_as_mfcc(path: Path, n_mfcc: int = 40, sr: int = 22050, duration: float = 4.0):
    """
    Loads audio, pads/crops to a fixed duration, and returns an MFCC tensor.
    Result shape: (1, n_mfcc, time_steps)
    """
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
    x = torch.from_numpy(mfcc).float().unsqueeze(0)  # (1, n_mfcc, T)
    return x

def extract_patches_from_mfcc(mfcc: torch.Tensor, window_width: int, max_patches: int | None = None, device: torch.device = "cpu"):
    """
    mfcc: (1, n_mfcc, T) -> patches: (N, 1, n_mfcc, window_width).
    Treats the MFCC as an image and slices it along the time axis.
    """
    assert mfcc.ndim == 3  # (C, H, W)
    C, n_mfcc, T = mfcc.shape
    kernel_size = (n_mfcc, window_width)
    stride = (1, window_width // 2)  # 50% overlap
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
    patches = unfold(mfcc.unsqueeze(0)) # (1, 1, n_mfcc, T)
    patches = patches.transpose(1, 2).reshape(-1, C, n_mfcc, window_width) # Reshape to (N, C, H, W) -> (N, 1, n_mfcc, window_width)
    if max_patches is not None and patches.shape[0] > max_patches:
        idx = torch.randperm(patches.shape[0])[:max_patches]
        patches = patches[idx]
    return patches.to(device)

def extract_mel_and_mfcc(
    audio_path: str | Path,
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, np.ndarray | int]:
    """
    Load one audio file and extract mel-spectrogram + MFCC using file-native sample rate.

    Returns:
        dict with keys:
            - sr: sampling rate (int)
            - win_length: analysis window length in samples
            - hop_length: hop length in samples
            - mel_spectrogram: power mel spectrogram [n_mels, n_frames]
            - log_mel_db: log-mel spectrogram in dB [n_mels, n_frames]
            - mfcc: MFCC features [n_mfcc, n_frames]
    """
    y, sr = librosa.load(str(audio_path), sr=None)

    win_length = max(1, int(round(sr * (window_ms / 1000.0))))
    hop_length = max(1, int(round(sr * (hop_ms / 1000.0))))
    n_fft = win_length

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, win_length=win_length,
        hop_length=hop_length, n_mels=n_mels, center=True, power=2.0,
    )
    log_mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel_db, n_mfcc=n_mfcc)

    return {
        "sr": sr, "win_length": win_length, "hop_length": hop_length,
        "mel_spectrogram": mel_spectrogram, "log_mel_db": log_mel_db, "mfcc": mfcc,
    }

def extract_features_from_files(
    audio_paths: Iterable[str | Path],
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Extract mel-spectrogram and MFCC for multiple audio files."""
    results: dict[str, dict[str, np.ndarray | int]] = {}
    for path in audio_paths:
        results[str(path)] = extract_mel_and_mfcc(
            audio_path=path, n_mels=n_mels, n_mfcc=n_mfcc,
            window_ms=window_ms, hop_ms=hop_ms,
        )
    return results

def extract_time_patches(
    features: np.ndarray,
    sr: int,
    hop_length: int,
    patch_ms: float = 300.0,
    patch_hop_frames: int = 1,
) -> np.ndarray:
    """
    Extract sliding-window time patches from a feature matrix.

    How patch size is computed:
        patch_frames = round((patch_ms / 1000) * sr / hop_length)
        e.g. patch_ms=300, sr=22050, hop_length=220  -> patch_frames = round(30.0) = 30
        e.g. patch_ms=2,   sr=16000, hop_length=8    -> patch_frames = round(4.0)  = 4

    Args:
        features:         2D array [n_features, n_frames] (MFCC or mel-spectrogram).
        sr:               Sampling rate used to generate features (Hz).
        hop_length:       Hop size in samples used to generate features.
        patch_ms:         Duration of each patch in milliseconds.
                          Controls the temporal context a single patch sees.
        patch_hop_frames: Stride between consecutive patches (frames).
                          1 = maximum overlap (every frame starts a new patch);
                          larger values sub-sample at the cost of coverage.

    Returns:
        3D array [n_patches, n_features, patch_frames].
        Each patch covers patch_ms ms of audio.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D array with shape [n_features, n_frames].")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0.")
    if patch_hop_frames <= 0:
        raise ValueError("patch_hop_frames must be > 0.")

    n_features, n_frames = features.shape
    patch_frames = max(1, int(round((patch_ms / 1000.0) * sr / hop_length)))

    if n_frames < patch_frames:
        return np.empty((0, n_features, patch_frames), dtype=features.dtype)

    starts = range(0, n_frames - patch_frames + 1, patch_hop_frames)
    return np.stack([features[:, s : s + patch_frames] for s in starts], axis=0)

def zca_whitening_matrix(X):
    """
    Compute ZCA whitening matrix (aka Mahalanobis whitening).
    Input X: [M x N] — rows are variables, columns are observations.
    Returns ZCAMatrix: [M x M].
    """
    sigma = np.cov(X, rowvar=True)
    U, S, V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return ZCAMatrix

def preprocess_patches(patches: np.ndarray):
    """
    Preprocess patches for SMT pipeline:
      1. Flatten to 2D if needed.
      2. Per-feature standardisation (zero mean, unit variance across the dataset).
      3. ZCA whitening.
      4. L2 normalisation per patch.

    Returns:
        patches: preprocessed array (N, feature_dim)
        mean:    per-feature mean (feature_dim,) — keep for optional inverse transform
        zcaMatrix: whitening matrix (feature_dim, feature_dim)
    """
    # Flatten 3-D patches (N, F, T) -> (N, F*T)
    if len(patches.shape) >= 3:
        patches = patches.reshape(patches.shape[0], -1)

    # Per-feature (dataset-level) standardisation
    mean = np.mean(patches, axis=0)
    std  = np.std(patches, axis=0)
    patches = (patches - mean) / (std + 1e-6)

    # ZCA whitening
    zcaMatrix = zca_whitening_matrix(patches.T)
    patches = np.dot(zcaMatrix, patches.T).T

    # L2 normalise each patch vector
    patches = normalize(patches, norm='l2')

    return patches, mean, zcaMatrix

def apply_kmeans_to_patches(patches, n_clusters=100, sample_size=1000000):
    """
    Apply MiniBatchKMeans clustering to patches.

    Returns:
        kmeans: fitted KMeans object
    """
    if sample_size is not None and sample_size < len(patches):
        indices = np.random.choice(len(patches), sample_size, replace=False)
        patches = patches[indices]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=43, n_init=10)
    kmeans.fit(patches)
    return kmeans

def spectral_decomp(labels: np.ndarray, utterance_bounds: list, n_clusters: int):
    """
    Solve generalised eigenvalue decomposition: M u = lambda V u
    where M = A D D^T A^T  (slowness / temporal-penalty matrix)
    and   V = (1/N) A^T A + eps I  (whitening constraint).

    M is built by accumulating outer products of consecutive cluster-assignment
    differences within each utterance (no penalty at utterance boundaries).
    The eigenvectors with the SMALLEST eigenvalues are the slowest-varying
    directions in cluster space — the SMT manifold basis.

    Args:
        labels:           (N,) cluster assignment per frame.
        utterance_bounds: list of (start, end) frame-index pairs, one per utterance.
        n_clusters:       K — number of K-means clusters.

    Returns:
        eigvals:  (K,)   eigenvalues sorted ascending.
        eigvecs:  (K, K) corresponding eigenvectors (columns).
                  Caller should take eigvecs[:, 1:d+1].T to get the d-dim embedding P.
    """
    n_samples = labels.shape[0]

    A = csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (np.arange(n_samples), labels)),
        shape=(n_samples, n_clusters),
    )

    M = np.zeros((A.shape[1], A.shape[1]))
    for (start, end) in utterance_bounds:
        # Vectorised: convert the utterance slice to dense, diff across rows,
        # then accumulate with a single matmul.  Equivalent to summing the outer
        # products of consecutive cluster-assignment differences (same result as
        # the previous per-frame loop) but ~1000× faster and allocation-free.
        A_slice = A[start:end].toarray()    # (L, K) dense
        diffs   = np.diff(A_slice, axis=0)  # (L-1, K) frame-to-frame differences
        M      += diffs.T @ diffs           # (K, K)

    A = A.T

    n_samples = A.shape[1]
    eps = 1e-6
    V = (A @ A.T) / n_samples
    V += eps * np.eye(V.shape[0])

    eigvals, eigvecs = eigh(M, V)
    return eigvals, eigvecs

def plot_first_n_mel_patches(mel_patches, n_show=10, cols=5, cmap='magma'):
    """Plot the first n_show mel patches in a single figure grid."""
    import matplotlib.pyplot as plt

    mel_patches = np.asarray(mel_patches)
    if mel_patches.ndim != 3:
        raise ValueError('mel_patches must have shape [n_patches, n_mels, n_frames].')

    n_show = min(n_show, mel_patches.shape[0])
    if n_show == 0:
        raise ValueError('mel_patches is empty.')

    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 2.5 * rows),
        squeeze=False, constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for i in range(n_show):
        ax = axes_flat[i]
        im = ax.imshow(mel_patches[i], origin='lower', aspect='auto', cmap=cmap)
        ax.set_title(f'Patch {i}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mel bin')

    for j in range(n_show, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(f'First {n_show} Mel-Spectrogram Patches')
    plt.show()

def patch_multiple_utterances(mels: list[np.ndarray], sr: int, hop_length: int,
                               patch_ms: float = 300.0) -> tuple[np.ndarray, list]:
    """
    Extract time patches from multiple utterances and record per-utterance boundaries.

    The utterance_bounds list lets downstream functions (e.g. spectral_decomp) avoid
    penalising transitions that cross utterance boundaries, since the slow-feature
    criterion should only apply within a single continuous utterance.

    Patch size recap:
        patch_frames = round((patch_ms / 1000) * sr / hop_length)
        e.g. patch_ms=300, sr=22050, hop=220  -> 30 frames per patch (300 ms)
        e.g. patch_ms=500, sr=16000, hop=160  -> 50 frames per patch (500 ms)

    Args:
        mels:         List of (n_features, n_frames) feature matrices, one per utterance.
        sr:           Sampling rate (Hz).
        hop_length:   Hop size in samples.
        patch_ms:     Each patch duration in ms (default 300 ms).

    Returns:
        patches:          (N_total, n_features, patch_frames) array — all utterances stacked.
        utterance_bounds: List of (start, end) patch-index pairs, one per utterance.
    """
    patches = []
    utterance_bounds = []
    start = 0
    for mel in mels:
        p = extract_time_patches(
            features=mel,
            sr=sr,
            hop_length=hop_length,
            patch_ms=patch_ms,
            patch_hop_frames=1,
        )
        end = start + p.shape[0]
        patches.append(p)
        utterance_bounds.append((start, end))
        start = end
    patches = np.concatenate(patches)
    return patches, utterance_bounds

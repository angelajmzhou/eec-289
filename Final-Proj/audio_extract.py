import zipfile
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from Algos import reconstruct_audio

def setup_audio_dataset(base_path: Path, zip_name: str):
    """Unzip audio into extracted_audio/ if missing or empty."""
    audio_dir = base_path / "extracted_audio"
    zip_path = base_path / zip_name

    if audio_dir.exists() and any(audio_dir.iterdir()):
        print(f"Directory '{audio_dir.name}' already exists and is not empty. Skipping extraction.")
    else:
        if not zip_path.exists():
            raise FileNotFoundError(f"Could not find zip file at {zip_path}")

        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path}...")
        
        with zipfile.ZipFile(zip_path, "r") as z:
            counts = {}
            valid_files = [
                x for x in z.infolist() 
                if not x.is_dir() 
                and x.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))
                and "__MACOSX" not in x.filename
            ]

            for f in valid_files:
                name = Path(f.filename).name
                counts[name] = counts.get(name, 0) + 1
                if counts[name] == 1:
                    out_name = name
                else:
                    stem = Path(name).stem
                    suffix = Path(name).suffix
                    out_name = f"{stem}_{counts[name]-1}{suffix}"
                
                with z.open(f) as src, open(audio_dir / out_name, "wb") as dst:
                    dst.write(src.read())

    files = sorted(list(audio_dir.rglob("*.[wmop][afg][vc3]*")))
    print(f"Found {len(files)} files. Preview: {[f.name for f in files[:5]]}")
    return files

def load_audio_as_mfcc(path: Path, n_mfcc: int = 40, sr: int = 22050, duration: float = 4.0):
    """MFCC tensor (1, n_mfcc, T) fixed length."""
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
    """Slice (1, n_mfcc, T) into (N, 1, n_mfcc, window_width) patches."""
    assert mfcc.ndim == 3
    C, n_mfcc, T = mfcc.shape
    kernel_size = (n_mfcc, window_width)
    stride = (1, window_width // 2)
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
    patches = unfold(mfcc.unsqueeze(0))
    patches = patches.transpose(1, 2).reshape(-1, C, n_mfcc, window_width)
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
    """Mel + MFCC for one file at native sr; dict keys sr, win_length, hop_length, mel_spectrogram, log_mel_db, mfcc."""
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
    """Per-path dicts from extract_mel_and_mfcc."""
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
    """Stack patches from features (F, T); patch width from patch_ms, sr, hop_length."""
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
    """ZCA matrix for X with variables as rows."""
    sigma = np.cov(X, rowvar=True)
    U, S, V = np.linalg.svd(sigma)
    epsilon = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return ZCAMatrix

def preprocess_patches(patches: np.ndarray):
    """Standardise, ZCA-whiten, L2-normalise; returns (patches, mean, zcaMatrix)."""
    if len(patches.shape) >= 3:
        patches = patches.reshape(patches.shape[0], -1)

    mean = np.mean(patches, axis=0)
    std  = np.std(patches, axis=0)
    patches = (patches - mean) / (std + 1e-6)

    zcaMatrix = zca_whitening_matrix(patches.T)
    patches = np.dot(zcaMatrix, patches.T).T

    patches = normalize(patches, norm='l2')

    return patches, mean, zcaMatrix

def apply_kmeans_to_patches(patches, n_clusters=100, sample_size=1000000):
    """MiniBatchKMeans on patches (optional row subsample)."""
    if sample_size is not None and sample_size < len(patches):
        indices = np.random.choice(len(patches), sample_size, replace=False)
        patches = patches[indices]

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=43, n_init=10)
    kmeans.fit(patches)
    return kmeans

def spectral_decomp(labels: np.ndarray, utterance_bounds: list, n_clusters: int):
    """Generalised eigendecomp for slowness vs covariance; returns eigvals, eigvecs."""
    n_samples = labels.shape[0]

    A = csr_matrix(
        (np.ones(n_samples, dtype=np.float32), (np.arange(n_samples), labels)),
        shape=(n_samples, n_clusters),
    )

    M = np.zeros((A.shape[1], A.shape[1]))
    for (start, end) in utterance_bounds:
        A_slice = A[start:end].toarray()
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
    """Grid plot of first n_show mel patches."""
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
    """Concatenate per-utterance patches; utterance_bounds index ranges for spectral_decomp."""
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

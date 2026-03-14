"""
dataset.py — Data generation, preprocessing, and DataLoader creation.

Task: one-step-ahead sine wave prediction.
    Given a window of seq_len values, predict the next seq_len values (shifted by 1).

Why sine wave?
    - Zero download required.
    - GRU memory is genuinely exercised: predicting the next point requires
      remembering the current phase of the wave.
    - Results are visually interpretable — anyone can see if the prediction tracks.

Normalisation strategy:
    Mean and standard deviation are computed on the TRAINING split only,
    then applied to validation and test splits.  Fitting normalisation on
    the full dataset would leak future information into the training statistics.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_signal(n_points: int, noise_std: float = 0.05, seed: int = 42) -> np.ndarray:
    """
    Generate a clean, short-period sine wave with light Gaussian noise.

    Design rationale:
        The signal uses integer sample indices as the time axis with a period of
        exactly 30 steps.  With seq_len=50, each training window covers ~1.67 full
        cycles — enough for the GRU to observe a complete oscillation, learn its
        phase, and predict the continuation.

        Why period=30 matters:
            If the period is long relative to seq_len (e.g., period=1000, seq_len=50),
            each window contains only a near-linear slice of the curve.  The model
            then correctly learns a local trend rather than a periodic pattern —
            predictions look like a smoothed line, not a sine wave.  This is not a
            model failure; it is MSE regression finding the conditional mean of an
            under-constrained prediction problem.

            With period=30 and seq_len=50, the window unambiguously contains
            oscillations and the model is forced to learn phase tracking.

        Why noise_std=0.05:
            Heavy noise (≥0.1) pushes the oscillation amplitude below the noise
            floor in the high-frequency components.  MSE-trained models rationally
            smooth over unpredictable noise, producing blurry predictions.
            Light noise (0.05) keeps the signal-to-noise ratio high enough that
            tracking the oscillations is rewarded more than predicting the mean.

    Args:
        n_points  (int):   total number of samples to generate
        noise_std (float): standard deviation of additive Gaussian noise
        seed      (int):   numpy random seed for reproducibility

    Returns:
        signal (ndarray): shape (n_points,), dtype float32
    """
    rng = np.random.default_rng(seed)

    # Integer step indices — period is defined in steps, not in radians
    t = np.arange(n_points, dtype=np.float64)

    # Clean sine with period=30 steps + light noise
    PERIOD = 30.0
    signal = np.sin(2 * np.pi * t / PERIOD).astype(np.float32)
    signal += (noise_std * rng.standard_normal(n_points)).astype(np.float32)

    return signal


def build_windows(
    signal: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length seq_len over the signal to create (X, y) pairs.

    X[i] = signal[i : i + seq_len]        → input window
    y[i] = signal[i+1 : i + seq_len + 1]  → one-step-ahead targets

    Args:
        signal  (ndarray): 1-D array, shape (N,)
        seq_len (int):     window length

    Returns:
        X (ndarray): shape (n_windows, seq_len, 1)
        y (ndarray): shape (n_windows, seq_len, 1)
    """
    n_windows = len(signal) - seq_len
    X = np.stack([signal[i : i + seq_len]     for i in range(n_windows)])
    y = np.stack([signal[i + 1 : i + seq_len + 1] for i in range(n_windows)])

    # Add feature dimension: (n_windows, seq_len) → (n_windows, seq_len, 1)
    return X[..., np.newaxis], y[..., np.newaxis]


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Z-score normalise using statistics from the training set only.

    Args:
        train, val, test (ndarray): raw data arrays

    Returns:
        train_n, val_n, test_n: normalised arrays
        mean (float):  training mean (needed to denormalise predictions for plotting)
        std  (float):  training std
    """
    mean = float(train.mean())
    std  = float(train.std()) + 1e-8   # epsilon avoids division by zero

    train_n = (train - mean) / std
    val_n   = (val   - mean) / std
    test_n  = (test  - mean) / std

    return train_n, val_n, test_n, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    Build train / validation / test DataLoaders for the sine-wave prediction task.

    Pipeline:
        1. Generate noisy sine wave signal.
        2. Build sliding-window (X, y) pairs.
        3. Split into train / val / test by index (no shuffling before split to
           preserve temporal ordering).
        4. Normalise using training statistics only.
        5. Wrap each split in a TensorDataset and DataLoader.

    Args:
        cfg (Config): hyperparameter configuration object

    Returns:
        train_loader (DataLoader): shuffled
        val_loader   (DataLoader): not shuffled
        test_loader  (DataLoader): not shuffled
        norm_mean    (float):      mean used for normalisation (needed for denorm)
        norm_std     (float):      std  used for normalisation
    """
    total_points = cfg.N_SAMPLES + cfg.SEQ_LEN
    signal = generate_signal(n_points=total_points, seed=cfg.SEED)
    X, y   = build_windows(signal, cfg.SEQ_LEN)     # (N, seq_len, 1)

    n = len(X)
    n_train = int(n * cfg.TRAIN_SPLIT)
    n_val   = int(n * cfg.VAL_SPLIT)

    X_train, y_train = X[:n_train],                   y[:n_train]
    X_val,   y_val   = X[n_train : n_train + n_val],  y[n_train : n_train + n_val]
    X_test,  y_test  = X[n_train + n_val :],          y[n_train + n_val :]

    X_train, X_val, X_test, mean, std = normalise(X_train, X_val, X_test)
    # Normalise targets with the same statistics
    y_train = (y_train - mean) / std
    y_val   = (y_val   - mean) / std
    y_test  = (y_test  - mean) / std

    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32))

    def _make_loader(X_arr, y_arr, shuffle: bool) -> DataLoader:
        ds = TensorDataset(_to_tensor(X_arr), _to_tensor(y_arr))
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, drop_last=False)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = _make_loader(X_test,  y_test,  shuffle=False)

    print(
        f"Dataset ready  —  "
        f"train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}  "
        f"| norm mean={mean:.4f}  std={std:.4f}"
    )
    return train_loader, val_loader, test_loader, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Quick verification when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    train_loader, val_loader, test_loader, mean, std = create_dataloaders(cfg)

    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shapes  —  X: {tuple(X_batch.shape)},  y: {tuple(y_batch.shape)}")
    print(f"X range after norm: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
    print("dataset.py OK")

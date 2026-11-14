"""numpy twin of channel.py, used on the board and by the server-side ONNX tools.

Symbol tensors are float32 (B, C, h, w); adjacent channel pairs are the I/Q parts of one complex symbol.
This file must only depend on numpy.
"""

import numpy as np


def power_normalize(symbols: np.ndarray, power: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Scale each sample so its mean complex-symbol power equals `power`."""
    num_complex = symbols[0].size / 2
    energy = np.sum(symbols.astype(np.float64) ** 2, axis=(1, 2, 3), keepdims=True)
    return (symbols * np.sqrt(num_complex * power / np.maximum(energy, eps))).astype(np.float32)


def noise_std(snr_db: float | np.ndarray, power: float = 1.0) -> np.ndarray:
    """Per-real-component noise std for complex symbols of mean power `power`."""
    return np.sqrt(power / (2.0 * 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0)))


def transmit(
    symbols: np.ndarray, unit_noise: np.ndarray, snr_db: float | np.ndarray, active: int, channels: int
) -> np.ndarray:
    """Encoder output -> decoder input.

    Keep the first `active` channels, normalize power, add `unit_noise` (standard normal, at least `active`
    channels) scaled for `snr_db` (scalar or one value per sample), then zero-pad to `channels`.
    """
    sent = power_normalize(symbols[:, :active])
    std = noise_std(snr_db).reshape(-1, 1, 1, 1)
    received = np.zeros((symbols.shape[0], channels, *symbols.shape[2:]), dtype=np.float32)
    received[:, :active] = sent + std * unit_noise[:, :active]
    return received


def psnr(reference: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Per-image PSNR in dB for arrays in [0, 1], shape (B, 3, H, W)."""
    mse = np.mean((reference.astype(np.float64) - output.astype(np.float64)) ** 2, axis=(1, 2, 3))
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))

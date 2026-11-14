"""Everything between encoder and decoder. None of this is part of either model graph.

Symbol tensors are real, (B, C, h, w); adjacent channel pairs are the real/imaginary parts of one
complex symbol. bench/chain.py holds the numpy twin of these functions for the board.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def power_normalize(symbols: Tensor, power: float = 1.0, eps: float = 1e-12) -> Tensor:
    """Scale each sample so its mean complex-symbol power equals `power`."""
    num_complex = symbols[0].numel() / 2
    energy = symbols.pow(2).sum(dim=(1, 2, 3), keepdim=True)
    return symbols * torch.sqrt(num_complex * power / energy.clamp_min(eps))


def noise_std(snr_db: Tensor | float, power: float = 1.0) -> Tensor:
    """Per-real-component noise std for complex symbols of mean power `power` at `snr_db`."""
    snr_db = torch.as_tensor(snr_db, dtype=torch.float32)
    return torch.sqrt(power / (2.0 * 10.0 ** (snr_db / 10.0)))


def awgn(
    symbols: Tensor, snr_db: Tensor | float, power: float = 1.0, generator: torch.Generator | None = None
) -> Tensor:
    """Add complex AWGN. `snr_db` is a float or a (B, 1) tensor; `symbols` must have mean power `power`."""
    std = noise_std(snr_db, power).to(symbols.device).reshape(-1, 1, 1, 1)
    if generator is None:
        noise = torch.randn_like(symbols)
    else:
        noise = torch.randn(symbols.shape, generator=generator, device=generator.device).to(symbols.device)
    return symbols + std.to(symbols.dtype) * noise.to(symbols.dtype)


def pad_channels(symbols: Tensor, channels: int) -> Tensor:
    """Zero-pad truncated symbol channels back to the decoder's full channel count."""
    missing = channels - symbols.shape[1]
    if missing < 0:
        raise ValueError(f"Got {symbols.shape[1]} channels, more than the decoder's {channels}")
    return F.pad(symbols, (0, 0, 0, 0, 0, missing)) if missing else symbols


def pad_to_multiple(image: Tensor, multiple: int) -> Tensor:
    """Replicate-pad the bottom/right so H and W are multiples of `multiple`. Crop the output back."""
    height, width = image.shape[-2:]
    pad_h, pad_w = -height % multiple, -width % multiple
    return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else image

"""The simulated link: encoder -> truncate -> power normalize -> AWGN -> zero-pad -> decoder -> crop."""

import torch
from torch import Tensor, nn

from channel import awgn, pad_channels, pad_to_multiple, power_normalize


def run_link(
    model: nn.ModuleDict,
    images: Tensor,
    snr_db: Tensor,
    spp: float,
    enc_iters: int | None = None,
    dec_iters: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """`snr_db` is (B, 1). Images of any size are padded to the model stride and cropped back."""
    encoder, decoder = model["encoder"], model["decoder"]
    height, width = images.shape[-2:]
    symbols = encoder(pad_to_multiple(images, encoder.stride), snr_db, enc_iters, spp)
    transmitted = power_normalize(symbols.float())
    received = pad_channels(awgn(transmitted, snr_db, generator=generator), decoder.channels)
    x_hat = decoder(received, snr_db, dec_iters)[..., :height, :width]
    return {"x_hat": x_hat, "y": transmitted}

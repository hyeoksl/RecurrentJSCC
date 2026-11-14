"""DeepJSCC_lite baseline (kept for regression checks).

Layer layout and state-dict keys match the separ branch. Channel zero-padding and image
padding/cropping moved out of the modules so both models share one link (utils/link.py).
"""

import torch
from torch import Tensor, nn

STRIDE = 16


def conv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2)


def deconv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1
    )


def symbol_channels(spp: float, dim: int) -> int:
    channels = 2 * int(spp * STRIDE * STRIDE)
    if channels < 2 or dim <= 0 or dim % 2:
        raise ValueError("spp must allow at least one symbol and dim must be a positive even integer")
    return channels


class ResNetBlock(nn.Module):
    """Bottleneck residual block: 1x1 -> 3x3 -> 1x1."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, kernel_size=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.path(x) + self.skip(x)


class DeepJSCCEncoder(nn.Sequential):
    """image (B,3,H,W), H and W multiples of 16 -> real symbol channels (B,C,H/16,W/16)."""

    stride = STRIDE
    uses_snr = False
    iters_set: tuple[int, ...] = ()

    def __init__(self, spp: float = 0.25, dim: int = 384) -> None:
        channels = symbol_channels(spp, dim)
        super().__init__(
            conv(3, dim, 3),
            ResNetBlock(dim, dim),
            conv(dim, 3 * dim // 2, 3),
            ResNetBlock(3 * dim // 2, 3 * dim // 2),
            conv(3 * dim // 2, 2 * dim, 3),
            ResNetBlock(2 * dim, 2 * dim),
            conv(2 * dim, 3 * dim, 3),
            ResNetBlock(3 * dim, 3 * dim),
            nn.Conv2d(3 * dim, channels, 1),
        )
        self.max_spp = spp
        self.channels = channels

    def active_channels(self, spp: float) -> int:
        if not 0 < spp <= self.max_spp:
            raise ValueError(f"spp must be in (0, {self.max_spp}], got {spp}")
        return max(2, min(self.channels, 2 * round(spp / self.max_spp * (self.channels // 2))))

    def forward(
        self, image: Tensor, snr: Tensor | None = None, iters: int | None = None, spp: float | None = None
    ) -> Tensor:
        symbols = super().forward(image)
        return symbols if spp is None else symbols[:, : self.active_channels(spp)]


class DeepJSCCDecoder(nn.Sequential):
    """received (B,C,h,w) with all C channels (zero-padded outside) -> image (B,3,16h,16w) in [0, 1]."""

    stride = STRIDE
    uses_snr = False
    iters_set: tuple[int, ...] = ()

    def __init__(self, spp: float = 0.25, dim: int = 384) -> None:
        channels = symbol_channels(spp, dim)
        super().__init__(
            nn.Conv2d(channels, 3 * dim, 1),
            ResNetBlock(3 * dim, 3 * dim),
            deconv(3 * dim, 2 * dim, 3),
            ResNetBlock(2 * dim, 2 * dim),
            deconv(2 * dim, 3 * dim // 2, 3),
            ResNetBlock(3 * dim // 2, 3 * dim // 2),
            deconv(3 * dim // 2, dim, 3),
            ResNetBlock(dim, dim),
            deconv(dim, 3, 3),
        )
        self.max_spp = spp
        self.channels = channels

    def forward(self, received: Tensor, snr: Tensor | None = None, iters: int | None = None) -> Tensor:
        return torch.clamp(super().forward(received), 0, 1)

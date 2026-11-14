"""RecurrentJSCC: a weight-shared ConvNeXt core applied K times, with FiLM depth conditioning.

Power normalization, the channel, channel zero-padding and image padding/cropping live outside these
modules (see channel.py and utils/link.py).
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from model.blocks import AFModule, ConvNeXtBlock, FiLM, FilmParams

STRIDE = 8


def symbol_channels(spp: float, stride: int = STRIDE) -> int:
    """Real-valued symbol channels at the latent grid for `spp` complex symbols per pixel."""
    channels = 2 * round(spp * stride * stride)
    if channels < 2:
        raise ValueError(f"spp={spp} does not allow a single complex symbol at stride {stride}")
    return channels


class RecurrentCore(nn.Module):
    """`core_blocks` ConvNeXt blocks applied `iters` times.

    share_core=True: one set of blocks reused at every step (parameters independent of `iters`).
    share_core=False: `max(iters_set)` independent sets stacked; only that single depth can be run.
    """

    def __init__(
        self,
        dim: int,
        core_blocks: int,
        iters_set: Sequence[int],
        film: str,
        share_core: bool,
        norm: str,
        act: str,
        grn: bool,
    ) -> None:
        super().__init__()
        if not iters_set or min(iters_set) < 1:
            raise ValueError("model.iters must be a non-empty list of positive integers")
        if not share_core and len(set(iters_set)) != 1:
            raise ValueError("share_core=false requires a single depth in model.iters")
        self.k_max = max(iters_set)
        self.share_core = share_core
        num_sets = 1 if share_core else self.k_max
        norm_steps = self.k_max if share_core else 1
        self.steps = nn.ModuleList(
            nn.ModuleList(ConvNeXtBlock(dim, norm, act, grn, norm_steps) for _ in range(core_blocks))
            for _ in range(num_sets)
        )
        self.film = FiLM(dim, film, self.k_max)

    def film_table(self, iters: int) -> list[FilmParams]:
        """Constant (gamma, beta) per step, for export with a fixed K."""
        with torch.no_grad():
            return [self.film(t, iters) for t in range(iters)]

    def forward(self, x: Tensor, iters: int, film_table: Sequence[FilmParams] | None = None) -> Tensor:
        if not self.share_core and iters != len(self.steps):
            raise ValueError(f"Unshared core was built for {len(self.steps)} iterations, got {iters}")
        for t in range(iters):
            film = self.film(t, iters) if film_table is None else film_table[t]
            for block in self.steps[0 if self.share_core else t]:
                x = block(x, film, t if self.share_core else 0)
        return x


class RecurrentEncoder(nn.Module):
    """image (B,3,H,W), H and W multiples of 8 -> real symbol channels (B,C,H/8,W/8)."""

    stride = STRIDE
    uses_snr = True

    def __init__(
        self,
        spp: float,
        dim: int,
        stage_blocks: int,
        core_blocks: int,
        iters: Sequence[int],
        film: str,
        share_core: bool,
        norm: str,
        act: str,
        grn: bool,
    ) -> None:
        super().__init__()
        self.max_spp = spp
        self.channels = symbol_channels(spp)
        self.iters_set = tuple(iters)
        self.stem = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.stage = nn.ModuleList(ConvNeXtBlock(dim, norm, act, grn) for _ in range(stage_blocks))
        self.af1 = AFModule(dim)
        self.down = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2)
        self.core = RecurrentCore(2 * dim, core_blocks, iters, film, share_core, norm, act, grn)
        self.af2 = AFModule(2 * dim)
        self.head = nn.Conv2d(2 * dim, self.channels, kernel_size=1)

    def active_channels(self, spp: float) -> int:
        if not 0 < spp <= self.max_spp:
            raise ValueError(f"spp must be in (0, {self.max_spp}], got {spp}")
        return max(2, min(self.channels, symbol_channels(spp)))

    def forward(
        self,
        image: Tensor,
        snr: Tensor,
        iters: int,
        spp: float | None = None,
        film_table: Sequence[FilmParams] | None = None,
    ) -> Tensor:
        x = self.stem(image)
        for block in self.stage:
            x = block(x)
        x = self.down(self.af1(x, snr))
        x = self.core(x, iters, film_table)
        symbols = self.head(self.af2(x, snr))
        return symbols if spp is None else symbols[:, : self.active_channels(spp)]


class RecurrentDecoder(nn.Module):
    """received (B,C,h,w) with all C channels (zero-padded outside) -> image (B,3,8h,8w) in [0, 1]."""

    stride = STRIDE
    uses_snr = True

    def __init__(
        self,
        spp: float,
        dim: int,
        stage_blocks: int,
        core_blocks: int,
        iters: Sequence[int],
        film: str,
        share_core: bool,
        norm: str,
        act: str,
        grn: bool,
    ) -> None:
        super().__init__()
        self.max_spp = spp
        self.channels = symbol_channels(spp)
        self.iters_set = tuple(iters)
        self.head = nn.Conv2d(self.channels, 2 * dim, kernel_size=1)
        self.af1 = AFModule(2 * dim)
        self.core = RecurrentCore(2 * dim, core_blocks, iters, film, share_core, norm, act, grn)
        self.up = nn.ConvTranspose2d(2 * dim, dim, kernel_size=2, stride=2)
        self.af2 = AFModule(dim)
        self.stage = nn.ModuleList(ConvNeXtBlock(dim, norm, act, grn) for _ in range(stage_blocks))
        self.out = nn.ConvTranspose2d(dim, 3, kernel_size=4, stride=4)

    def forward(
        self,
        received: Tensor,
        snr: Tensor,
        iters: int,
        film_table: Sequence[FilmParams] | None = None,
    ) -> Tensor:
        x = self.af1(self.head(received), snr)
        x = self.core(x, iters, film_table)
        x = self.af2(self.up(x), snr)
        for block in self.stage:
            x = block(x)
        return self.out(x).clamp(0, 1)

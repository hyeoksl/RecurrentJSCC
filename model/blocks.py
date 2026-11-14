"""Building blocks shared by the models. Everything stays NCHW and Transpose-free."""

import math

import torch
from torch import Tensor, nn

NORMS = ("layernorm2d", "groupnorm1", "batchnorm")
ACTIVATIONS = ("gelu", "relu", "hardswish")
FILM_MODES = ("relative", "absolute", "none")

FilmParams = tuple[Tensor, Tensor] | None


def make_activation(kind: str) -> nn.Module:
    if kind == "gelu":
        return nn.GELU()
    if kind == "relu":
        return nn.ReLU()
    if kind == "hardswish":
        return nn.Hardswish()
    raise ValueError(f"Unknown activation '{kind}'; expected one of {ACTIVATIONS}")


class _AxisNorm(nn.Module):
    """Standardize over `axes` of an NCHW tensor, then a per-channel affine.

    Written with mean/sqrt so the export has no Transpose and no shape-dependent operators
    (nn.GroupNorm exports through Shape + Reshape + InstanceNormalization).
    """

    axes: tuple[int, ...]

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x - x.mean(dim=self.axes, keepdim=True)
        var = (x * x).mean(dim=self.axes, keepdim=True)
        return x / torch.sqrt(var + self.eps) * self.weight + self.bias


class LayerNorm2d(_AxisNorm):
    """LayerNorm over the channel axis at every position (ConvNeXt's norm, without the permute)."""

    axes = (1,)


class GroupNorm1(_AxisNorm):
    """GroupNorm with a single group: statistics over channels and positions."""

    axes = (1, 2, 3)


class StepNorm(nn.Module):
    """Normalization selectable by config.

    `batchnorm` keeps one BatchNorm per recurrence step, because a weight-shared core sees different
    activation statistics at every iteration. The other kinds are stateless and ignore `step`.
    """

    def __init__(self, kind: str, dim: int, num_steps: int = 1) -> None:
        super().__init__()
        if kind == "layernorm2d":
            norms = [LayerNorm2d(dim)]
        elif kind == "groupnorm1":
            norms = [GroupNorm1(dim)]
        elif kind == "batchnorm":
            norms = [nn.BatchNorm2d(dim) for _ in range(num_steps)]
        else:
            raise ValueError(f"Unknown norm '{kind}'; expected one of {NORMS}")
        self.kind = kind
        self.norms = nn.ModuleList(norms)

    def forward(self, x: Tensor, step: int = 0) -> Tensor:
        if len(self.norms) == 1:
            return self.norms[0](x)
        if step >= len(self.norms):
            raise ValueError(f"batchnorm has statistics for {len(self.norms)} steps, got step {step}")
        return self.norms[step](x)


class GRN(nn.Module):
    """Global response normalization (ConvNeXt V2) for NCHW tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # RMS instead of the L2 norm: the ratio below is the same, the range is friendlier to quantization.
        gx = torch.sqrt((x * x).mean(dim=(2, 3), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise conv -> norm -> [FiLM] -> 1x1 conv -> act -> [GRN] -> 1x1 conv, plus the residual."""

    def __init__(self, dim: int, norm: str, act: str, grn: bool, num_steps: int = 1, expansion: int = 4) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = StepNorm(norm, dim, num_steps)
        self.pwconv1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.act = make_activation(act)
        self.grn = GRN(expansion * dim) if grn else nn.Identity()
        self.pwconv2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)

    def forward(self, x: Tensor, film: FilmParams = None, step: int = 0) -> Tensor:
        h = self.norm(self.dwconv(x), step)
        if film is not None:
            gamma, beta = film
            h = gamma * h + beta
        h = self.pwconv2(self.grn(self.act(self.pwconv1(h))))
        return x + h


class AFModule(nn.Module):
    """SNR-adaptive channel attention. `snr` is a (B, 1) tensor in dB, so it is a graph input on export."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(dim // 8, 8)
        self.predictor = nn.Sequential(
            nn.Conv2d(dim + 1, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, snr: Tensor) -> Tensor:
        pooled = x.mean(dim=(2, 3), keepdim=True)
        context = torch.cat((snr.reshape(-1, 1, 1, 1), pooled), dim=1)
        return x * self.predictor(context)


class FiLM(nn.Module):
    """Depth conditioning: one (gamma, beta) pair per recurrence step t of a K-step unroll.

    relative: s = t / K.   absolute: s = t / k_max (independent of K).   none: no conditioning.
    The MLP is shared across steps and its last layer starts at zero, so gamma = 1 and beta = 0 initially.
    """

    def __init__(self, dim: int, mode: str, k_max: int, embed_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        if mode not in FILM_MODES:
            raise ValueError(f"Unknown FiLM mode '{mode}'; expected one of {FILM_MODES}")
        if embed_dim % 2:
            raise ValueError("embed_dim must be even")
        self.mode = mode
        self.k_max = k_max
        self.dim = dim
        if mode == "none":
            return
        half = embed_dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (math.log(100.0) / max(half - 1, 1)))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def position(self, t: int, iters: int) -> float:
        return t / iters if self.mode == "relative" else t / self.k_max

    def forward(self, t: int, iters: int) -> FilmParams:
        if self.mode == "none":
            return None
        phase = self.freqs * self.position(t, iters)
        embedding = torch.cat((torch.sin(phase), torch.cos(phase))).unsqueeze(0)
        delta_gamma, beta = self.mlp(embedding).chunk(2, dim=1)
        return (1.0 + delta_gamma).reshape(1, self.dim, 1, 1), beta.reshape(1, self.dim, 1, 1)

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import build_model  # noqa: E402

ITERS = [1, 2, 4, 6]


def model_cfg(**overrides):
    cfg = {
        "type": "RecurrentJSCC",
        "spp": 0.5,
        "min_spp": 0.1,
        "min_SNRdB": 0,
        "max_SNRdB": 20,
        "dim": 16,
        "stage_blocks": 3,
        "core_blocks": 3,
        "iters": ITERS,
        "film": "relative",
        "share_core": True,
        "norm": "layernorm2d",
        "act": "gelu",
        "grn": False,
    }
    cfg.update(overrides)
    return cfg


def build(seed=0, **overrides):
    torch.manual_seed(seed)
    return build_model(model_cfg(**overrides))


def randomize_film(model, seed=1):
    """Give the zero-initialised FiLM output layers non-trivial weights."""
    generator = torch.Generator().manual_seed(seed)
    for part in model.values():
        last = part.core.film.mlp[-1]
        with torch.no_grad():
            last.weight.copy_(torch.randn(last.weight.shape, generator=generator) * 0.1)
            last.bias.copy_(torch.randn(last.bias.shape, generator=generator) * 0.1)


@pytest.fixture
def snr():
    return torch.full((2, 1), 10.0)

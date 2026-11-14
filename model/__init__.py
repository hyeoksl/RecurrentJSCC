"""Model registry. A model is an nn.ModuleDict({'encoder', 'decoder'}); the channel belongs to neither."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonargparse import ArgumentParser
from torch import nn

from model.blocks import ACTIVATIONS, FILM_MODES, NORMS
from model.deep_jscc import DeepJSCCDecoder, DeepJSCCEncoder
from model.recurrent_jscc import RecurrentDecoder, RecurrentEncoder


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[Mapping[str, Any]], nn.ModuleDict]
    add_arguments: Callable[[ArgumentParser], None]


def _add_common_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--model.type", type=str)
    parser.add_argument("--model.min_SNRdB", type=float)
    parser.add_argument("--model.max_SNRdB", type=float)
    parser.add_argument("--model.spp", type=float, help="maximum symbols per pixel")
    parser.add_argument("--model.min_spp", type=float)
    parser.add_argument("--model.dim", type=int)


def _build_deep_jscc(cfg: Mapping[str, Any]) -> nn.ModuleDict:
    kwargs = {"spp": cfg["spp"], "dim": cfg["dim"]}
    return nn.ModuleDict({"encoder": DeepJSCCEncoder(**kwargs), "decoder": DeepJSCCDecoder(**kwargs)})


def _add_recurrent_arguments(parser: ArgumentParser) -> None:
    _add_common_arguments(parser)
    parser.add_argument("--model.stage_blocks", type=int, default=3, help="unshared blocks per stage (N1)")
    parser.add_argument("--model.core_blocks", type=int, default=3)
    parser.add_argument("--model.iters", type=list[int], default=[1, 2, 4, 6])
    parser.add_argument("--model.film", type=str, choices=FILM_MODES, default="relative")
    parser.add_argument("--model.share_core", type=bool, default=True)
    parser.add_argument("--model.norm", type=str, choices=NORMS, default="layernorm2d")
    parser.add_argument("--model.act", type=str, choices=ACTIVATIONS, default="gelu")
    parser.add_argument("--model.grn", type=bool, default=False)


def _build_recurrent(cfg: Mapping[str, Any]) -> nn.ModuleDict:
    keys = ("spp", "dim", "stage_blocks", "core_blocks", "iters", "film", "share_core", "norm", "act", "grn")
    kwargs = {key: cfg[key] for key in keys}
    return nn.ModuleDict({"encoder": RecurrentEncoder(**kwargs), "decoder": RecurrentDecoder(**kwargs)})


MODELS: dict[str, ModelSpec] = {
    "DeepJSCC_lite": ModelSpec(_build_deep_jscc, _add_common_arguments),
    "RecurrentJSCC": ModelSpec(_build_recurrent, _add_recurrent_arguments),
}


def get_spec(model_type: str) -> ModelSpec:
    if model_type not in MODELS:
        raise KeyError(f"Unknown model.type '{model_type}'; registered: {sorted(MODELS)}")
    return MODELS[model_type]


def build_model(cfg: Mapping[str, Any]) -> nn.ModuleDict:
    """Build the encoder/decoder pair from the `model` section of a config."""
    return get_spec(cfg["type"]).build(cfg)


__all__ = ["MODELS", "ModelSpec", "build_model", "get_spec"]

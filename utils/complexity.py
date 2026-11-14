"""Parameter and FLOP counting.

FLOPs here are multiply-accumulates of Conv2d / ConvTranspose2d / Linear layers (the fvcore/thop
convention), without bias, normalization, activations or the FiLM affine. Hooks fire on every call,
so a weight-shared core executed K times is counted K times while its parameters are counted once.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

FLOPS_CONVENTION = "multiply-accumulates of Conv2d/ConvTranspose2d/Linear per image; every executed iteration counted"


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _layer_flops(layer: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> int:
    if isinstance(layer, nn.Conv2d):
        kernel = layer.kernel_size[0] * layer.kernel_size[1]
        return output.numel() * (layer.in_channels // layer.groups) * kernel
    if isinstance(layer, nn.ConvTranspose2d):
        kernel = layer.kernel_size[0] * layer.kernel_size[1]
        return inputs[0].numel() * (layer.out_channels // layer.groups) * kernel
    if isinstance(layer, nn.Linear):
        return output.numel() * layer.in_features
    return 0


def count_flops(module: nn.Module, run: Callable[[], Any]) -> int:
    """FLOPs of the layers inside `module` executed while calling `run()`."""
    total = 0

    def hook(layer: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total
        total += _layer_flops(layer, inputs, output)

    counted = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    handles = [m.register_forward_hook(hook) for m in module.modules() if isinstance(m, counted)]
    try:
        with torch.no_grad():
            run()
    finally:
        for handle in handles:
            handle.remove()
    return total


def link_complexity(
    model: nn.ModuleDict, height: int, width: int, enc_iters: int | None, dec_iters: int | None
) -> dict[str, float | int]:
    """Parameters and GFLOPs of encoder and decoder for one (enc_iters, dec_iters) pair at batch size 1."""
    encoder, decoder = model["encoder"], model["decoder"]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    image = torch.zeros(1, 3, height, width, device=device)
    snr = torch.zeros(1, 1, device=device)
    with torch.no_grad():
        symbols = encoder(image, snr, enc_iters)
    enc_flops = count_flops(encoder, lambda: encoder(image, snr, enc_iters))
    dec_flops = count_flops(decoder, lambda: decoder(symbols, snr, dec_iters))
    model.train(was_training)
    return {
        "encoder_params": count_parameters(encoder),
        "decoder_params": count_parameters(decoder),
        "encoder_gflops": enc_flops / 1e9,
        "decoder_gflops": dec_flops / 1e9,
    }

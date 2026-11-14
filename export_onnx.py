"""Export encoder and decoder as separate ONNX models, one file per recurrence count K.

Dynamic mode (default, for PC inference): batch, height and width are dynamic axes.
Static mode (--static H W, for the NPU): every shape is fixed; the only inputs are float tensors
`image (1,3,H,W)` / `received (1,C,H/s,W/s)` and `snr (1,1)`. With K fixed, FiLM's gamma/beta are constants.

Truncation to the operating spp, power normalization, the channel, channel zero-padding and image
padding/cropping are not part of either graph.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torch import Tensor, nn

from model import build_model
from utils.logging import find_best_ckpt, git_revision, load_config, load_weights
from utils.parsing import parse_config

OPSET = 17
TOLERANCE = 1e-4
# Operators whose output depends on runtime shapes or data-dependent control flow.
DYNAMIC_OPS = {"Shape", "Size", "NonZero", "Loop", "If", "Scan", "Range", "ConstantOfShape", "Expand", "Where"}

logger = logging.getLogger("export")


class EncoderExport(nn.Module):
    def __init__(self, encoder: nn.Module, iters: int | None) -> None:
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.film_table = encoder.core.film_table(iters) if iters is not None else None

    def forward(self, image: Tensor, snr: Tensor | None = None) -> Tensor:
        if self.iters is None:
            return self.encoder(image)
        return self.encoder(image, snr, self.iters, film_table=self.film_table)


class DecoderExport(nn.Module):
    def __init__(self, decoder: nn.Module, iters: int | None) -> None:
        super().__init__()
        self.decoder = decoder
        self.iters = iters
        self.film_table = decoder.core.film_table(iters) if iters is not None else None

    def forward(self, received: Tensor, snr: Tensor | None = None) -> Tensor:
        if self.iters is None:
            return self.decoder(received)
        return self.decoder(received, snr, self.iters, film_table=self.film_table)


def check_static_graph(path: Path) -> None:
    """Raise unless the graph has only fixed-shape float inputs and no shape-dependent operators."""
    graph = onnx.load(str(path)).graph
    initializers = {init.name for init in graph.initializer}
    for value in [v for v in graph.input if v.name not in initializers] + list(graph.output):
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{path}: '{value.name}' is not float32")
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in tensor.shape.dim):
            raise ValueError(f"{path}: '{value.name}' has a dynamic dimension")
    dynamic = sorted({node.op_type for node in graph.node} & DYNAMIC_OPS)
    if dynamic:
        raise ValueError(f"{path}: shape-dependent operators in a static graph: {dynamic}")


def max_abs_diff(path: Path, inputs: dict[str, Tensor], expected: Tensor) -> float:
    """Run the ONNX file on onnxruntime's CPU EP and compare with the PyTorch output."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {name: value.numpy() for name, value in inputs.items()})[0]
    return float(np.abs(output - expected.numpy()).max())


def export_module(
    module: nn.Module, role: str, iters: int | None, path: Path, static_hw: tuple[int, int] | None
) -> dict[str, Any]:
    """Export one encoder or decoder. Returns its metadata, including the ONNX-vs-PyTorch difference."""
    if role not in ("encoder", "decoder"):
        raise ValueError(f"Unknown role '{role}'")
    module = module.eval()
    height, width = static_hw or (256, 256)
    stride = module.stride
    if height % stride or width % stride:
        raise ValueError(f"Static size {height}x{width} is not a multiple of the model stride {stride}")
    generator = torch.Generator().manual_seed(0)
    if role == "encoder":
        wrapper: nn.Module = EncoderExport(module, iters)
        data_name, out_name = "image", "symbols"
        data = torch.rand(1, 3, height, width, generator=generator)
        axes = {"image": {0: "batch", 2: "height", 3: "width"}, "symbols": {0: "batch", 2: "sym_h", 3: "sym_w"}}
    else:
        wrapper = DecoderExport(module, iters)
        data_name, out_name = "received", "reconstruction"
        data = torch.randn(1, module.channels, height // stride, width // stride, generator=generator)
        axes = {
            "received": {0: "batch", 2: "sym_h", 3: "sym_w"},
            "reconstruction": {0: "batch", 2: "height", 3: "width"},
        }
    inputs = {data_name: data}
    if module.uses_snr:
        inputs["snr"] = torch.full((1, 1), 10.0)
        axes["snr"] = {0: "batch"}

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = wrapper(*inputs.values())
        torch.onnx.export(
            wrapper,
            tuple(inputs.values()),
            str(path),
            input_names=list(inputs),
            output_names=[out_name],
            dynamic_axes=None if static_hw else axes,
            opset_version=OPSET,
            dynamo=False,
        )
    if static_hw:
        check_static_graph(path)
    diff = max_abs_diff(path, inputs, expected)
    if diff > TOLERANCE:
        raise RuntimeError(f"{path}: onnxruntime output differs from PyTorch by {diff:.3g} (> {TOLERANCE})")
    return {
        "role": role,
        "iters": iters,
        "path": path.name,
        "mode": "static" if static_hw else "dynamic",
        "inputs": {name: list(value.shape) for name, value in inputs.items()},
        "output": {out_name: list(expected.shape)},
        "max_abs_diff_vs_torch": diff,
        "bytes": path.stat().st_size,
    }


def set_override(config: dict[str, Any], assignment: str) -> None:
    """Apply `a.b.c=value` (value parsed as YAML) to a nested config dict."""
    key, sep, value = assignment.partition("=")
    if not sep:
        raise ValueError(f"Override '{assignment}' is not of the form key=value")
    *parents, leaf = key.split(".")
    node = config
    for parent in parents:
        node = node[parent]
    if leaf not in node:
        raise KeyError(f"Override key '{key}' does not exist in the config")
    node[leaf] = yaml.safe_load(value)


def load_source(source: Path, random_init: bool, overrides: list[str], seed: int) -> tuple[dict, nn.ModuleDict, str]:
    """Config + model from an experiment directory (best checkpoint) or, with random_init, a YAML config."""
    if source.is_dir():
        config = load_config(source)
    elif random_init:
        config = parse_config(["--config_file", str(source)])
    else:
        raise ValueError(f"{source} is not an experiment directory; a config file needs --random-init")
    for assignment in overrides:
        set_override(config, assignment)
    torch.manual_seed(seed)
    model = build_model(config["model"]).eval()
    if random_init:
        return config, model, f"random-init(seed={seed})"
    checkpoint = find_best_ckpt(source)
    load_weights(model, checkpoint)
    return config, model, str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="logs/<exp> directory, or a config YAML with --random-init")
    parser.add_argument("--enc-iters", type=int, nargs="+", help="encoder K values (default: model.iters)")
    parser.add_argument("--dec-iters", type=int, nargs="+", help="decoder K values (default: model.iters)")
    parser.add_argument("--static", type=int, nargs=2, metavar=("H", "W"), help="fixed input size (NPU export)")
    parser.add_argument("--random-init", action="store_true", help="skip the checkpoint (toolchain checks only)")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE", help="e.g. model.norm=batchnorm"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for --random-init weights")
    parser.add_argument("--name", help="artifact name (default: experiment name)")
    parser.add_argument("--out", type=Path, help="output directory (default: artifacts/<name>/onnx)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

    config, model, weights = load_source(args.source.expanduser(), args.random_init, args.override, args.seed)
    name = args.name or config["experiment_name"]
    out_dir = args.out or Path("artifacts") / name / "onnx"
    static_hw = tuple(args.static) if args.static else None
    default_iters = config["model"].get("iters") or [None]
    suffix = "" if static_hw else "_dyn"

    exported = []
    for role, iters_list in (("encoder", args.enc_iters), ("decoder", args.dec_iters)):
        for iters in iters_list or default_iters:
            stem = role if iters is None else f"{role}_k{iters}"
            info = export_module(model[role], role, iters, out_dir / f"{stem}{suffix}.onnx", static_hw)
            logger.info(
                "Exported %s (max |onnx - torch| = %.3g)", out_dir / info["path"], info["max_abs_diff_vs_torch"]
            )
            exported.append(info)

    meta_path = out_dir / ("export.json" if static_hw else "export_dyn.json")
    meta = {
        "name": name,
        "weights": weights,
        "random_init": args.random_init,
        "overrides": args.override,
        "git": git_revision(),
        "opset": OPSET,
        "onnxruntime": ort.__version__,
        "static_hw": list(static_hw) if static_hw else None,
        "stride": model["encoder"].stride,
        "symbol_channels": model["encoder"].channels,
        "config": config,
        "models": exported,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s", meta_path)


if __name__ == "__main__":
    main()

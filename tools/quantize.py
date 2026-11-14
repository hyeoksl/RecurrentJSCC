"""Quantize static FP32 ONNX exports to QDQ models for the QNN execution provider.

Uses onnxruntime.quantization's QNN helpers (qnn_preprocess_model + get_qnn_qdq_config) with 8-bit
weights and 8-bit or 16-bit activations. Every output is loaded and run on this machine's onnxruntime
CPU EP, which is pinned to the board's version, so a model that passes here is readable on the board.

Also writes results/npu/<name>/onnx_stats.json: file and initializer sizes per K, duplicated
initializers inside a model, and how many initializer bytes different K models have in common.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

logger = logging.getLogger("quantize")
ACTIVATION_TYPES = {8: QuantType.QUInt8, 16: QuantType.QUInt16}


class NpyReader(CalibrationDataReader):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.count = len(next(iter(arrays.values())))
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= self.count:
            return None
        sample = {name: array[self.index : self.index + 1] for name, array in self.arrays.items()}
        self.index += 1
        return sample

    def rewind(self) -> None:
        self.index = 0


def calibration_arrays(artifact: Path, model_meta: dict[str, Any]) -> dict[str, np.ndarray]:
    files = {"image": "image.npy", "received": "received.npy", "snr": "snr.npy"}
    return {name: np.load(artifact / "calib" / files[name]) for name in model_meta["inputs"]}


def run_on_cpu(path: Path, arrays: dict[str, np.ndarray]) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {name: array[:1] for name, array in arrays.items()})[0]


def initializer_hashes(path: Path) -> dict[str, tuple[str, int]]:
    """name -> (content hash, bytes) for every initializer."""
    hashes = {}
    for init in onnx.load(str(path)).graph.initializer:
        raw = onnx.numpy_helper.to_array(init).tobytes()
        digest = hashlib.sha256(str(init.data_type).encode() + str(list(init.dims)).encode() + raw).hexdigest()
        hashes[init.name] = (digest, len(raw))
    return hashes


def model_stats(path: Path) -> dict[str, Any]:
    hashes = initializer_hashes(path)
    unique = {digest: size for digest, size in hashes.values()}
    graph = onnx.load(str(path)).graph
    uses = Counter(name for node in graph.node for name in node.input if name in hashes)
    return {
        "file_bytes": path.stat().st_size,
        "num_nodes": len(graph.node),
        "ops": dict(sorted(Counter(node.op_type for node in graph.node).items())),
        "num_initializers": len(hashes),
        "initializer_bytes": sum(size for _, size in hashes.values()),
        "unique_initializer_bytes": sum(unique.values()),
        "duplicated_initializer_bytes": sum(size for _, size in hashes.values()) - sum(unique.values()),
        "initializers_used_by_several_nodes": sum(1 for count in uses.values() if count > 1),
        "max_uses_of_one_initializer": max(uses.values(), default=0),
    }


def cross_k_sharing(paths: dict[int | None, Path]) -> dict[str, Any]:
    """Initializer bytes (by content) common to all K models of one role and precision."""
    per_k = {k: {digest: size for digest, size in initializer_hashes(path).values()} for k, path in paths.items()}
    common = set.intersection(*(set(h) for h in per_k.values()))
    reference = next(iter(per_k.values()))
    union: dict[str, int] = {}
    for hashes in per_k.values():
        union.update(hashes)
    return {
        "k_values": list(per_k),
        "bytes_common_to_all_k": sum(reference[d] for d in common),
        "unique_bytes_per_k": {str(k): sum(h.values()) for k, h in per_k.items()},
        "bytes_if_stored_separately": sum(sum(h.values()) for h in per_k.values()),
        "bytes_if_deduplicated_across_k": sum(union.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", type=Path, help="artifacts/<name> with onnx/ and calib/")
    parser.add_argument("--activation-bits", type=int, nargs="+", default=[8, 16], choices=sorted(ACTIVATION_TYPES))
    parser.add_argument("--per-channel", action="store_true", help="per-channel weights for Conv/ConvTranspose")
    parser.add_argument("--results", type=Path, help="default: results/npu/<name>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

    artifact = args.artifact
    meta = json.loads((artifact / "onnx" / "export.json").read_text())
    pre_dir = artifact / "qdq" / "pre"
    pre_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"name": meta["name"], "onnxruntime": ort.__version__, "per_channel": args.per_channel}
    stats["models"] = {}
    by_group: dict[tuple[str, str], dict[int | None, Path]] = {}
    quantized_meta = []

    for model_meta in meta["models"]:
        source = artifact / "onnx" / model_meta["path"]
        arrays = calibration_arrays(artifact, model_meta)
        preprocessed = pre_dir / model_meta["path"]
        changed = qnn_preprocess_model(source, preprocessed)
        float_model = preprocessed if changed else source
        reference = run_on_cpu(source, arrays)
        stats["models"][f"fp32/{model_meta['path']}"] = model_stats(source)
        by_group.setdefault(("fp32", model_meta["role"]), {})[model_meta["iters"]] = source

        for bits in args.activation_bits:
            precision = f"a{bits}"
            target = artifact / "qdq" / precision / model_meta["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            config = get_qnn_qdq_config(
                float_model,
                NpyReader(arrays),
                activation_type=ACTIVATION_TYPES[bits],
                weight_type=QuantType.QUInt8,
                per_channel=args.per_channel,
            )
            quantize(float_model, target, config)
            output = run_on_cpu(target, arrays)
            error = float(np.sqrt(np.mean((output - reference) ** 2)))
            logger.info("%s/%s: rms error vs fp32 on one calibration sample = %.4g", precision, target.name, error)
            stats["models"][f"{precision}/{model_meta['path']}"] = {
                **model_stats(target),
                "rms_error_vs_fp32_one_sample": error,
            }
            by_group.setdefault((precision, model_meta["role"]), {})[model_meta["iters"]] = target
            quantized_meta.append({**model_meta, "precision": precision, "preprocess_changed_graph": bool(changed)})

    stats["sharing_across_k"] = {
        f"{precision}/{role}": cross_k_sharing(paths) for (precision, role), paths in by_group.items() if len(paths) > 1
    }
    (artifact / "qdq" / "quantize.json").write_text(
        json.dumps({"per_channel": args.per_channel, "weight_type": "QUInt8", "models": quantized_meta}, indent=2)
    )
    results = args.results or Path("results/npu") / meta["name"]
    results.mkdir(parents=True, exist_ok=True)
    (results / "onnx_stats.json").write_text(json.dumps(stats, indent=1))
    logger.info("Wrote %s", results / "onnx_stats.json")


if __name__ == "__main__":
    main()

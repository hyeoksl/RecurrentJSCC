"""M0a: a few-conv dummy model to validate the toolchain end to end, independent of RecurrentJSCC.

Writes artifacts/m0a_dummy/ in the same layout as a real export (onnx/, calib/), so tools/quantize.py
and tools/make_bundle.py work on it unchanged. Delete this file and the artifact once M0a passes.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export_onnx import OPSET, check_static_graph, max_abs_diff  # noqa: E402

OUT = Path("artifacts/m0a_dummy")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 8, 1),
    ).eval()  # fmt: skip
    image = torch.rand(1, 3, 64, 64)
    path = OUT / "onnx" / "dummy.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        expected = model(image)
        torch.onnx.export(model, (image,), str(path), input_names=["image"], output_names=["features"],
                          opset_version=OPSET, dynamo=False)  # fmt: skip
    check_static_graph(path)
    diff = max_abs_diff(path, {"image": image}, expected)
    meta = {
        "name": "m0a_dummy", "weights": "random-init(seed=0)", "random_init": True, "overrides": [], "git": None,
        "static_hw": [64, 64], "config": {},
        "models": [{"role": "single", "iters": None, "path": "dummy.onnx", "mode": "static",
                    "inputs": {"image": [1, 3, 64, 64]}, "max_abs_diff_vs_torch": diff}],
    }  # fmt: skip
    (OUT / "onnx" / "export.json").write_text(json.dumps(meta, indent=2))
    (OUT / "calib").mkdir(exist_ok=True)
    np.save(OUT / "calib" / "image.npy", np.random.default_rng(0).random((32, 3, 64, 64), dtype=np.float32))
    logging.info("Wrote %s (max |onnx - torch| = %.3g)", path, diff)


if __name__ == "__main__":
    main()

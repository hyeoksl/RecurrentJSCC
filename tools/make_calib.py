"""Create calibration and benchmark inputs for a static export (artifacts/<name>/onnx).

calib/  image.npy (N,3,H,W), snr.npy (N,1), received.npy (N,C,h,w)   -> tools/quantize.py
bench/  images.npy (M,3,H,W), noise.npy (M,C,h,w), bench.json        -> board PSNR chain

Decoder calibration inputs are FP32 encoder outputs (cycling over the exported encoder K values) pushed
through the same numpy channel the board uses: truncate to a random spp, normalize, AWGN, zero-pad.
Benchmark noise is unit Gaussian generated here, so server and board see identical channel noise.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.chain import transmit  # noqa: E402
from utils.dataset import list_images  # noqa: E402

logger = logging.getLogger("make_calib")


def load_crop(path: Path, height: int, width: int, rng: np.random.Generator | None) -> np.ndarray:
    """Random crop if `rng` is given, else center crop. Images smaller than the crop are edge-padded."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pad_h, pad_w = max(height - pixels.shape[0], 0), max(width - pixels.shape[1], 0)
    if pad_h or pad_w:
        pixels = np.pad(pixels, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    free_h, free_w = pixels.shape[0] - height, pixels.shape[1] - width
    top = int(rng.integers(0, free_h + 1)) if rng is not None else free_h // 2
    left = int(rng.integers(0, free_w + 1)) if rng is not None else free_w // 2
    return pixels[top : top + height, left : left + width].transpose(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", type=Path, help="artifacts/<name> containing onnx/export.json")
    parser.add_argument("--calib-dataset", default="~/datasets/DIV2K/train")
    parser.add_argument("--bench-dataset", default="~/datasets/Kodak")
    parser.add_argument("--num-calib", type=int, default=64)
    parser.add_argument("--num-bench", type=int, default=24)
    parser.add_argument("--bench-snr", type=float, help="default: midpoint of the training SNR range")
    parser.add_argument("--bench-spp", type=float, help="default: model.spp (all channels)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

    meta = json.loads((args.artifact / "onnx" / "export.json").read_text())
    height, width = meta["static_hw"]
    cfg = meta["config"]["model"]
    channels, stride = meta["symbol_channels"], meta["stride"]
    rng = np.random.default_rng(args.seed)

    def active_channels(spp: float) -> int:
        return max(2, min(channels, 2 * round(spp * stride * stride)))

    calib_paths = list_images(args.calib_dataset)
    picks = rng.choice(len(calib_paths), size=args.num_calib, replace=len(calib_paths) < args.num_calib)
    images = np.stack([load_crop(calib_paths[i], height, width, rng) for i in picks])
    snr = rng.uniform(cfg["min_SNRdB"], cfg["max_SNRdB"], size=(args.num_calib, 1)).astype(np.float32)

    encoders = [m for m in meta["models"] if m["role"] == "encoder"]
    sessions = [
        ort.InferenceSession(str(args.artifact / "onnx" / m["path"]), providers=["CPUExecutionProvider"])
        for m in encoders
    ]
    received = np.zeros((args.num_calib, channels, height // stride, width // stride), dtype=np.float32)
    for index in range(args.num_calib):
        session = sessions[index % len(sessions)]
        feeds = {"image": images[index : index + 1]}
        if "snr" in meta["models"][0]["inputs"]:
            feeds["snr"] = snr[index : index + 1]
        symbols = session.run(None, feeds)[0]
        noise = rng.standard_normal(symbols.shape).astype(np.float32)
        active = active_channels(rng.uniform(cfg["min_spp"], cfg["spp"]))
        received[index] = transmit(symbols, noise, float(snr[index, 0]), active, channels)[0]

    calib_dir = args.artifact / "calib"
    calib_dir.mkdir(exist_ok=True)
    np.save(calib_dir / "image.npy", images)
    np.save(calib_dir / "snr.npy", snr)
    np.save(calib_dir / "received.npy", received)

    bench_paths = list_images(args.bench_dataset)[: args.num_bench]
    bench_images = np.stack([load_crop(path, height, width, None) for path in bench_paths])
    bench_noise = rng.standard_normal((len(bench_paths), channels, height // stride, width // stride))
    bench_snr = args.bench_snr if args.bench_snr is not None else (cfg["min_SNRdB"] + cfg["max_SNRdB"]) / 2
    bench_dir = args.artifact / "bench"
    bench_dir.mkdir(exist_ok=True)
    np.save(bench_dir / "images.npy", bench_images)
    np.save(bench_dir / "noise.npy", bench_noise.astype(np.float32))
    bench_meta = {
        "snr_db": float(bench_snr),
        "spp": args.bench_spp or cfg["spp"],
        "active_channels": active_channels(args.bench_spp or cfg["spp"]),
        "symbol_channels": channels,
        "uses_snr": "snr" in meta["models"][0]["inputs"],
        "images": [path.name for path in bench_paths],
        "crop": "center",
        "seed": args.seed,
    }
    (bench_dir / "bench.json").write_text(json.dumps(bench_meta, indent=2))
    logger.info("Wrote %d calibration samples to %s and %d bench images to %s",
                args.num_calib, calib_dir, len(bench_paths), bench_dir)  # fmt: skip


if __name__ == "__main__":
    main()

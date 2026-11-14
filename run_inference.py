"""Run dynamic-axis encoder and decoder ONNX models on one image with a simulated AWGN channel."""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from bench.chain import transmit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("encoder", type=Path)
    parser.add_argument("decoder", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--snr", type=float, default=10.0, help="simulated channel SNR in dB")
    parser.add_argument("--channels", type=int, help="real symbol channels to transmit (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("reconstruction.png"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with Image.open(args.image.expanduser()) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    height, width = pixels.shape[-2:]
    encoder = ort.InferenceSession(str(args.encoder.expanduser()), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(args.decoder.expanduser()), providers=["CPUExecutionProvider"])
    channels = decoder.get_inputs()[0].shape[1]
    # 16 is a multiple of every model stride here (8 for RecurrentJSCC, 16 for DeepJSCC_lite).
    padded = np.pad(pixels, ((0, 0), (0, 0), (0, -height % 16), (0, -width % 16)), mode="edge")
    snr = np.full((1, 1), args.snr, dtype=np.float32)

    def feeds(session: ort.InferenceSession, data: np.ndarray) -> dict[str, np.ndarray]:
        names = [i.name for i in session.get_inputs()]
        return {names[0]: data, **({"snr": snr} if "snr" in names else {})}

    symbols = encoder.run(None, feeds(encoder, padded))[0]
    noise = np.random.default_rng(args.seed).standard_normal(symbols.shape).astype(np.float32)
    received = transmit(symbols, noise, args.snr, args.channels or channels, channels)
    result = decoder.run(None, feeds(decoder, received))[0][:, :, :height, :width]
    output = (np.clip(result[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output)
    logging.info("Saved reconstruction to %s", args.output)


if __name__ == "__main__":
    main()

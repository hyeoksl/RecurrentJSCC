"""Simulation evaluation of one experiment on full-resolution images (Kodak).

Writes results/sim/<exp>.json with, for every (encoder K, decoder K) pair x SNR x spp:
PSNR and MS-SSIM averaged over images and noise repeats, plus parameters and GFLOPs per pair and the
encoder/decoder asymmetry table for mirrored pairs. No figures are drawn here (see tools/plots.py).
"""

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from torch import Tensor
from torchvision.transforms.v2 import functional as TF

from model import build_model
from utils.complexity import FLOPS_CONVENTION, link_complexity
from utils.dataset import list_images
from utils.gpus import pick_gpu
from utils.link import run_link
from utils.logging import find_best_ckpt, git_revision, load_config, load_weights

logger = logging.getLogger("evaluate")


def load_image(path: Path, device: torch.device) -> Tensor:
    with Image.open(path) as image:
        return TF.to_dtype(TF.to_image(image.convert("RGB")), torch.float32, scale=True).unsqueeze(0).to(device)


def mean_std(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": tensor.mean().item(), "std": tensor.std(correction=0).item()}


@torch.no_grad()
def evaluate_point(
    model: torch.nn.ModuleDict,
    images: list[Tensor],
    enc_iters: int | None,
    dec_iters: int | None,
    snr_db: float,
    spp: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Per-image PSNR / MS-SSIM averaged over `repeats` noise draws; statistics are over images.

    The noise generator depends only on (seed, image index, repeat), so every (K, K, SNR, spp) point sees
    the same unit-noise realisations.
    """
    device = images[0].device
    psnrs, ssims = [], []
    for index, image in enumerate(images):
        image_psnr, image_ssim = [], []
        for repeat in range(repeats):
            generator = torch.Generator().manual_seed(seed * 1_000_003 + index * 1_009 + repeat)
            snr = torch.full((1, 1), float(snr_db), device=device)
            x_hat = run_link(model, image, snr, spp, enc_iters, dec_iters, generator)["x_hat"]
            mse = torch.mean((x_hat - image) ** 2).clamp_min(1e-12)
            image_psnr.append((10 * torch.log10(1 / mse)).item())
            image_ssim.append(ms_ssim(x_hat, image, data_range=1.0).item())
        psnrs.append(sum(image_psnr) / repeats)
        ssims.append(sum(image_ssim) / repeats)
    return {"psnr": mean_std(psnrs), "ms_ssim": mean_std(ssims)}


def mirror_table(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each pair (a, b) with a < b: PSNR of deep-decoder (a, b) minus deep-encoder (b, a)."""
    by_key = {(p["enc_iters"], p["dec_iters"], p["snr_db"], p["spp"]): p for p in points}
    rows = []
    for (enc, dec, snr_db, spp), point in sorted(by_key.items()):
        mirrored = by_key.get((dec, enc, snr_db, spp))
        if enc is None or enc >= dec or mirrored is None:
            continue
        rows.append(
            {
                "shallow": enc,
                "deep": dec,
                "snr_db": snr_db,
                "spp": spp,
                "psnr_deep_decoder": point["psnr"]["mean"],
                "psnr_deep_encoder": mirrored["psnr"]["mean"],
                "psnr_diff_decoder_minus_encoder": point["psnr"]["mean"] - mirrored["psnr"]["mean"],
                "ms_ssim_deep_decoder": point["ms_ssim"]["mean"],
                "ms_ssim_deep_encoder": mirrored["ms_ssim"]["mean"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--iters", nargs="+", default=["all"], help="'all' (= model.iters) or K values, e.g. 3 5")
    parser.add_argument("--snrs", type=float, nargs="+", help="override eval.snrs")
    parser.add_argument("--spps", type=float, nargs="+", help="override eval.spps")
    parser.add_argument("--repeats", type=int, help="override eval.repeats")
    parser.add_argument("--dataset", help="override eval.dataset")
    parser.add_argument("--max-images", type=int, help="use only the first N images (smoke runs)")
    parser.add_argument("--checkpoint", type=Path, help="default: the experiment's best checkpoint")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--out", type=Path, help="default: results/sim/<exp>.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

    experiment = args.experiment_dir.expanduser()
    config = load_config(experiment)
    cfg_eval = config["eval"]
    dataset = args.dataset or cfg_eval["dataset"]
    snrs = args.snrs or cfg_eval["snrs"]
    spps = args.spps or cfg_eval["spps"]
    repeats = args.repeats or cfg_eval["repeats"]
    if not dataset or not snrs or not spps:
        parser.error("eval.dataset, eval.snrs and eval.spps must be set in the config or on the command line")
    model_iters = config["model"].get("iters")
    if model_iters is None:
        iters: list[int | None] = [None]
    elif args.iters == ["all"]:
        iters = list(model_iters)
    else:
        iters = [int(k) for k in args.iters]

    gpu_id = None if args.use_cpu else pick_gpu()
    device = torch.device("cpu" if gpu_id is None else f"cuda:{gpu_id}")
    checkpoint = args.checkpoint or find_best_ckpt(experiment)
    model = build_model(config["model"])
    epoch = load_weights(model, checkpoint)["epoch"]
    model = model.to(device).eval()

    paths = list_images(dataset)[: args.max_images]
    images = [load_image(path, device) for path in paths]
    logger.info(
        "Evaluating %s (epoch %d) on %d images from %s using %s", checkpoint, epoch, len(images), dataset, device
    )

    flops_h, flops_w = cfg_eval["flops_resolution"]
    combos = list(itertools.product(iters, iters))
    complexity = []
    points = []
    for enc_iters, dec_iters in combos:
        info = link_complexity(model, flops_h, flops_w, enc_iters, dec_iters)
        complexity.append({"enc_iters": enc_iters, "dec_iters": dec_iters, **info})
        for snr_db, spp in itertools.product(snrs, spps):
            point = evaluate_point(model, images, enc_iters, dec_iters, snr_db, spp, repeats, cfg_eval["seed"])
            points.append({"enc_iters": enc_iters, "dec_iters": dec_iters, "snr_db": snr_db, "spp": spp, **point})
        best = max((p for p in points if (p["enc_iters"], p["dec_iters"]) == (enc_iters, dec_iters)), key=_psnr)
        logger.info("enc K=%s dec K=%s done (best PSNR %.2f dB)", enc_iters, dec_iters, _psnr(best))

    result = {
        "experiment": config["experiment_name"],
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "git": git_revision(),
        "train_git": config.get("git"),
        "model": config["model"],
        "train_iters": config.get("train", {}).get("iters"),
        "dataset": str(dataset),
        "num_images": len(images),
        "iters": iters,
        "snrs_db": snrs,
        "spps": spps,
        "repeats": repeats,
        "noise_seed": cfg_eval["seed"],
        "metric_notes": "PSNR/MS-SSIM on float outputs in [0,1]; per-image values averaged over noise repeats, "
        "then mean/std over images. CBR = spp / 3.",
        "flops_resolution": [flops_h, flops_w],
        "flops_convention": FLOPS_CONVENTION,
        "complexity": complexity,
        "points": points,
        "mirror_pairs": mirror_table(points),
    }
    out = args.out or Path("results/sim") / f"{config['experiment_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    logger.info("Wrote %s (%d points)", out, len(points))


def _psnr(point: dict[str, Any]) -> float:
    return point["psnr"]["mean"]


if __name__ == "__main__":
    main()

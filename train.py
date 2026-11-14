"""Train an encoder/decoder pair over a simulated AWGN link."""

import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from criterion import CustomJSCCLoss
from model import build_model
from utils.complexity import link_complexity
from utils.dataset import load_dataloaders
from utils.gpus import pick_gpu
from utils.link import run_link
from utils.logging import load_ckpt, save_ckpt, save_config, setup_logger
from utils.parsing import parse_config
from utils.progress_meter import ProgressMeter
from utils.seed import seed_everything

VALID_NOISE_SEED = 0


def psnr(mse: Tensor) -> Tensor:
    return 10 * torch.log10(1 / mse)


def train_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: nn.ModuleDict,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    writer: SummaryWriter,
    logger: logging.Logger,
    config: dict[str, Any],
) -> None:
    """Each minibatch draws encoder/decoder iterations and spp once, and an SNR per sample."""
    model.train()
    cfg_model, cfg_train = config["model"], config["train"]
    iters_set = cfg_train["iters"] or [None]
    device = next(model.parameters()).device
    max_steps = cfg_train["max_steps"]
    progress = ProgressMeter("train", writer, logger, len(dataloader.dataset), epoch, sys.stdout.isatty())

    for step, data in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        data = data.to(device)
        if cfg_train["channels_last"]:
            data = data.contiguous(memory_format=torch.channels_last)
        enc_iters, dec_iters = random.choice(iters_set), random.choice(iters_set)
        spp = random.uniform(cfg_model["min_spp"], cfg_model["spp"])
        snr = torch.empty(len(data), 1, device=device).uniform_(cfg_model["min_SNRdB"], cfg_model["max_SNRdB"])

        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler is not None):
            out_net = run_link(model, data, snr, spp, enc_iters, dec_iters)
        loss_dict = criterion(out_net, data)
        if scaler is None:
            loss_dict["loss"].backward()
        else:
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
        if config["clip_max_norm"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config["clip_max_norm"])
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        loss_dict["PSNR"] = psnr(loss_dict["MSE"])
        progress.update({k: v.detach() for k, v in loss_dict.items()}, len(data))
    progress.write_summary()


@torch.no_grad()
def validate(
    epoch: int,
    dataloader: DataLoader,
    model: nn.ModuleDict,
    criterion: nn.Module,
    writer: SummaryWriter,
    logger: logging.Logger,
    config: dict[str, Any],
) -> float:
    """Fixed crops, SNR, spp and noise seed. Returns the loss averaged over the (K, K) combinations."""
    model.eval()
    cfg_model, cfg_valid = config["model"], config["valid"]
    combos = [(k, k) for k in config["train"]["iters"] or [None]]
    snr_db = cfg_valid["snr"] if cfg_valid["snr"] is not None else (cfg_model["min_SNRdB"] + cfg_model["max_SNRdB"]) / 2
    spp = cfg_valid["spp"] if cfg_valid["spp"] is not None else cfg_model["spp"]
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(VALID_NOISE_SEED)
    progress = ProgressMeter("valid", writer, logger, len(dataloader.dataset), epoch, sys.stdout.isatty())

    for data in dataloader:
        data = data.to(device)
        if config["train"]["channels_last"]:
            data = data.contiguous(memory_format=torch.channels_last)
        snr = torch.full((len(data), 1), float(snr_db), device=device)
        states: dict[str, Tensor] = {}
        losses, mses = [], []
        for enc_iters, dec_iters in combos:
            loss_dict = criterion(run_link(model, data, snr, spp, enc_iters, dec_iters, generator), data)
            losses.append(loss_dict["loss"])
            mses.append(loss_dict["MSE"])
            if enc_iters is not None:
                states[f"PSNR/enc{enc_iters}_dec{dec_iters}"] = psnr(loss_dict["MSE"])
        states["loss"] = torch.stack(losses).mean()
        states["MSE"] = torch.stack(mses).mean()
        states["PSNR"] = torch.stack([psnr(mse) for mse in mses]).mean()
        progress.update(states, len(data))
    return progress.write_summary()["loss"]


def main(argv: list[str]) -> None:
    config = parse_config(argv)

    gpu_id = None if config["use_cpu"] else pick_gpu()
    if not config["use_cpu"] and gpu_id is None:
        raise RuntimeError("No CUDA device available; pass --use_cpu to train on CPU")
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")

    if config["seed"] is not None:
        seed_everything(config["seed"])

    cfg_log = config["logging"]
    save_path = Path(cfg_log["save_dir"]) / config["experiment_name"]
    logger = setup_logger(save_path, cfg_log["save_every"])
    save_config(config, save_path)
    logger.info("Using %s", device)
    logger.info("Config: %s", config)
    train_writer = SummaryWriter(str(save_path / "train"))
    valid_writer = SummaryWriter(str(save_path / "valid"))

    criterion = CustomJSCCLoss(loss_type=config["loss_type"])
    model = build_model(config["model"]).to(device)
    cfg_train = config["train"]
    if cfg_train["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    for iters in sorted(set(cfg_train["iters"] or [None]), key=lambda k: k or 0):
        logger.info("Complexity at 256x256, iters=%s: %s", iters, link_complexity(model, 256, 256, iters, iters))

    train_dl, valid_dl = load_dataloaders(**config["dataset"], seed=config["seed"])
    logger.info("Number of dataset | train : %d, valid : %d", len(train_dl.dataset), len(valid_dl.dataset))

    cfg_optim = config["optimizer"]
    if cfg_optim["type"] != "Adam":
        raise NameError(f"optimizer.type {cfg_optim['type']} is not supported")
    optimizer = optim.Adam(model.parameters(), lr=cfg_optim["learning_rate"])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, cfg_optim["lr_milestone_epoch"], gamma=cfg_optim["lr_gamma"]
    )
    scaler = torch.amp.GradScaler(device.type) if cfg_train["amp"] else None

    start_epoch, best_loss = load_ckpt(cfg_log["pretrain_ckpt"], model, optimizer, lr_scheduler, logger, scaler)

    old_lr = None
    for epoch in range(start_epoch, config["epochs"]):
        now_lr = optimizer.param_groups[0]["lr"]
        if old_lr != now_lr:
            logger.info("Learning rate change: %s -> %s", old_lr, now_lr)
            old_lr = now_lr

        train_epoch(epoch, train_dl, model, criterion, optimizer, scaler, train_writer, logger, config)
        loss = validate(epoch, valid_dl, model, criterion, valid_writer, logger, config)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if cfg_log["save_model"]:
            save_ckpt(
                save_path,
                cfg_log["save_every"],
                epoch,
                loss,
                best_loss,
                is_best,
                model,
                optimizer,
                lr_scheduler,
                logger,
                scaler,
            )


if __name__ == "__main__":
    main(sys.argv[1:])

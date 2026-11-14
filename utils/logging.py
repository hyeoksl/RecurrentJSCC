import logging
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

LOGGER_NAME = "jscc"


def setup_logger(save_path: Path, save_every: int = 0) -> logging.Logger:
    """Log to the console and to <save_path>/logging.log. Creates the experiment directory."""
    save_path.mkdir(parents=True, exist_ok=True)
    if save_every > 0:
        (save_path / "weights").mkdir(exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logfile = logging.FileHandler(save_path / "logging.log")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(logfile)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def git_revision() -> dict[str, Any]:
    """Commit hash of the working tree and whether it has uncommitted changes (None if not a git repo)."""
    root = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def save_config(config: dict[str, Any], save_path: Path) -> None:
    """Write the config as a plain dict (readable with yaml.safe_load), with the git revision."""
    with (save_path / "config.yaml").open("w") as file:
        yaml.safe_dump({**config, "git": git_revision()}, file, sort_keys=False)


def load_config(experiment_dir: Path) -> dict[str, Any]:
    with (experiment_dir / "config.yaml").open() as file:
        return yaml.safe_load(file)


def find_best_ckpt(experiment_dir: Path) -> Path:
    checkpoints = sorted(experiment_dir.glob("best_ckpt_ep*.pth.tar"))
    if len(checkpoints) != 1:
        raise FileNotFoundError(f"Expected exactly one best checkpoint in {experiment_dir}, found {len(checkpoints)}")
    return checkpoints[0]


def load_weights(model: nn.Module, ckpt: Path) -> dict[str, Any]:
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()})
    return checkpoint


def load_ckpt(
    ckpt: str | Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: logging.Logger,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, float]:
    """Resume from `ckpt`. Returns (start_epoch, best_loss). Any failure to load is raised."""
    if ckpt is None:
        logger.info("Training from scratch...")
        return 0, float("inf")
    checkpoint = torch.load(ckpt, map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "best_loss" not in checkpoint:
        logger.warning("Checkpoint %s has no best_loss; using its own loss as the best so far", ckpt)
    best_loss = float(checkpoint.get("best_loss", checkpoint["loss"]))
    logger.info("Loaded %s (epoch %d, best_loss %.6f)", ckpt, checkpoint["epoch"], best_loss)
    return checkpoint["epoch"] + 1, best_loss


def save_ckpt(
    save_path: Path,
    save_every: int,
    epoch: int,
    loss: float,
    best_loss: float,
    is_best: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    logger: logging.Logger,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    def save_only(prefix: str) -> None:
        target = save_path / f"{prefix}_ep{epoch}.pth.tar"
        torch.save(state, target)
        for old in save_path.glob(f"{prefix}_ep*.pth.tar"):
            if old != target:
                old.unlink()

    save_only("latest_ckpt")
    logger.info("Saved latest model in epoch : %d", epoch)
    if save_every > 0 and (epoch + 1) % save_every == 0:
        torch.save(state, save_path / "weights" / f"Epoch_{epoch}.pth.tar")
        logger.info("Saved model milestone in epoch : %d", epoch)
    if is_best:
        save_only("best_ckpt")
        logger.info("Saved best model in epoch : %d", epoch)

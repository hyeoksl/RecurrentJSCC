import logging
import sys
from collections.abc import Mapping

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ProgressMeter:
    """Batch-size-weighted running averages for one epoch, written to TensorBoard and the log at the end."""

    def __init__(
        self, desc: str, writer: SummaryWriter, logger: logging.Logger, total: int, epoch: int, use_pbar: bool
    ) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0
        self.writer = writer
        self.logger = logger
        self.epoch = epoch
        self.total = total
        self.desc = f"Epoch {epoch} | {desc}: "
        self.pbar = tqdm(total=total, dynamic_ncols=True, file=sys.stdout, desc=self.desc) if use_pbar else None

    def update(self, states: Mapping[str, torch.Tensor | float], batch_size: int) -> None:
        if "loss" not in states:
            raise KeyError("states must contain 'loss'")
        self.count += batch_size
        for name, value in states.items():
            self.sums[name] = self.sums.get(name, 0.0) + float(value) * batch_size
        if self.pbar is not None:
            self.pbar.set_postfix({k: f"{v:.5f}" for k, v in self.averages().items()}, refresh=False)
            self.pbar.update(batch_size)

    def averages(self) -> dict[str, float]:
        return {name: total / self.count for name, total in self.sums.items()}

    def write_summary(self) -> dict[str, float]:
        averages = self.averages()
        for name, value in averages.items():
            self.writer.add_scalar(name, value, self.epoch)
        if self.pbar is not None:
            self.pbar.close()
        stats = "".join(f" | {name}: {value:.5f}" for name, value in averages.items())
        self.logger.info("Epoch Summary | %s(%d/%d)%s", self.desc, self.count, self.total, stats)
        return averages

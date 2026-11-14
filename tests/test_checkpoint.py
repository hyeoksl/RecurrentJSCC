import logging

import pytest
import torch
import yaml
from conftest import build, model_cfg

from utils.logging import find_best_ckpt, load_ckpt, load_config, save_ckpt, save_config


def make_training_state(seed=0):
    model = build(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2], gamma=0.1)
    return model, optimizer, scheduler


def test_save_load_resume(tmp_path):
    logger = logging.getLogger("test")
    model, optimizer, scheduler = make_training_state()
    for _ in range(3):
        optimizer.zero_grad()
        sum(p.pow(2).sum() for p in model.parameters()).backward()
        optimizer.step()
        scheduler.step()
    save_ckpt(tmp_path, 0, 2, 0.5, 0.25, True, model, optimizer, scheduler, logger)

    resumed, optimizer2, scheduler2 = make_training_state(seed=1)
    start_epoch, best_loss = load_ckpt(find_best_ckpt(tmp_path), resumed, optimizer2, scheduler2, logger)
    assert start_epoch == 3
    assert best_loss == 0.25
    assert optimizer2.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    assert scheduler2.last_epoch == scheduler.last_epoch
    for key, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    first, second = optimizer.state_dict()["state"], optimizer2.state_dict()["state"]
    assert first.keys() == second.keys() and len(first) > 0
    for index in first:
        assert first[index]["step"] == second[index]["step"] == 3
        assert torch.equal(first[index]["exp_avg"], second[index]["exp_avg"])


def test_latest_and_best_are_rotated(tmp_path):
    logger = logging.getLogger("test")
    model, optimizer, scheduler = make_training_state()
    (tmp_path / "weights").mkdir()
    save_ckpt(tmp_path, 2, 0, 1.0, 1.0, True, model, optimizer, scheduler, logger)
    save_ckpt(tmp_path, 2, 1, 2.0, 1.0, False, model, optimizer, scheduler, logger)
    names = sorted(p.name for p in tmp_path.glob("*.pth.tar"))
    assert names == ["best_ckpt_ep0.pth.tar", "latest_ckpt_ep1.pth.tar"]
    assert [p.name for p in (tmp_path / "weights").iterdir()] == ["Epoch_1.pth.tar"]


def test_load_failure_is_raised(tmp_path):
    logger = logging.getLogger("test")
    model, optimizer, scheduler = make_training_state()
    with pytest.raises(FileNotFoundError):
        load_ckpt(tmp_path / "missing.pth.tar", model, optimizer, scheduler, logger)
    other = build(dim=32)
    save_ckpt(tmp_path, 0, 0, 1.0, 1.0, False, other, torch.optim.Adam(other.parameters()), scheduler, logger)
    with pytest.raises(RuntimeError):
        load_ckpt(tmp_path / "latest_ckpt_ep0.pth.tar", model, optimizer, scheduler, logger)


def test_config_is_plain_yaml(tmp_path):
    save_config({"experiment_name": "x", "model": model_cfg()}, tmp_path)
    with (tmp_path / "config.yaml").open() as file:
        loaded = yaml.safe_load(file)
    assert loaded["model"] == model_cfg()
    assert set(loaded["git"]) == {"commit", "dirty"}
    assert load_config(tmp_path) == loaded

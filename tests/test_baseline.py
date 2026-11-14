import torch
import yaml

import train
from export_onnx import TOLERANCE, export_module
from model import build_model
from utils.logging import find_best_ckpt, load_config, load_weights
from utils.parsing import parse_config


def write_images(folder, count, size=96):
    from PIL import Image

    folder.mkdir()
    generator = torch.Generator().manual_seed(0)
    for index in range(count):
        pixels = (torch.rand(size, size, 3, generator=generator) * 255).to(torch.uint8).numpy()
        Image.fromarray(pixels).save(folder / f"{index}.png")


def smoke_args(tmp_path, config_file):
    write_images(tmp_path / "train", 4)
    write_images(tmp_path / "valid", 2)
    return [
        "--config_file", config_file, "--use_cpu", "--epochs", "1", "--train.max_steps", "1",
        "--model.dim", "16", "--dataset.train_dataset", str(tmp_path / "train"),
        "--dataset.valid_dataset", str(tmp_path / "valid"), "--dataset.train_batch_size", "2",
        "--dataset.valid_batch_size", "2", "--dataset.num_workers", "0", "--dataset.crop_size", "64",
        "--dataset.valid_crop_size", "64", "--dataset.train_resize", "null",
        "--logging.save_dir", str(tmp_path / "logs"), "--logging.save_every", "0",
    ]  # fmt: skip


def test_baseline_trains_one_step_and_exports(tmp_path):
    train.main(smoke_args(tmp_path, "configs/baseline.yaml"))
    experiment = tmp_path / "logs" / "baseline"
    with (experiment / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    assert config["model"]["type"] == "DeepJSCC_lite"
    model = build_model(config["model"])
    checkpoint = load_weights(model, find_best_ckpt(experiment))
    assert checkpoint["epoch"] == 0 and checkpoint["best_loss"] == checkpoint["loss"]
    for role in ("encoder", "decoder"):
        for static_hw in (None, (64, 64)):
            info = export_module(model[role], role, None, tmp_path / f"{role}.onnx", static_hw)
            assert info["max_abs_diff_vs_torch"] <= TOLERANCE


def test_recurrent_trains_and_resumes(tmp_path):
    args = smoke_args(tmp_path, "configs/recurrent/joint_rel.yaml")
    train.main(args)
    experiment = tmp_path / "logs" / "joint_rel"
    config = load_config(experiment)
    assert config["train"]["iters"] == [1, 2, 4, 6]
    first = torch.load(find_best_ckpt(experiment), weights_only=True)
    latest = next(experiment.glob("latest_ckpt_ep0.pth.tar"))
    resumed = [a if a != "1" or args[i - 1] != "--epochs" else "2" for i, a in enumerate(args)]
    train.main([*resumed, "--logging.pretrain_ckpt", str(latest)])
    second = torch.load(next(experiment.glob("latest_ckpt_ep1.pth.tar")), weights_only=True)
    assert second["epoch"] == 1
    assert second["best_loss"] <= first["best_loss"]


def test_parse_config_fills_train_iters_and_rejects_bad_unshared():
    config = parse_config(["--config_file", "configs/recurrent/fixed_k2_rel.yaml"])
    assert config["train"]["iters"] == [2] and config["model"]["iters"] == [1, 2, 4, 6]
    config = parse_config(["--config_file", "configs/recurrent/unshared_k4.yaml", "--model.dim", "32"])
    assert config["model"]["dim"] == 32 and not config["model"]["share_core"]
    try:
        parse_config(["--config_file", "configs/recurrent/unshared_k4.yaml", "--train.iters", "[1, 4]"])
    except ValueError:
        pass
    else:
        raise AssertionError("unshared core with several training depths must be rejected")

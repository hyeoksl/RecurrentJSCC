import argparse
from typing import Any

import yaml
from jsonargparse import ActionConfigFile, ArgumentParser

from model import get_spec


def _model_type(argv: list[str]) -> str:
    """model.type decides which model arguments exist, so read it before building the real parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_file", type=str)
    pre.add_argument("--model.type", type=str, dest="model_type")
    args, _ = pre.parse_known_args(argv)
    if args.model_type is not None:
        return args.model_type
    if args.config_file is None:
        pre.error("--config_file is required")
    with open(args.config_file) as file:
        model_type = (yaml.safe_load(file).get("model") or {}).get("type")
    if model_type is None:
        pre.error("model.type must be given in the config file or on the command line")
    return model_type


def build_parser(model_type: str) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config_file", action=ActionConfigFile, help="YAML file with parameters")
    parser.add_argument("--use_cpu", action="store_true", help="do not use a GPU")

    parser.add_argument("--experiment_name", type=str, default="TEST")
    parser.add_argument("--seed", type=int | None, default=None)
    parser.add_argument("--loss_type", type=str, default="MSE")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--clip_max_norm", type=float, default=0.0)

    parser.add_argument("--train.iters", type=list[int] | None, default=None, help="default: model.iters")
    parser.add_argument("--train.amp", type=bool, default=False)
    parser.add_argument("--train.channels_last", type=bool, default=False)
    parser.add_argument("--train.max_steps", type=int | None, default=None, help="stop each epoch early (smoke runs)")

    parser.add_argument("--valid.snr", type=float | None, default=None, help="default: midpoint of the SNR range")
    parser.add_argument("--valid.spp", type=float | None, default=None, help="default: model.spp")

    parser.add_argument("--optimizer.type", type=str, default="Adam")
    parser.add_argument("--optimizer.learning_rate", type=float)
    parser.add_argument("--optimizer.lr_milestone_epoch", type=list[int], default=[])
    parser.add_argument("--optimizer.lr_gamma", type=float, default=0.1)

    parser.add_argument("--dataset.train_dataset", type=str)
    parser.add_argument("--dataset.valid_dataset", type=str)
    parser.add_argument("--dataset.train_batch_size", type=int)
    parser.add_argument("--dataset.valid_batch_size", type=int)
    parser.add_argument("--dataset.num_workers", type=int)
    parser.add_argument("--dataset.crop_size", type=int)
    parser.add_argument("--dataset.valid_crop_size", type=int)
    parser.add_argument("--dataset.train_resize", type=list[int] | None, default=None)

    parser.add_argument("--eval.dataset", type=str | None, default=None)
    parser.add_argument("--eval.snrs", type=list[float], default=[])
    parser.add_argument("--eval.spps", type=list[float], default=[])
    parser.add_argument("--eval.repeats", type=int, default=1)
    parser.add_argument("--eval.seed", type=int, default=0)
    parser.add_argument("--eval.flops_resolution", type=list[int], default=[256, 256])

    parser.add_argument("--logging.save_dir", type=str, default="logs/")
    parser.add_argument("--logging.save_model", type=bool, default=True)
    parser.add_argument("--logging.save_every", type=int, default=0)
    parser.add_argument("--logging.pretrain_ckpt", type=str | None, default=None)

    get_spec(model_type).add_arguments(parser)
    return parser


def parse_config(argv: list[str]) -> dict[str, Any]:
    """Parse YAML + CLI overrides (e.g. `--model.dim 96`) into a plain nested dict."""
    config = build_parser(_model_type(argv)).parse_args(argv).as_dict()
    config.pop("config_file", None)
    config.pop("__path__", None)
    model = config["model"]
    if model["min_SNRdB"] > model["max_SNRdB"] or not 0 < model["min_spp"] <= model["spp"]:
        raise ValueError("Require min_SNRdB <= max_SNRdB and 0 < min_spp <= spp")
    if "iters" in model:
        train_iters = config["train"]["iters"] or list(model["iters"])
        config["train"]["iters"] = train_iters
        if not model["share_core"] and (len(train_iters) != 1 or train_iters != list(model["iters"])):
            raise ValueError("share_core=false requires model.iters == train.iters == [K]")
    return config

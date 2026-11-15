# RecurrentJSCC

Image transmission with a weight-shared recurrent core: one ConvNeXt core applied K times in the encoder
and K times in the decoder (K chosen independently on each side), FiLM depth conditioning, and static
unrolling for NPU export. `DeepJSCC_lite` from the `separ` branch is kept as a regression baseline.
The full specification and milestones are in `CLAUDE.md`.

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/). PyTorch uses the cu128 index; `onnxruntime` is pinned to
1.23.x to match the board.

```sh
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

## Train, evaluate, export

```sh
uv run python train.py --config_file configs/recurrent/joint_rel.yaml
uv run python evaluate.py logs/joint_rel --iters all                       # results/sim/joint_rel.json
uv run python evaluate.py logs/joint_rel --iters 3 5 --out results/sim/joint_rel_heldout.json
uv run python export_onnx.py logs/joint_rel                                # dynamic axes, PC inference
uv run python export_onnx.py logs/joint_rel --static 256 256               # fixed shapes, NPU
uv run python run_inference.py artifacts/joint_rel/onnx/encoder_k2_dyn.onnx \
    artifacts/joint_rel/onnx/decoder_k4_dyn.onnx input.png --snr 10
uv run python tools/plots.py                                               # results/figures, results/tables
```

Any config value can be overridden on the command line (`--model.dim 96`, `--train.iters "[2]"`).
`--train.max_steps N` stops each epoch after N steps for smoke runs. Experiments are written to
`logs/<experiment_name>/` (`config.yaml` as a plain dict with the git revision, `logging.log`, TensorBoard,
`latest_ckpt_ep*`, `best_ckpt_ep*`, `weights/Epoch_*`). Resume with `--logging.pretrain_ckpt <ckpt>`.

| config | what it trains |
|---|---|
| `recurrent/joint_{rel,abs,none}` | shared core, K sampled from {1, 2, 4, 6} per minibatch, FiLM relative / absolute / none |
| `recurrent/fixed_k{1,2,4,6}_rel` | shared core, a single training depth (`train.iters: [K]`) |
| `recurrent/unshared_k{1,2,4,6}` | unshared stack of the same depth (`share_core: false`, `model.iters: [K]`) |
| `baseline` | DeepJSCC_lite |

## The link

`utils/link.py::run_link` is the only place the pieces meet; nothing below is inside a model graph:

    pad image to the stride -> encoder(image, snr, K) -> keep the first channels for the spp
    -> per-sample power normalization -> AWGN -> zero-pad channels -> decoder(received, snr, K) -> crop

Symbols are real `(B, C, h, w)` tensors; adjacent channel pairs are I/Q of one complex symbol. At stride 8,
spp 0.5 is 64 real channels. `snr` is a `(B, 1)` tensor in dB and a graph input after export.
`bench/chain.py` is the numpy twin of `channel.py` used by the ONNX tools and on the board.

## NPU pipeline (Rubik Pi 3, ONNX Runtime QNN EP)

Server side (this repository):

```sh
uv run python export_onnx.py logs/joint_rel --static 256 256        # artifacts/joint_rel/onnx
uv run python tools/make_calib.py artifacts/joint_rel               # calib/ and bench/ inputs
uv run python tools/quantize.py artifacts/joint_rel                 # qdq/a8, qdq/a16, results/npu/joint_rel/onnx_stats.json
uv run python tools/make_bundle.py artifacts/joint_rel --name joint_rel   # artifacts/bundles/joint_rel.tar.gz
uv run python tools/collect_npu.py joint_rel --server-ref           # CPU-EP reference -> results/npu/joint_rel/server_cpu
```

Toolchain checks without a checkpoint:

```sh
uv run python tools/m0a_dummy.py && uv run python tools/quantize.py artifacts/m0a_dummy
uv run python tools/make_bundle.py artifacts/m0a_dummy --name m0a --fp32-on-qnn
uv run python export_onnx.py configs/recurrent/joint_rel.yaml --random-init --static 256 256 \
    --enc-iters 1 6 --dec-iters 1 6 --name m0b_bn --override model.norm=batchnorm
```

Board side. A bundle is self-contained (models, inputs, `bench/`); it needs only numpy and onnxruntime-qnn:

```sh
tar xzf joint_rel.tar.gz && cd <recurrent-npu-jscc project>
uv run python <path>/joint_rel/bench/run_all.py            # writes <path>/joint_rel/out/<timestamp>/
```

`tools/relay.sh <bundle>` does server -> local PC -> board -> run -> results back to `incoming/` in one go
(see the header of the script for the environment variables). With `BOARD_SSH` and `BOARD_PROJECT` set on
the server, `tools/make_bundle.py --send` and `tools/collect_npu.py <bundle> --run` do the same directly.
Either way, `uv run python tools/collect_npu.py <bundle>` parses a results tarball into
`results/npu/<bundle>/<timestamp>/` (`latency.csv`, `psnr.csv`, `thermal.csv`, `placement.json`, `sysinfo.json`).

`bench/run_all.py` measures, per model: QNN compile time and latency (median/p95), reload time from the
context cache, peak RSS, context binary size, a run with CPU fallback disabled, node placement (verbose log
and ORT profile), CPU-EP latency, and thermal state around every step; then PSNR for every
(encoder K, decoder K) pair with fixed channel noise, saving raw outputs as `.npy`.

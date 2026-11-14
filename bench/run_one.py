"""Measure one ONNX model in a fresh process (so peak RSS and session-creation time are per model).

Modes:
  bench      create a session (QNN: compiles for HTP and dumps a context cache), warm up, time N runs
  reload     QNN only: create a session from the dumped context model, time it, a few runs
  strict     QNN only: same as bench but with CPU fallback disabled; succeeds only if every node is on HTP
  placement  verbose ORT logging (to stderr) + ORT profiling for a few runs, to see node -> provider

Depends only on numpy and onnxruntime. Writes one JSON file; never raises on a model failure.
"""

import argparse
import json
import os
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort

NUMPY_TYPES = {"tensor(float)": np.float32, "tensor(uint8)": np.uint8, "tensor(uint16)": np.uint16}


def peak_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def qnn_provider_options(args: argparse.Namespace) -> dict[str, str]:
    options = {"backend_path": args.backend_path, "htp_performance_mode": args.perf_mode}
    for assignment in args.provider_option:
        key, _, value = assignment.partition("=")
        options[key] = value
    return options


def make_session(args: argparse.Namespace) -> tuple[ort.InferenceSession, dict]:
    options = ort.SessionOptions()
    used: dict = {"session_config": {}, "provider_options": None}

    def config(key: str, value: str) -> None:
        options.add_session_config_entry(key, value)
        used["session_config"][key] = value

    if args.mode == "placement":
        options.log_severity_level = 0
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(args.out).with_suffix("")) + "_ortprofile"
    if args.ep == "cpu":
        if args.threads:
            options.intra_op_num_threads = args.threads
        return ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"]), used

    if args.mode == "bench" and args.ctx:
        ctx = Path(args.ctx)
        ctx.parent.mkdir(parents=True, exist_ok=True)
        for stale in ctx.parent.glob(ctx.stem + "*"):
            stale.unlink()
        config("ep.context_enable", "1")
        config("ep.context_file_path", str(ctx))
        config("ep.context_embed_mode", "0")
    if args.mode == "strict":
        config("session.disable_cpu_ep_fallback", "1")
    used["provider_options"] = qnn_provider_options(args)
    providers = [("QNNExecutionProvider", used["provider_options"])]
    if args.mode != "strict":
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(args.model, options, providers=providers), used


def random_feeds(session: ort.InferenceSession, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds = {}
    for meta in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in meta.shape]
        if meta.name == "snr":
            feeds[meta.name] = np.full(shape, 10.0, dtype=np.float32)
        elif meta.name == "image":
            feeds[meta.name] = rng.random(shape, dtype=np.float32)
        else:
            feeds[meta.name] = rng.standard_normal(shape).astype(NUMPY_TYPES.get(meta.type, np.float32))
    return feeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ep", choices=["qnn", "cpu"], required=True)
    parser.add_argument("--mode", choices=["bench", "reload", "strict", "placement"], default="bench")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ctx", help="context cache model path to dump (QNN bench mode)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0, help="CPU EP intra-op threads (0 = default)")
    parser.add_argument("--backend-path", default="libQnnHtp.so")
    parser.add_argument("--perf-mode", default="burst")
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    result: dict = {
        "model": args.model,
        "ep": args.ep,
        "mode": args.mode,
        "ok": False,
        "onnxruntime": ort.__version__,
        "pid": os.getpid(),
        "rss_before_session_kb": peak_rss_kb(),
    }
    try:
        start = time.perf_counter()
        session, used = make_session(args)
        result["create_s"] = time.perf_counter() - start
        result.update(used)
        result["providers"] = session.get_providers()
        result["rss_after_session_kb"] = peak_rss_kb()
        feeds = random_feeds(session)
        result["inputs"] = {name: list(value.shape) for name, value in feeds.items()}

        start = time.perf_counter()
        session.run(None, feeds)
        result["first_run_ms"] = (time.perf_counter() - start) * 1e3
        for _ in range(max(args.warmup - 1, 0)):
            session.run(None, feeds)
        latencies = []
        for _ in range(args.runs):
            start = time.perf_counter()
            session.run(None, feeds)
            latencies.append((time.perf_counter() - start) * 1e3)
        if latencies:
            values = np.asarray(latencies)
            result["latency_ms"] = {
                "n": len(latencies),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            result["latencies_ms"] = [round(v, 4) for v in latencies]
        if args.mode == "placement":
            result["ort_profile"] = session.end_profiling()
        result["peak_rss_kb"] = peak_rss_kb()
        result["ok"] = True
    except Exception as error:  # noqa: BLE001 - every failure is a measurement result here
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        result["peak_rss_kb"] = peak_rss_kb()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

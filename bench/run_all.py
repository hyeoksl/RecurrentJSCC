"""Run every model of a bundle on this machine and record the measurements.

Per model and execution provider (each step in its own process, see run_one.py):
  qnn: compile + latency (dumps the context cache), reload from the cache, strict (no CPU fallback),
       placement (verbose log + ORT profile)
  cpu: latency
Then the PSNR chain per artifact/precision: encoder -> numpy channel with fixed noise -> decoder, for
every (encoder K, decoder K) pair. Raw outputs are saved as .npy so the server can recompute PSNR.

Thermal zones, cooling devices and CPU frequency limits are sampled around every step.
Depends only on numpy and onnxruntime. Run from anywhere: `uv run python bench/run_all.py`.
"""

import argparse
import glob
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain import psnr, transmit  # noqa: E402

logger = logging.getLogger("bench")
RUN_ONE = Path(__file__).resolve().parent / "run_one.py"


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def thermal_snapshot() -> dict:
    """Temperatures (deg C), cooling-device states and cpufreq limits. Missing files are skipped."""
    zones = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        temp, kind = read_text(f"{zone}/temp"), read_text(f"{zone}/type")
        if temp is not None and temp.lstrip("-").isdigit():
            zones[f"{Path(zone).name}:{kind}"] = int(temp) / 1000.0
    cooling = {}
    for device in sorted(glob.glob("/sys/class/thermal/cooling_device*")):
        state, limit = read_text(f"{device}/cur_state"), read_text(f"{device}/max_state")
        if state is not None and state.isdigit():
            cooling[f"{Path(device).name}:{read_text(f'{device}/type')}"] = {"cur": int(state), "max": limit}
    cpufreq = {}
    for policy in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        cpufreq[Path(policy).name] = {
            key: read_text(f"{policy}/{key}") for key in ("scaling_cur_freq", "scaling_max_freq", "cpuinfo_max_freq")
        }
    throttled = any(c["cur"] > 0 for c in cooling.values()) or any(
        f["scaling_max_freq"] and f["cpuinfo_max_freq"] and int(f["scaling_max_freq"]) < int(f["cpuinfo_max_freq"])
        for f in cpufreq.values()
    )
    return {
        "time": time.time(),
        "max_temp_c": max(zones.values(), default=None),
        "zones_c": zones,
        "cooling_devices": cooling,
        "cpufreq": cpufreq,
        "throttled": throttled,
    }


def sysinfo() -> dict:
    capi = Path(ort.__file__).parent / "capi"
    libs = sorted(p for pattern in ("libQnn*", "*[Ss]kel*", "libonnxruntime*") for p in capi.glob(pattern))
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "onnxruntime_capi_dir": str(capi),
        "bundled_libraries": {p.name: p.stat().st_size for p in libs},
        "has_libQnnHtp": (capi / "libQnnHtp.so").exists(),
        "env": {key: os.environ.get(key) for key in ("ADSP_LIBRARY_PATH", "LD_LIBRARY_PATH")},
        "cpu_count": os.cpu_count(),
        "soc": read_text("/sys/devices/soc0/machine") or read_text("/proc/device-tree/model"),
        "soc_id": read_text("/sys/devices/soc0/soc_id"),
        "meminfo": (read_text("/proc/meminfo") or "").splitlines()[:3],
    }


def run_step(out_dir: Path, tag: str, timeout: float, cli: list[str]) -> dict:
    """Run run_one.py in a subprocess; stderr (ORT logs) goes to logs/<tag>.log."""
    result_path = out_dir / "steps" / f"{tag}.json"
    log_path = out_dir / "logs" / f"{tag}.log"
    for path in (result_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    before = thermal_snapshot()
    command = [sys.executable, str(RUN_ONE), "--out", str(result_path), *cli]
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        returncode: int | str = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = "timeout"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"ok": False, "error": f"run_one.py produced no result (exit: {returncode})"}
    result.update({"tag": tag, "returncode": returncode, "log": str(log_path.relative_to(out_dir))})
    result["thermal_before"], result["thermal_after"] = before, thermal_snapshot()
    result.pop("traceback", None)
    return result


def placement_summary(log_path: Path) -> dict:
    """Lines of the verbose ORT log that say where nodes were placed."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    keep = re.compile(r"placed on|Node placements|Provider: \[|number of nodes|Number of partitions|fallback", re.I)
    return {"lines": [line.strip()[:2000] for line in text.splitlines() if keep.search(line)][:200]}


def context_files(ctx: Path) -> dict:
    return {p.name: p.stat().st_size for p in sorted(ctx.parent.glob(ctx.stem + "*"))} if ctx.parent.exists() else {}


def measure_model(entry: dict, bundle: Path, out_dir: Path, eps: list[str], args: argparse.Namespace) -> dict:
    model = bundle / entry["path"]
    tag = entry["id"].replace("/", "__")
    record = {**entry, "file_bytes": model.stat().st_size, "steps": {}}
    common = ["--model", str(model), "--warmup", str(args.warmup), "--runs", str(args.runs)]
    use_qnn = "qnn" in eps and (entry["precision"] != "fp32" or args.fp32_on_qnn)
    if use_qnn:
        ctx = out_dir / "ctx" / f"{tag}_ctx.onnx"
        qnn = [*common, "--ep", "qnn", "--perf-mode", args.perf_mode]
        qnn += [arg for option in args.provider_option for arg in ("--provider-option", option)]
        record["steps"]["qnn_bench"] = run_step(out_dir, f"{tag}.qnn_bench", args.timeout, [*qnn, "--ctx", str(ctx)])
        record["context_files"] = context_files(ctx)
        if ctx.exists():
            reload_cli = [*qnn, "--mode", "reload", "--runs", "20"]
            reload_cli[reload_cli.index("--model") + 1] = str(ctx)
            record["steps"]["qnn_reload"] = run_step(out_dir, f"{tag}.qnn_reload", args.timeout, reload_cli)
        strict = [*qnn, "--mode", "strict", "--runs", "20"]
        record["steps"]["qnn_strict"] = run_step(out_dir, f"{tag}.qnn_strict", args.timeout, strict)
        placement = [*qnn, "--mode", "placement", "--warmup", "1", "--runs", "3"]
        record["steps"]["qnn_placement"] = run_step(out_dir, f"{tag}.qnn_placement", args.timeout, placement)
        record["placement_log"] = placement_summary(out_dir / "logs" / f"{tag}.qnn_placement.log")
    if "cpu" in eps:
        record["steps"]["cpu_bench"] = run_step(out_dir, f"{tag}.cpu_bench", args.timeout, [*common, "--ep", "cpu"])
    for name, step in record["steps"].items():
        latency = step.get("latency_ms", {})
        logger.info(
            "%s %s: ok=%s create=%.2fs median=%.2fms p95=%.2fms %s",
            entry["id"], name, step.get("ok"), step.get("create_s", float("nan")),
            latency.get("median", float("nan")), latency.get("p95", float("nan")), step.get("error", ""),
        )  # fmt: skip
    return record


def open_session(model: Path, ep: str, ctx: Path, args: argparse.Namespace) -> ort.InferenceSession:
    if ep == "cpu":
        return ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    options = {"backend_path": "libQnnHtp.so", "htp_performance_mode": args.perf_mode}
    for assignment in args.provider_option:
        key, _, value = assignment.partition("=")
        options[key] = value
    source = ctx if ctx.exists() else model
    return ort.InferenceSession(str(source), providers=[("QNNExecutionProvider", options), "CPUExecutionProvider"])


def run_chain(group: dict, bundle: Path, out_dir: Path, ep: str, args: argparse.Namespace) -> dict:
    """PSNR for every (encoder K, decoder K) pair of one artifact/precision on one execution provider."""
    bench = group["bench"]
    images = np.load(bundle / bench["images"])
    noise = np.load(bundle / bench["noise"])
    snr_db, active, channels = bench["snr_db"], bench["active_channels"], bench["symbol_channels"]
    snr = np.full((1, 1), snr_db, dtype=np.float32)
    save_dir = out_dir / "outputs" / group["artifact"] / group["precision"] / ep
    save_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def infer(entry: dict, name: str, batch: np.ndarray) -> np.ndarray:
        ctx = out_dir / "ctx" / f"{entry['id'].replace('/', '__')}_ctx.onnx"
        session = open_session(bundle / entry["path"], ep, ctx, args)
        extra = {"snr": snr} if "snr" in {i.name for i in session.get_inputs()} else {}
        return np.concatenate([session.run(None, {name: batch[i : i + 1], **extra})[0] for i in range(len(batch))])

    symbols = {}
    for entry in group["encoders"]:
        symbols[entry["k"]] = infer(entry, "image", images)
        np.save(save_dir / f"symbols_k{entry['k']}.npy", symbols[entry["k"]])
    pairs = []
    for entry in group["decoders"]:
        for enc_k, sym in symbols.items():
            received = transmit(sym, noise, snr_db, active, channels)
            recon = infer(entry, "received", received)
            values = psnr(images, recon)
            pairs.append({"enc_k": enc_k, "dec_k": entry["k"], "psnr_mean": float(values.mean()),
                          "psnr_per_image": [round(float(v), 4) for v in values]})  # fmt: skip
            if args.save_recon == "all" or (args.save_recon == "diag" and enc_k == entry["k"]):
                np.save(save_dir / f"recon_e{enc_k}_d{entry['k']}.npy", recon.astype(np.float16))
            logger.info("chain %s/%s %s enc K=%s dec K=%s: PSNR %.3f dB",
                        group["artifact"], group["precision"], ep, enc_k, entry["k"], values.mean())  # fmt: skip
    return {"artifact": group["artifact"], "precision": group["precision"], "ep": ep, "snr_db": snr_db,
            "active_channels": active, "num_images": len(images), "seconds": time.time() - start,
            "outputs": str(save_dir.relative_to(out_dir)), "pairs": pairs}  # fmt: skip


def chain_groups(manifest: dict) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for entry in manifest["models"]:
        bench = manifest.get("bench", {}).get(entry["artifact"])
        if bench is None or entry["role"] not in ("encoder", "decoder"):
            continue
        group = groups.setdefault(
            (entry["artifact"], entry["precision"]),
            {"artifact": entry["artifact"], "precision": entry["precision"], "bench": bench,
             "encoders": [], "decoders": []},
        )  # fmt: skip
        group[entry["role"] + "s"].append(entry)
    return [g for g in groups.values() if g["encoders"] and g["decoders"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--out", type=Path, help="default: <bundle>/out/<timestamp>")
    parser.add_argument("--eps", nargs="+", choices=["qnn", "cpu"], default=["qnn", "cpu"], help="latency runs")
    parser.add_argument("--chain-eps", nargs="*", choices=["qnn", "cpu"], default=["qnn"], help="PSNR chain runs")
    parser.add_argument("--only", help="regex on model ids; others are skipped")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per step")
    parser.add_argument("--perf-mode", default="burst")
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--fp32-on-qnn", action="store_true", help="also try FP32 models on the QNN EP")
    parser.add_argument("--save-recon", choices=["none", "diag", "all"], default="diag")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    args.fp32_on_qnn = args.fp32_on_qnn or manifest.get("fp32_on_qnn", False)
    out_dir = (args.out or bundle / "out" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s;%(levelname)s;%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run_all.log")],
    )
    eps = [ep for ep in args.eps if ep != "qnn" or "QNNExecutionProvider" in ort.get_available_providers()]
    if eps != args.eps:
        logger.warning("QNNExecutionProvider is not available in this onnxruntime build; skipping qnn")
    chain_eps = [ep for ep in args.chain_eps if ep != "qnn" or "qnn" in eps]

    summary = {
        "bundle": manifest["name"],
        "manifest": manifest,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sysinfo": sysinfo(),
        "thermal_start": thermal_snapshot(),
        "models": [],
        "chains": [],
    }

    def flush() -> None:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    flush()
    for entry in manifest["models"]:
        if args.only and not re.search(args.only, entry["id"]):
            continue
        summary["models"].append(measure_model(entry, bundle, out_dir, eps, args))
        flush()
    for group in chain_groups(manifest):
        if args.only and not re.search(args.only, f"{group['artifact']}/{group['precision']}"):
            continue
        for ep in chain_eps:
            if ep == "qnn" and group["precision"] == "fp32" and not args.fp32_on_qnn:
                continue
            try:
                summary["chains"].append(run_chain(group, bundle, out_dir, ep, args))
            except Exception as error:  # noqa: BLE001 - keep measuring the remaining groups
                logger.exception("chain %s/%s on %s failed", group["artifact"], group["precision"], ep)
                summary["chains"].append({"artifact": group["artifact"], "precision": group["precision"],
                                          "ep": ep, "error": f"{type(error).__name__}: {error}"})  # fmt: skip
            flush()
    summary["thermal_end"] = thermal_snapshot()
    flush()
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()

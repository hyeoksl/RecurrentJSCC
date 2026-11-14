"""Collect benchmark results for a bundle and parse them into results/npu/<bundle>/<label>/.

Sources (pick one):
  --from TAR      results tarball brought back by tools/relay.sh (default: incoming/<bundle>_results.tar.gz)
  --run           BOARD_SSH is reachable: run bench/run_all.py on the board now and fetch the results
  --server-ref    run the same bench on this machine's CPU EP (the reference the board is compared with)

Raw outputs (.npy, logs, context models) go to artifacts/npu_raw/<bundle>/<label>/ (not committed).
Parsed tables: latency.csv, psnr.csv, thermal.csv, placement.json, sysinfo.json, summary.json.
PSNR is recomputed here from the saved reconstructions; the board's own numbers are kept alongside.
"""

import argparse
import csv
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.chain import psnr  # noqa: E402

logger = logging.getLogger("collect_npu")
SERVER_LABEL = "server_cpu"


def run_on_server(bundle: str, raw_root: Path, extra: list[str]) -> Path:
    out = raw_root / SERVER_LABEL
    if out.exists():
        shutil.rmtree(out)
    script = ROOT / "artifacts" / "bundles" / bundle / "bench" / "run_all.py"
    command = [sys.executable, str(script), "--eps", "cpu", "--chain-eps", "cpu", "--out", str(out), *extra]
    subprocess.run(command, check=True)
    return out


def run_on_board(bundle: str, raw_root: Path, extra: list[str]) -> Path:
    board_ssh = os.environ.get("BOARD_SSH")
    project = os.environ.get("BOARD_PROJECT")
    if not board_ssh or not project:
        raise SystemExit("--run needs BOARD_SSH and BOARD_PROJECT (the uv project path on the board)")
    ssh = shlex.split(board_ssh)
    workdir = os.environ.get("BOARD_WORKDIR", "jscc_bench")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote = (
        f"bundle=$(cd {shlex.quote(workdir)}/{shlex.quote(bundle)} && pwd) && cd {shlex.quote(project)} && "
        f'uv run python "$bundle/bench/run_all.py" --out "$bundle/out/{stamp}" {shlex.join(extra)}'
    )
    status = subprocess.run([*ssh, remote]).returncode
    logger.info("run_all.py on the board exited with %s", status)
    tarball = ROOT / "incoming" / f"{bundle}_results.tar.gz"
    tarball.parent.mkdir(exist_ok=True)
    fetch = f"tar czf - --exclude='*.bin' -C {shlex.quote(workdir)}/{shlex.quote(bundle)}/out {stamp}"
    with tarball.open("wb") as stream:
        subprocess.run([*ssh, fetch], stdout=stream, check=True)
    return extract(tarball, raw_root)


def extract(tarball: Path, raw_root: Path) -> Path:
    """Unpack a results tarball (one top-level <stamp>/ directory) under raw_root."""
    with tarfile.open(tarball) as tar:
        tops = {member.name.split("/")[0] for member in tar.getmembers()}
        if len(tops) != 1:
            raise ValueError(f"{tarball}: expected one top-level directory, found {sorted(tops)}")
        raw_root.mkdir(parents=True, exist_ok=True)
        tar.extractall(raw_root, filter="data")
    return raw_root / tops.pop()


def step_fields(step: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if step is None:
        return {}
    latency = step.get("latency_ms", {})
    return {
        f"{prefix}_ok": step.get("ok"),
        f"{prefix}_create_s": step.get("create_s"),
        f"{prefix}_median_ms": latency.get("median"),
        f"{prefix}_p95_ms": latency.get("p95"),
        f"{prefix}_peak_rss_mb": step["peak_rss_kb"] / 1024 if step.get("peak_rss_kb") else None,
        f"{prefix}_error": (step.get("error") or "")[:300],
    }


def profile_providers(path: Path) -> dict[str, list[str]]:
    """provider -> node names, from an onnxruntime profiling JSON."""
    providers: dict[str, set[str]] = {}
    for event in json.loads(path.read_text()):
        args = event.get("args") or {}
        if event.get("cat") == "Node" and "provider" in args:
            name = event["name"].removesuffix("_kernel_time")
            providers.setdefault(args["provider"], set()).add(f"{args.get('op_name', '?')}:{name}")
    return {provider: sorted(nodes) for provider, nodes in providers.items()}


def placement(model: dict[str, Any], raw: Path) -> dict[str, Any]:
    """Where the nodes ran: context-model contents, ORT profile, strict-mode outcome, log excerpts."""
    info: dict[str, Any] = {"log_lines": model.get("placement_log", {}).get("lines", [])}
    steps = model["steps"]
    if "qnn_strict" in steps:
        info["runs_without_cpu_fallback"] = steps["qnn_strict"].get("ok")
        info["strict_error"] = steps["qnn_strict"].get("error")
    ctx = raw / "ctx" / f"{model['id'].replace('/', '__')}_ctx.onnx"
    if ctx.exists():
        nodes = onnx.load(str(ctx), load_external_data=False).graph.node
        outside = [f"{node.op_type}:{node.name}" for node in nodes if node.op_type != "EPContext"]
        info["context_model"] = {
            "epcontext_nodes": sum(node.op_type == "EPContext" for node in nodes),
            "nodes_outside_qnn": outside,
            "ops_outside_qnn": dict(Counter(node.split(":")[0] for node in outside)),
        }
    profile = steps.get("qnn_placement", {}).get("ort_profile")
    if profile and (raw / "steps" / Path(profile).name).exists():
        providers = profile_providers(raw / "steps" / Path(profile).name)
        info["profile_nodes_per_provider"] = {provider: len(nodes) for provider, nodes in providers.items()}
        info["profile_cpu_nodes"] = providers.get("CPUExecutionProvider", [])
    return info


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_psnr(results_root: Path) -> dict[tuple[str, int, int], float]:
    """(artifact, enc_k, dec_k) -> FP32 PSNR from the server reference run, if it exists."""
    path = results_root / SERVER_LABEL / "psnr.csv"
    if not path.exists():
        return {}
    with path.open() as file:
        rows = [row for row in csv.DictReader(file) if row["precision"] == "fp32"]
    return {(r["artifact"], int(r["enc_k"]), int(r["dec_k"])): float(r["psnr_recomputed"] or r["psnr"]) for r in rows}


def parse(raw: Path, results: Path, results_root: Path, bundle: str) -> None:
    summary = json.loads((raw / "summary.json").read_text())
    results.mkdir(parents=True, exist_ok=True)
    (results / "sysinfo.json").write_text(json.dumps(summary["sysinfo"], indent=1))
    for model in summary["models"]:
        for step in model["steps"].values():
            step.pop("latencies_ms", None)
    (results / "summary.json").write_text(json.dumps(summary, indent=1))

    latency_rows, thermal_rows, placements = [], [], {}
    for model in summary["models"]:
        steps = model["steps"]
        context = model.get("context_files", {})
        row = {
            "id": model["id"],
            "artifact": model["artifact"],
            "precision": model["precision"],
            "role": model["role"],
            "k": model["k"],
            "onnx_bytes": model["file_bytes"],
            "context_bin_bytes": sum(size for name, size in context.items() if name.endswith(".bin")) or None,
            "context_onnx_bytes": sum(size for name, size in context.items() if name.endswith(".onnx")) or None,
            **step_fields(steps.get("qnn_bench"), "qnn"),
            **step_fields(steps.get("qnn_reload"), "qnn_reload"),
            **step_fields(steps.get("cpu_bench"), "cpu"),
        }
        if "qnn_bench" in steps:
            placements[model["id"]] = placement(model, raw)
            outside = placements[model["id"]].get("context_model", {}).get("nodes_outside_qnn")
            row["nodes_outside_qnn"] = None if outside is None else len(outside)
            row["runs_without_cpu_fallback"] = placements[model["id"]].get("runs_without_cpu_fallback")
        latency_rows.append(row)
        for name, step in steps.items():
            before, after = step.get("thermal_before", {}), step.get("thermal_after", {})
            thermal_rows.append(
                {
                    "id": model["id"],
                    "step": name,
                    "max_temp_before_c": before.get("max_temp_c"),
                    "max_temp_after_c": after.get("max_temp_c"),
                    "throttled_before": before.get("throttled"),
                    "throttled_after": after.get("throttled"),
                }
            )
    write_csv(results / "latency.csv", latency_rows)
    write_csv(results / "thermal.csv", thermal_rows)
    (results / "placement.json").write_text(json.dumps(placements, indent=1))

    reference = reference_psnr(results_root)
    images = {
        name: np.load(ROOT / "artifacts" / "bundles" / bundle / bench["images"])
        for name, bench in summary["manifest"].get("bench", {}).items()
        if (ROOT / "artifacts" / "bundles" / bundle / bench["images"]).exists()
    }
    psnr_rows = []
    for chain in summary["chains"]:
        for pair in chain.get("pairs", []):
            recon = raw / chain["outputs"] / f"recon_e{pair['enc_k']}_d{pair['dec_k']}.npy"
            recomputed = None
            if recon.exists() and chain["artifact"] in images:
                recomputed = float(psnr(images[chain["artifact"]], np.load(recon).astype(np.float32)).mean())
            fp32 = reference.get((chain["artifact"], pair["enc_k"], pair["dec_k"]))
            value = recomputed if recomputed is not None else pair["psnr_mean"]
            psnr_rows.append(
                {
                    "artifact": chain["artifact"],
                    "precision": chain["precision"],
                    "ep": chain["ep"],
                    "enc_k": pair["enc_k"],
                    "dec_k": pair["dec_k"],
                    "snr_db": chain["snr_db"],
                    "active_channels": chain["active_channels"],
                    "num_images": chain["num_images"],
                    "psnr": pair["psnr_mean"],
                    "psnr_recomputed": recomputed,
                    "psnr_fp32_server": fp32,
                    "drop_vs_fp32_db": None if fp32 is None else fp32 - value,
                }
            )
    write_csv(results / "psnr.csv", psnr_rows)
    failed = [c for c in summary["chains"] if "error" in c]
    logger.info("Parsed %d models, %d PSNR rows (%d failed chains) into %s",
                len(latency_rows), len(psnr_rows), len(failed), results)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", help="bundle name (artifacts/bundles/<bundle>)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from", dest="tarball", type=Path)
    source.add_argument("--run", action="store_true")
    source.add_argument("--server-ref", action="store_true")
    parser.add_argument("--label", help="results subdirectory (default: date of the run)")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER, help="after --, passed to bench/run_all.py")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
    extra = [arg for arg in args.bench_args if arg != "--"]

    raw_root = ROOT / "artifacts" / "npu_raw" / args.bundle
    results_root = ROOT / "results" / "npu" / args.bundle
    if args.server_ref:
        raw, label = run_on_server(args.bundle, raw_root, extra), SERVER_LABEL
    else:
        if args.run:
            raw = run_on_board(args.bundle, raw_root, extra)
        else:
            raw = extract(args.tarball or ROOT / "incoming" / f"{args.bundle}_results.tar.gz", raw_root)
        label = args.label or raw.name
    parse(raw, results_root / label, results_root, args.bundle)


if __name__ == "__main__":
    main()

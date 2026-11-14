"""Pack one or more artifacts into a bundle for the board: models + bench inputs + bench/ scripts.

  artifacts/bundles/<name>/            staging directory (also runnable on this machine)
  artifacts/bundles/<name>.tar.gz      what goes to the board

Every model in the bundle is loaded and run with this machine's onnxruntime (pinned to the board's
version) before packing. With --send and BOARD_SSH set, the tarball is streamed to the board; if the
board is unreachable the bundle is left for tools/relay.sh.
"""

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import git_revision  # noqa: E402

logger = logging.getLogger("make_bundle")
ROOT = Path(__file__).resolve().parent.parent
BUNDLES = ROOT / "artifacts" / "bundles"


def check_runs(path: Path) -> None:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {i.name: np.zeros(i.shape, dtype=np.float32) for i in session.get_inputs()}
    session.run(None, feeds)


def board_reachable(board_ssh: str) -> bool:
    command = [*shlex.split(board_ssh), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "true"]
    return subprocess.run(command, capture_output=True).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", type=Path, nargs="+", help="artifacts/<name> directories")
    parser.add_argument("--name", required=True, help="bundle name")
    parser.add_argument("--precisions", nargs="+", default=["fp32", "a8", "a16"])
    parser.add_argument("--fp32-on-qnn", action="store_true", help="ask the board to also try FP32 on the QNN EP")
    parser.add_argument("--send", action="store_true", help="stream the bundle to the board via $BOARD_SSH")
    parser.add_argument("--board-dir", default=os.environ.get("BOARD_WORKDIR", "jscc_bench"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

    stage = BUNDLES / args.name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bench").mkdir(parents=True)
    for script in sorted((ROOT / "bench").glob("*.py")):
        shutil.copy2(script, stage / "bench" / script.name)

    manifest = {
        "name": args.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
        "onnxruntime": ort.__version__,
        "fp32_on_qnn": args.fp32_on_qnn,
        "models": [],
        "bench": {},
        "exports": {},
    }
    for artifact in args.artifacts:
        meta = json.loads((artifact / "onnx" / "export.json").read_text())
        name = meta["name"]
        manifest["exports"][name] = {k: meta[k] for k in ("weights", "random_init", "overrides", "static_hw", "git")}
        manifest["exports"][name]["model"] = meta["config"].get("model")
        for model in meta["models"]:
            for precision in args.precisions:
                source = artifact / ("onnx" if precision == "fp32" else f"qdq/{precision}") / model["path"]
                if not source.exists():
                    raise FileNotFoundError(f"{source} is missing; run tools/quantize.py first")
                check_runs(source)
                relative = Path("models") / name / precision / model["path"]
                (stage / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, stage / relative)
                manifest["models"].append(
                    {
                        "id": f"{name}/{precision}/{Path(model['path']).stem}",
                        "artifact": name,
                        "precision": precision,
                        "role": model["role"],
                        "k": model["iters"],
                        "path": str(relative),
                        "bytes": source.stat().st_size,
                    }
                )
        bench_meta = artifact / "bench" / "bench.json"
        if bench_meta.exists():
            inputs = stage / "inputs" / name
            inputs.mkdir(parents=True)
            for file in ("images.npy", "noise.npy"):
                shutil.copy2(artifact / "bench" / file, inputs / file)
            manifest["bench"][name] = {
                **json.loads(bench_meta.read_text()),
                "images": f"inputs/{name}/images.npy",
                "noise": f"inputs/{name}/noise.npy",
            }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))

    tarball = BUNDLES / f"{args.name}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(stage, arcname=args.name)
    logger.info("Bundle: %s (%d models, %.1f MB)", tarball, len(manifest["models"]), tarball.stat().st_size / 1e6)

    if not args.send:
        return
    board_ssh = os.environ.get("BOARD_SSH")
    if not board_ssh or not board_reachable(board_ssh):
        logger.warning("Board not reachable through BOARD_SSH; run tools/relay.sh %s on the local PC", args.name)
        return
    remote = f"mkdir -p {shlex.quote(args.board_dir)} && tar xzf - -C {shlex.quote(args.board_dir)}"
    with tarball.open("rb") as stream:
        subprocess.run([*shlex.split(board_ssh), remote], stdin=stream, check=True)
    logger.info("Sent to the board: %s/%s", args.board_dir, args.name)


if __name__ == "__main__":
    main()

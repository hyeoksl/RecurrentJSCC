"""Paper figures and tables. Reads only results/ (JSON/CSV written by evaluate.py and collect_npu.py).

Outputs go to results/figures/ (PDF + PNG) and results/tables/ (CSV). Anything whose inputs are missing
is skipped with a log line; nothing is estimated or filled in.

  sim_psnr_vs_snr_<exp>     PSNR vs SNR at one spp, one line per K (encoder K = decoder K)
  sim_psnr_vs_spp_<exp>     PSNR vs spp at one SNR, one line per K
  sim_combos_<exp>          PSNR for every (encoder K, decoder K) pair at the reference point
  sim_psnr_vs_gflops        PSNR vs total GFLOPs at the reference point, one line per experiment
  npu_latency_<bundle>_<label>   median latency vs K, one panel per (role, execution provider)
  npu_psnr_<bundle>_<label>      PSNR vs K (encoder K = decoder K) per precision
  tables/sim_summary.csv, tables/sim_mirror_<exp>.csv, tables/npu_<bundle>_<label>.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger("plots")

# Categorical slots in fixed order (assigned to an entity, never to a rank), with a marker per slot so
# identity never rests on color alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e1e0d9"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)


def save(fig: Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{suffix}")
    plt.close(fig)
    logger.info("Wrote %s", out_dir / f"{name}.pdf")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


def nearest(values: list[float], target: float | None, default: float) -> float:
    wanted = default if target is None else target
    return min(values, key=lambda value: abs(value - wanted))


def line(ax: Axes, slot: int, xs: list[float], ys: list[float], label: str) -> None:
    color, marker = SERIES[slot % len(SERIES)], MARKERS[slot % len(MARKERS)]
    ax.plot(xs, ys, color=color, marker=marker, markeredgecolor="white", markeredgewidth=0.8, label=label)


def lookup(sim: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    return {(p["enc_iters"], p["dec_iters"], p["snr_db"], p["spp"]): p for p in sim["points"]}


def sim_sweeps(sim: dict[str, Any], out_dir: Path, snr: float, spp: float) -> None:
    """PSNR vs SNR and PSNR vs spp for the symmetric pairs (K, K)."""
    points, exp = lookup(sim), sim["experiment"]
    sweeps = (
        ("snr", sim["snrs_db"], "SNR (dB)", f"spp = {spp:g}", lambda k, x: (k, k, x, spp)),
        ("spp", sim["spps"], "spp (symbols per pixel)", f"SNR = {snr:g} dB", lambda k, x: (k, k, snr, x)),
    )
    for name, xs, xlabel, fixed, key in sweeps:
        if len(xs) < 2:
            logger.info("%s: only one %s value, skipping the sweep figure", exp, name)
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for slot, k in enumerate(sim["iters"]):
            ys = [points[key(k, x)]["psnr"]["mean"] for x in xs]
            line(ax, slot, xs, ys, "single depth" if k is None else f"K = {k}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{exp}: encoder K = decoder K, {fixed}", fontsize=9, color=INK, loc="left")
        if len(sim["iters"]) > 1:
            ax.legend()
        save(fig, out_dir, f"sim_psnr_vs_{name}_{exp}")


def sim_combos(sim: dict[str, Any], out_dir: Path, snr: float, spp: float) -> None:
    """Annotated grid of PSNR over (encoder K, decoder K)."""
    iters, points, exp = sim["iters"], lookup(sim), sim["experiment"]
    if len(iters) < 2:
        return
    grid = [[points[(enc, dec, snr, spp)]["psnr"]["mean"] for dec in iters] for enc in iters]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    image = ax.imshow(grid, cmap=SEQUENTIAL, origin="lower")
    low, high = min(map(min, grid)), max(map(max, grid))
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            dark = high > low and (value - low) / (high - low) > 0.55
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if dark else INK)
    ax.set_xticks(range(len(iters)), labels=iters)
    ax.set_yticks(range(len(iters)), labels=iters)
    ax.set_xlabel("decoder K")
    ax.set_ylabel("encoder K")
    ax.grid(False)
    ax.set_title(f"{exp}: PSNR (dB), SNR = {snr:g} dB, spp = {spp:g}", fontsize=9, color=INK, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, out_dir, f"sim_combos_{exp}")


def sim_tables(sims: list[dict[str, Any]], tables: Path, snr: float | None, spp: float | None) -> list[dict]:
    """One row per experiment and (encoder K, decoder K) at that experiment's reference point."""
    rows = []
    for sim in sims:
        ref_snr, ref_spp = nearest(sim["snrs_db"], snr, 10.0), nearest(sim["spps"], spp, max(sim["spps"]))
        points = lookup(sim)
        for item in sim["complexity"]:
            point = points[(item["enc_iters"], item["dec_iters"], ref_snr, ref_spp)]
            rows.append(
                {
                    "experiment": sim["experiment"],
                    "film": sim["model"].get("film"),
                    "share_core": sim["model"].get("share_core"),
                    "train_iters": " ".join(map(str, sim.get("train_iters") or [])),
                    "enc_iters": item["enc_iters"],
                    "dec_iters": item["dec_iters"],
                    "encoder_params": item["encoder_params"],
                    "decoder_params": item["decoder_params"],
                    "encoder_gflops": item["encoder_gflops"],
                    "decoder_gflops": item["decoder_gflops"],
                    "snr_db": ref_snr,
                    "spp": ref_spp,
                    "psnr": point["psnr"]["mean"],
                    "ms_ssim": point["ms_ssim"]["mean"],
                }
            )
        write_csv(tables / f"sim_mirror_{sim['experiment']}.csv", sim["mirror_pairs"])
    write_csv(tables / "sim_summary.csv", rows)
    return rows


def sim_psnr_vs_gflops(rows: list[dict], out_dir: Path) -> None:
    """Symmetric pairs only: total GFLOPs on x, PSNR on y, one series per experiment."""
    symmetric = [row for row in rows if row["enc_iters"] == row["dec_iters"]]
    experiments = list(dict.fromkeys(row["experiment"] for row in symmetric))
    if not experiments:
        return
    if len(experiments) > len(SERIES):
        logger.warning("More than %d experiments; plotting the first %d only", len(SERIES), len(SERIES))
        experiments = experiments[: len(SERIES)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for slot, exp in enumerate(experiments):
        mine = sorted((row for row in symmetric if row["experiment"] == exp), key=lambda row: row["encoder_gflops"])
        xs = [row["encoder_gflops"] + row["decoder_gflops"] for row in mine]
        line(ax, slot, xs, [row["psnr"] for row in mine], exp)
    ax.set_xlabel("encoder + decoder GFLOPs")
    ax.set_ylabel("PSNR (dB)")
    reference = {(row["snr_db"], row["spp"]) for row in symmetric}
    if len(reference) == 1:
        ref_snr, ref_spp = reference.pop()
        ax.set_title(f"SNR = {ref_snr:g} dB, spp = {ref_spp:g}", fontsize=9, color=INK, loc="left")
    ax.legend()
    save(fig, out_dir, "sim_psnr_vs_gflops")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        return list(csv.DictReader(file))


def number(text: str | None) -> float | None:
    return float(text) if text not in (None, "") else None


def npu_latency(rows: list[dict[str, str]], out_dir: Path, tag: str) -> None:
    """Median latency vs K. QNN and CPU differ by orders of magnitude, so each gets its own panels."""
    rows = [row for row in rows if row["role"] in ("encoder", "decoder") and row["k"]]
    eps = [ep for ep in ("qnn", "cpu") if any(number(row.get(f"{ep}_median_ms")) is not None for row in rows)]
    if not eps:
        logger.info("%s: no latency measurements, skipping", tag)
        return
    series = list(dict.fromkeys((row["artifact"], row["precision"]) for row in rows))[: len(SERIES)]
    fig, axes = plt.subplots(len(eps), 2, figsize=(7.2, 2.8 * len(eps)), squeeze=False)
    for r, ep in enumerate(eps):
        for c, role in enumerate(("encoder", "decoder")):
            ax = axes[r][c]
            for slot, (artifact, precision) in enumerate(series):
                mine = sorted(
                    (
                        (int(row["k"]), number(row.get(f"{ep}_median_ms")))
                        for row in rows
                        if (row["artifact"], row["precision"], row["role"]) == (artifact, precision, role)
                    ),
                )
                mine = [(k, value) for k, value in mine if value is not None]
                if mine:
                    label = precision if len({a for a, _ in series}) == 1 else f"{artifact} {precision}"
                    line(ax, slot, [k for k, _ in mine], [value for _, value in mine], label)
            ax.set_title(f"{role}, {'QNN EP (HTP)' if ep == 'qnn' else 'CPU EP'}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("K")
            ax.set_xticks(sorted({int(row["k"]) for row in rows}))
            ax.set_ylabel("median latency (ms)")
            ax.set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out_dir, f"npu_latency_{tag}")


def npu_psnr(rows: list[dict[str, str]], out_dir: Path, tag: str) -> None:
    rows = [row for row in rows if row["enc_k"] == row["dec_k"]]
    series = list(dict.fromkeys((row["artifact"], row["precision"], row["ep"]) for row in rows))[: len(SERIES)]
    if not series:
        logger.info("%s: no PSNR chain results, skipping", tag)
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for slot, key in enumerate(series):
        mine = sorted(
            (int(row["enc_k"]), number(row["psnr_recomputed"]) or float(row["psnr"]))
            for row in rows
            if (row["artifact"], row["precision"], row["ep"]) == key
        )
        label = f"{key[1]} ({key[2]})" if len({a for a, _, _ in series}) == 1 else " ".join(key)
        line(ax, slot, [k for k, _ in mine], [value for _, value in mine], label)
    ax.set_xlabel("K (encoder = decoder)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{tag}: SNR = {float(rows[0]['snr_db']):g} dB", fontsize=9, color=INK, loc="left")
    ax.legend()
    save(fig, out_dir, f"npu_psnr_{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--snr", type=float, help="reference SNR in dB (default: nearest to 10)")
    parser.add_argument("--spp", type=float, help="reference spp (default: the largest evaluated)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")
    figures, tables = args.results / "figures", args.results / "tables"

    sims = [json.loads(path.read_text()) for path in sorted((args.results / "sim").glob("*.json"))]
    if not sims:
        logger.info("No results/sim/*.json yet; skipping simulation figures")
    for sim in sims:
        snr, spp = nearest(sim["snrs_db"], args.snr, 10.0), nearest(sim["spps"], args.spp, max(sim["spps"]))
        sim_sweeps(sim, figures, snr, spp)
        sim_combos(sim, figures, snr, spp)
    sim_psnr_vs_gflops(sim_tables(sims, tables, args.snr, args.spp), figures)

    runs = sorted(path.parent for path in (args.results / "npu").glob("*/*/latency.csv"))
    if not runs:
        logger.info("No results/npu/<bundle>/<label>/latency.csv yet; skipping NPU figures")
    for run in runs:
        tag = f"{run.parent.name}_{run.name}"
        latency = read_csv(run / "latency.csv")
        npu_latency(latency, figures, tag)
        write_csv(tables / f"npu_{tag}.csv", latency)
        if (run / "psnr.csv").exists():
            npu_psnr(read_csv(run / "psnr.csv"), figures, tag)
        if (run / "power_manual.csv").exists():
            write_csv(tables / f"npu_power_{tag}.csv", read_csv(run / "power_manual.csv"))


if __name__ == "__main__":
    main()

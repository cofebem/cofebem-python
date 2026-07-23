"""Reproducible normal-contact benchmark for tyre compliance strategies."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from cofebem.mesh.tyre_dihedral_hex import generate_tyre_mesh


MESHES = {
    "coarse": (12, 16),
    "medium": (18, 24),
    "fine": (24, 32),
}
STRATEGIES = (
    "hmatrix",
    "hmatrix_full",
    "fe_matrix_free",
    "mumps_schur",
)
LABELS = {
    "hmatrix": r"$\mathcal{H}$ (dihedral)",
    "hmatrix_full": r"$\mathcal{H}$ (no dihedral)",
    "fe_matrix_free": r"factorized $K$",
    "mumps_schur": "MUMPS Schur",
}
STAGE_FIELDS = (
    "floor_build",
    "factorization",
    "inflation_solve",
    "compliance_sampling",
    "operator_build",
    "contact_solve",
    "verification",
    "final_solve",
    "pressure_postprocess",
)


def _scalar(archive, name, default=0.0, cast=float):
    if name not in archive.files:
        return default
    return cast(np.asarray(archive[name]).reshape(-1)[0])


def _child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _write_motion(path: Path) -> None:
    payload = {
        "time": [0.0, 1.0],
        "interval_steps": [9],
        "indentation": [0.001, 0.01],
        "floor_rotation_y_deg": 0.0,
        "floor_rotation_z_deg": 0.0,
        "floor_translation_x": 0.0,
        "floor_translation_y": 0.0,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_case(
    *,
    root: Path,
    result_root: Path,
    mesh_name: str,
    axial: int,
    circumferential: int,
    mesh_path: Path,
    history: str,
    strategy: str,
    repetition: int,
) -> dict[str, object]:
    run_dir = result_root / mesh_name / history / strategy / f"repeat_{repetition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    motion_path = result_root / "ten_load_states.json"
    command = [
        sys.executable,
        str(root / "examples" / "tyre_dihedral_contact.py"),
        "--mesh",
        str(mesh_path),
        "--output-dir",
        str(run_dir),
        "--axial-divisions",
        str(axial),
        "--circumferential-divisions",
        str(circumferential),
        "--compliance-strategy",
        strategy,
        "--factor-solver-type",
        "mumps",
        "--indentation",
        "0.01",
        "--floor",
        "flat",
        "--floor-grid-size",
        "32",
        "--warning-distance",
        "0.02",
        "--h-leaf-size",
        "8",
        "--h-eta",
        "1.0",
        "--h-tol",
        "1e-7",
        "--h-max-rank",
        "40",
        "--contact-solver",
        "ppcg",
        "--tol",
        "1e-10",
        "--max-iter",
        "5000",
        "--stress-recovery",
        "raw",
        "--no-write-vtk",
        "--no-save-volume-fields",
        "--no-progress",
    ]
    if history == "ten":
        command.extend(["--motion-file", str(motion_path)])

    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    cpu_start = _child_cpu_seconds()
    wall_start = perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    total_wall = perf_counter() - wall_start
    total_cpu = _child_cpu_seconds() - cpu_start
    (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"benchmark failed for {mesh_name}/{history}/{strategy}/"
            f"repeat {repetition}; see {run_dir / 'run.log'}"
        )

    metric_path = (
        run_dir / "motion_history.npz"
        if history == "ten"
        else run_dir / f"contact_result_{strategy}.npz"
    )
    final_path = run_dir / f"contact_result_{strategy}.npz"
    with np.load(metric_path, allow_pickle=False) as metrics:
        record: dict[str, object] = {
            "mesh": mesh_name,
            "axial_divisions": axial,
            "circumferential_divisions": circumferential,
            "history": history,
            "load_states": 10 if history == "ten" else 1,
            "strategy": strategy,
            "repetition": repetition,
            "external_total_wall_seconds": total_wall,
            "external_total_cpu_seconds": total_cpu,
            "peak_rss_bytes": _scalar(metrics, "peak_rss_bytes", cast=int),
            "fe_unknowns": _scalar(metrics, "fe_unknowns", cast=int),
            "volume_cells": _scalar(metrics, "volume_cells", cast=int),
            "potential_contact_unknowns": _scalar(
                metrics, "potential_contact_unknowns", cast=int
            ),
            "compliance_stored_entries": _scalar(
                metrics, "compliance_stored_entries", cast=int
            ),
            "compliance_sampling_solves": _scalar(
                metrics, "compliance_sampling_solves", cast=int
            ),
            "contact_iterations": _scalar(
                metrics,
                "total_contact_iterations" if history == "ten" else "iterations",
                cast=int,
            ),
            "factorization_count": _scalar(
                metrics, "factorization_count", cast=int
            ),
            "final_result": str(final_path),
        }
        for stage in STAGE_FIELDS:
            record[f"{stage}_wall_seconds"] = _scalar(
                metrics, f"{stage}_seconds"
            )
            record[f"{stage}_cpu_seconds"] = _scalar(
                metrics, f"{stage}_cpu_seconds"
            )
    return record


def _median_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record["mesh"]), str(record["history"]), str(record["strategy"]))
        grouped.setdefault(key, []).append(record)
    summaries = []
    for key, group in grouped.items():
        summary = dict(group[0])
        summary["repetition"] = "median"
        for field in group[0]:
            values = [item[field] for item in group]
            if field not in {
                "mesh",
                "history",
                "strategy",
                "repetition",
                "final_result",
            } and all(isinstance(value, (int, float)) for value in values):
                summary[field] = float(np.median(values))
        summaries.append(summary)
    mesh_order = {name: index for index, name in enumerate(MESHES)}
    strategy_order = {name: index for index, name in enumerate(STRATEGIES)}
    summaries.sort(
        key=lambda item: (
            mesh_order[str(item["mesh"])],
            0 if item["history"] == "one" else 1,
            strategy_order[str(item["strategy"])],
        )
    )
    return summaries


def _add_accuracy(summaries: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for summary in summaries:
        grouped.setdefault(
            (str(summary["mesh"]), str(summary["history"])), []
        ).append(summary)
    for group in grouped.values():
        reference = next(
            item for item in group if item["strategy"] == "fe_matrix_free"
        )
        with np.load(str(reference["final_result"]), allow_pickle=False) as archive:
            reference_force = np.asarray(archive["force"], dtype=float)
            reference_clearance = np.asarray(archive["clearance"], dtype=float)
            reference_gap = np.asarray(archive["gap"], dtype=float)
            reference_candidates = np.asarray(
                archive["candidate_indices"], dtype=np.int64
            )
        force_norm = max(np.linalg.norm(reference_force), np.finfo(float).tiny)
        active_tolerance = max(
            float(np.max(reference_force)) * 1.0e-10,
            np.finfo(float).eps,
        )
        reference_active = reference_force > active_tolerance
        for item in group:
            with np.load(str(item["final_result"]), allow_pickle=False) as archive:
                force = np.asarray(archive["force"], dtype=float)
                clearance = np.asarray(archive["clearance"], dtype=float)
                gap = np.asarray(archive["gap"], dtype=float)
                candidates = np.asarray(
                    archive["candidate_indices"], dtype=np.int64
                )
            if force.shape != reference_force.shape:
                raise ValueError("strategy force vectors use different orderings")
            if not np.array_equal(candidates, reference_candidates):
                raise ValueError("strategy potential-contact sets do not match")
            if not np.allclose(gap, reference_gap, rtol=0.0, atol=1.0e-13):
                raise ValueError("strategy initial gaps do not match")
            item["primal_violation"] = float(
                max(0.0, -float(np.min(force)))
            )
            item["dual_violation"] = float(
                max(0.0, -float(np.min(clearance)))
            )
            item["complementarity"] = float(
                abs(np.dot(force, clearance))
            )
            item["active_set_difference"] = int(
                np.count_nonzero(
                    (force > active_tolerance) != reference_active
                )
            )
            item["force_relative_l2"] = float(
                np.linalg.norm(force - reference_force) / force_norm
            )
            item["clearance_linf"] = float(
                np.linalg.norm(clearance - reference_clearance, ord=np.inf)
            )


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = sorted({field for record in records for field in record})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_latex(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        r"% Generated by examples/benchmark_tyre_contact_strategies.py",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrr}",
        r"\toprule",
        r"Mesh & Method & DOFs & $N_c$ & Peak [MiB] & FE setup & Compliance & H build & Contact & Recovery & Total wall \\",
        r" & & & & & \multicolumn{5}{c}{CPU time [s]} & [s] \\",
        r"\midrule",
    ]
    previous = None
    for item in summaries:
        group = (item["history"], item["mesh"])
        if previous is not None and group != previous:
            lines.append(r"\midrule")
        mesh_label = f"{item['mesh']} ({int(item['load_states'])} states)"
        fe_setup_cpu = float(item["factorization_cpu_seconds"]) + float(
            item["inflation_solve_cpu_seconds"]
        )
        compliance_cpu = float(item["compliance_sampling_cpu_seconds"])
        contact_cpu = float(item["contact_solve_cpu_seconds"]) + float(
            item["verification_cpu_seconds"]
        )
        recovery_cpu = float(item["final_solve_cpu_seconds"]) + float(
            item["pressure_postprocess_cpu_seconds"]
        )
        lines.append(
            f"{mesh_label} & {LABELS[str(item['strategy'])]} & "
            f"{int(item['fe_unknowns'])} & "
            f"{int(item['potential_contact_unknowns'])} & "
            f"{float(item['peak_rss_bytes']) / 2**20:.1f} & "
            f"{fe_setup_cpu:.3f} & "
            f"{compliance_cpu:.3f} & "
            f"{float(item['operator_build_cpu_seconds']):.3f} & "
            f"{contact_cpu:.3f} & "
            f"{recovery_cpu:.3f} & "
            f"{float(item['external_total_wall_seconds']):.3f} \\\\"
        )
        previous = group
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\caption{Normal tyre contact benchmark. Times are medians of single-threaded runs; mesh generation is excluded. Compliance is symmetry sampling or exact ACA-requested FE sampling, depending on the H-matrix variant.}",
            r"\label{tab:tyre-contact-strategies}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, summaries: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    colors = dict(zip(STRATEGIES, ("C0", "C1", "C2", "C3")))
    markers = dict(zip(STRATEGIES, ("o", "s", "^", "D")))
    for column, history in enumerate(("one", "ten")):
        selected = [item for item in summaries if item["history"] == history]
        for strategy in STRATEGIES:
            values = [item for item in selected if item["strategy"] == strategy]
            x = np.array([item["fe_unknowns"] for item in values], dtype=float)
            wall = np.array(
                [item["external_total_wall_seconds"] for item in values]
            )
            memory = np.array([item["peak_rss_bytes"] for item in values]) / 2**20
            axes[0, column].plot(
                x,
                wall,
                marker=markers[strategy],
                color=colors[strategy],
                label=LABELS[strategy],
            )
            axes[1, column].plot(
                x,
                memory,
                marker=markers[strategy],
                color=colors[strategy],
            )
        axes[0, column].set_title(
            "one contact state" if history == "one" else "ten load states"
        )
        axes[0, column].set_ylabel("total wall time [s]")
        axes[1, column].set_ylabel("peak RSS [MiB]")
        axes[1, column].set_xlabel("finite-element displacement DOFs")
        for row in range(2):
            axes[row, column].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _write_stage_plot(path: Path, summaries: list[dict[str, object]]) -> None:
    """Plot the fine-mesh CPU-time decomposition for both histories."""
    stage_groups = (
        (
            "FE setup",
            ("factorization_cpu_seconds", "inflation_solve_cpu_seconds"),
        ),
        (
            "compliance",
            ("compliance_sampling_cpu_seconds", "operator_build_cpu_seconds"),
        ),
        ("contact", ("contact_solve_cpu_seconds", "verification_cpu_seconds")),
        (
            "recovery",
            ("final_solve_cpu_seconds", "pressure_postprocess_cpu_seconds"),
        ),
    )
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for axis, history in zip(axes, ("one", "ten")):
        values = [
            item
            for item in summaries
            if item["mesh"] == "fine" and item["history"] == history
        ]
        x = np.arange(len(values))
        bottom = np.zeros(len(values))
        for (label, fields), color in zip(stage_groups, colors):
            height = np.array(
                [sum(float(item[field]) for field in fields) for item in values]
            )
            axis.bar(x, height, bottom=bottom, label=label, color=color)
            bottom += height
        axis.set_xticks(x, [LABELS[str(item["strategy"])] for item in values])
        axis.tick_params(axis="x", rotation=20)
        axis.set_title("one contact state" if history == "one" else "ten load states")
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("measured stage CPU time [s]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "tyre_strategy_benchmark",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--regenerate-meshes", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate tables/plots from an existing records.json",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    result_root = args.output_dir.expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    _write_motion(result_root / "ten_load_states.json")

    if args.report_only:
        records = json.loads(
            (result_root / "records.json").read_text(encoding="utf-8")
        )
    else:
        records = []

        meshes = {}
        for name, (axial, circumferential) in MESHES.items():
            mesh_path = result_root / "meshes" / f"tyre_{name}.msh"
            if args.regenerate_meshes or not mesh_path.is_file():
                generate_tyre_mesh(
                    root / "geo_files" / "geometry_v2.geo",
                    mesh_path,
                    axial_divisions=axial,
                    circumferential_divisions=circumferential,
                    circumferential_layout="uniform",
                    coarsening_factor=1.0,
                    scale=0.001,
                )
            meshes[name] = mesh_path

        total_runs = len(MESHES) * 2 * len(STRATEGIES) * args.repetitions
        run_number = 0
        for repetition in range(1, args.repetitions + 1):
            for mesh_name, (axial, circumferential) in MESHES.items():
                for history in ("one", "ten"):
                    order = list(STRATEGIES)
                    if repetition % 2 == 0:
                        order.reverse()
                    for strategy in order:
                        run_number += 1
                        print(
                            f"[{run_number}/{total_runs}] {mesh_name}, {history}, "
                            f"{strategy}, repeat {repetition}",
                            flush=True,
                        )
                        record = _run_case(
                            root=root,
                            result_root=result_root,
                            mesh_name=mesh_name,
                            axial=axial,
                            circumferential=circumferential,
                            mesh_path=meshes[mesh_name],
                            history=history,
                            strategy=strategy,
                            repetition=repetition,
                        )
                        records.append(record)
                        (result_root / "records.json").write_text(
                            json.dumps(records, indent=2) + "\n", encoding="utf-8"
                        )

    summaries = _median_records(records)
    _add_accuracy(summaries)
    _write_csv(result_root / "benchmark_runs.csv", records)
    _write_csv(result_root / "benchmark_summary.csv", summaries)
    _write_latex(result_root / "benchmark_table.tex", summaries)
    _write_plot(result_root / "benchmark_scaling", summaries)
    _write_stage_plot(result_root / "benchmark_stage_cpu", summaries)
    (result_root / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote benchmark report under {result_root}")


if __name__ == "__main__":
    main()

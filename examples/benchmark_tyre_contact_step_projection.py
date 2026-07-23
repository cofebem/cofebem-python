"""Project one, ten, and 100 tyre-contact steps from measured online cost."""

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


STRATEGIES = ("hmatrix", "fe_matrix_free", "mumps_schur")
LABELS = {
    "hmatrix": r"$\mathcal{H}$ (dihedral)",
    "fe_matrix_free": r"factorized $K$",
    "mumps_schur": "MUMPS Schur",
}
STEP_COUNTS = (1, 10, 100)
SETUP_STAGES = (
    "factorization",
    "inflation_solve",
    "compliance_sampling",
    "operator_build",
)
ONLINE_STAGES = (
    "contact_solve",
    "verification",
    "final_solve",
    "pressure_postprocess",
)


def _method_label(strategy: str, item: dict[str, object]) -> str:
    if strategy == "hmatrix" and item.get("circumferential_layout") == "graded":
        return r"$\mathcal{H}$ (local symmetry)"
    return LABELS[strategy]


def _scalar(archive, name, default=0.0, cast=float):
    if name not in archive.files:
        return default
    return cast(np.asarray(archive[name]).reshape(-1)[0])


def _child_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _run_case(
    *,
    root: Path,
    output_root: Path,
    mesh_path: Path,
    axial: int,
    circumferential: int,
    circumferential_layout: str,
    coarsening_factor: float,
    warning_distance: float,
    h_leaf_size: int,
    h_eta: float,
    h_max_rank: int,
    strategy: str,
    repetition: int,
) -> dict[str, object]:
    run_dir = output_root / strategy / f"repeat_{repetition}"
    run_dir.mkdir(parents=True, exist_ok=True)
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
        "--circumferential-layout",
        circumferential_layout,
        "--coarsening-factor",
        str(coarsening_factor),
        "--compliance-strategy",
        strategy,
        "--factor-solver-type",
        "mumps",
        "--indentation",
        "0.01",
        "--floor",
        "flat",
        "--floor-grid-size",
        "64",
        "--warning-distance",
        str(warning_distance),
        "--h-leaf-size",
        str(h_leaf_size),
        "--h-eta",
        str(h_eta),
        "--h-tol",
        "1e-7",
        "--h-max-rank",
        str(h_max_rank),
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
    if strategy == "hmatrix" and circumferential_layout == "graded":
        command.extend(
            [
                "--local-symmetry-tag",
                "204",
                "--local-symmetry-validation-columns",
                "2",
                "--local-symmetry-tolerance",
                "0.05",
            ]
        )
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"

    wall_start = perf_counter()
    cpu_start = _child_cpu_seconds()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    external_wall = perf_counter() - wall_start
    external_cpu = _child_cpu_seconds() - cpu_start
    (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"benchmark failed for {strategy}, repetition {repetition}; "
            f"see {run_dir / 'run.log'}"
        )

    result_path = run_dir / f"contact_result_{strategy}.npz"
    with np.load(result_path, allow_pickle=False) as archive:
        record: dict[str, object] = {
            "strategy": strategy,
            "repetition": repetition,
            "axial_divisions": axial,
            "circumferential_divisions": circumferential,
            "circumferential_layout": circumferential_layout,
            "coarsening_factor": coarsening_factor,
            "warning_distance": warning_distance,
            "h_leaf_size": h_leaf_size,
            "h_eta": h_eta,
            "h_max_rank": h_max_rank,
            "volume_cells": _scalar(archive, "volume_cells", cast=int),
            "fe_unknowns": _scalar(archive, "fe_unknowns", cast=int),
            "global_contact_unknowns": _scalar(
                archive, "global_contact_unknowns", cast=int
            ),
            "potential_contact_unknowns": _scalar(
                archive, "potential_contact_unknowns", cast=int
            ),
            "iterations": _scalar(archive, "iterations", cast=int),
            "compliance_sampling_solves": _scalar(
                archive, "compliance_sampling_solves", cast=int
            ),
            "compliance_stored_entries": _scalar(
                archive, "compliance_stored_entries", cast=int
            ),
            "fe_operator_applications": _scalar(
                archive, "fe_operator_applications", cast=int
            ),
            "fe_linear_solves": _scalar(archive, "fe_linear_solves", cast=int),
            "factorization_count": _scalar(
                archive, "factorization_count", cast=int
            ),
            "peak_rss_bytes": _scalar(archive, "peak_rss_bytes", cast=int),
            "external_total_wall_seconds": external_wall,
            "external_total_cpu_seconds": external_cpu,
            "result_path": str(result_path),
        }
        for stage in SETUP_STAGES + ONLINE_STAGES:
            record[f"{stage}_wall_seconds"] = _scalar(
                archive, f"{stage}_seconds"
            )
            record[f"{stage}_cpu_seconds"] = _scalar(
                archive, f"{stage}_cpu_seconds"
            )
    record["setup_wall_seconds"] = sum(
        float(record[f"{stage}_wall_seconds"]) for stage in SETUP_STAGES
    )
    record["online_step_wall_seconds"] = sum(
        float(record[f"{stage}_wall_seconds"]) for stage in ONLINE_STAGES
    )
    record["setup_cpu_seconds"] = sum(
        float(record[f"{stage}_cpu_seconds"]) for stage in SETUP_STAGES
    )
    record["online_step_cpu_seconds"] = sum(
        float(record[f"{stage}_cpu_seconds"]) for stage in ONLINE_STAGES
    )
    return record


def _median_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = []
    for strategy in STRATEGIES:
        group = [record for record in records if record["strategy"] == strategy]
        if not group:
            raise ValueError(f"no records found for {strategy}")
        summary = dict(group[0])
        summary["repetition"] = "median"
        for field in group[0]:
            values = [item[field] for item in group]
            if field not in {"strategy", "repetition", "result_path"} and all(
                isinstance(value, (int, float)) for value in values
            ):
                summary[field] = float(np.median(values))
        summaries.append(summary)
    return summaries


def _audit_accuracy(summaries: list[dict[str, object]]) -> None:
    reference = next(
        item for item in summaries if item["strategy"] == "fe_matrix_free"
    )
    with np.load(str(reference["result_path"]), allow_pickle=False) as archive:
        reference_points = (
            np.asarray(archive["contact_points"], dtype=float)
            if "contact_points" in archive.files
            else None
        )
        reference_force = np.asarray(archive["force"], dtype=float)
        reference_clearance = np.asarray(archive["clearance"], dtype=float)
        reference_gap = np.asarray(archive["gap"], dtype=float)
        reference_candidates = np.asarray(archive["candidate_indices"])
    active_tolerance = max(
        float(np.max(reference_force)) * 1.0e-10,
        np.finfo(float).eps,
    )
    reference_active = reference_force > active_tolerance
    reference_lookup = (
        {
            tuple(np.round(point, decimals=13)): index
            for index, point in enumerate(reference_points)
        }
        if reference_points is not None
        else None
    )
    for item in summaries:
        with np.load(str(item["result_path"]), allow_pickle=False) as archive:
            points = (
                np.asarray(archive["contact_points"], dtype=float)
                if "contact_points" in archive.files
                else None
            )
            force = np.asarray(archive["force"], dtype=float)
            clearance = np.asarray(archive["clearance"], dtype=float)
            gap = np.asarray(archive["gap"], dtype=float)
            candidates = np.asarray(archive["candidate_indices"])
        same_ordering = (
            points is None
            and reference_points is None
            and force.shape == reference_force.shape
        ) or (
            points is not None
            and reference_points is not None
            and points.shape == reference_points.shape
            and np.allclose(points, reference_points, rtol=0.0, atol=1.0e-13)
        )
        if same_ordering:
            reference_indices = np.arange(force.shape[0])
            if not np.array_equal(candidates, reference_candidates):
                raise ValueError("potential-contact sets differ between strategies")
        else:
            if points is None or reference_lookup is None:
                raise ValueError(
                    "contact_points are required to compare different surfaces"
                )
            try:
                reference_indices = np.array(
                    [
                        reference_lookup[tuple(np.round(point, decimals=13))]
                        for point in points
                    ],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise ValueError(
                    "strategy contact point is absent from the reference surface"
                ) from exc
            reference_candidate_points = {
                tuple(np.round(point, decimals=13))
                for point in reference_points[reference_candidates]
            }
            candidate_points = {
                tuple(np.round(point, decimals=13)) for point in points[candidates]
            }
            if candidate_points != reference_candidate_points:
                raise ValueError("potential-contact coordinates differ")
            omitted = np.ones(reference_force.size, dtype=bool)
            omitted[reference_indices] = False
            if np.any(reference_force[omitted] > active_tolerance):
                raise ValueError("reference contact lies outside the tagged patch")
        reference_force_local = reference_force[reference_indices]
        reference_clearance_local = reference_clearance[reference_indices]
        reference_gap_local = reference_gap[reference_indices]
        reference_active_local = reference_active[reference_indices]
        local_force_norm = max(
            np.linalg.norm(reference_force_local), np.finfo(float).tiny
        )
        if not np.allclose(gap, reference_gap_local, rtol=0.0, atol=1.0e-13):
            raise ValueError("initial gaps differ between strategies")
        item["force_relative_l2"] = float(
            np.linalg.norm(force - reference_force_local) / local_force_norm
        )
        item["clearance_linf"] = float(
            np.linalg.norm(clearance - reference_clearance_local, ord=np.inf)
        )
        item["active_set_difference"] = int(
            np.count_nonzero((force > active_tolerance) != reference_active_local)
        )


def _project(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    projections = []
    for item in summaries:
        for steps in STEP_COUNTS:
            projection = {
                "strategy": item["strategy"],
                "contact_steps": steps,
                "setup_wall_seconds": item["setup_wall_seconds"],
                "online_step_wall_seconds": item["online_step_wall_seconds"],
                "projected_online_wall_seconds": (
                    steps * float(item["online_step_wall_seconds"])
                ),
                "projected_numerical_wall_seconds": (
                    float(item["setup_wall_seconds"])
                    + steps * float(item["online_step_wall_seconds"])
                ),
                "peak_rss_bytes": item["peak_rss_bytes"],
                "force_relative_l2": item["force_relative_l2"],
                "clearance_linf": item["clearance_linf"],
                "active_set_difference": item["active_set_difference"],
                "circumferential_layout": item["circumferential_layout"],
            }
            projections.append(projection)
    return projections


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = sorted({field for record in records for field in record})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_latex(path: Path, projections: list[dict[str, object]]) -> None:
    lines = [
        r"% Generated by benchmark_tyre_contact_step_projection.py",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Equivalent & Steps & Setup [s] & Per step [s] & Online [s] & Total [s] \\",
        r"\midrule",
    ]
    for strategy in STRATEGIES:
        for item in projections:
            if item["strategy"] != strategy:
                continue
            equivalent = (
                float(item["force_relative_l2"]) <= 1.0e-3
                and int(item["active_set_difference"]) == 0
            )
            lines.append(
                f"{_method_label(strategy, item)} & "
                f"{'yes' if equivalent else 'no'} & "
                f"{item['contact_steps']} & "
                f"{float(item['setup_wall_seconds']):.3f} & "
                f"{float(item['online_step_wall_seconds']):.4f} & "
                f"{float(item['projected_online_wall_seconds']):.3f} & "
                f"{float(item['projected_numerical_wall_seconds']):.3f} \\\\"
            )
        if strategy != STRATEGIES[-1]:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Projected normal-contact numerical wall time. Setup is measured once; online time is the measured one-step contact, verification, final displacement recovery, and pressure postprocessing multiplied by the requested step count. Equivalent requires relative force error at most $10^{-3}$ and the same active set as factorized $K$.}",
            r"\label{tab:tyre-contact-step-projection}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, projections: list[dict[str, object]]) -> None:
    colors = dict(zip(STRATEGIES, ("C0", "C2", "C3")))
    markers = dict(zip(STRATEGIES, ("o", "^", "D")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for strategy in STRATEGIES:
        values = [item for item in projections if item["strategy"] == strategy]
        equivalent = (
            float(values[0]["force_relative_l2"]) <= 1.0e-3
            and int(values[0]["active_set_difference"]) == 0
        )
        label = _method_label(strategy, values[0]) + (
            " (not equivalent)" if not equivalent else ""
        )
        steps = np.array([item["contact_steps"] for item in values])
        total = np.array(
            [item["projected_numerical_wall_seconds"] for item in values]
        )
        axes[0].plot(
            steps,
            total,
            marker=markers[strategy],
            color=colors[strategy],
            label=label,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(STEP_COUNTS, [str(value) for value in STEP_COUNTS])
    axes[0].set_xlabel("contact steps")
    axes[0].set_ylabel("projected numerical wall time [s]")
    axes[0].set_title("setup + repeated online work")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    x = np.arange(len(STRATEGIES))
    width = 0.24
    for offset, steps in enumerate(STEP_COUNTS):
        values = [
            next(
                item
                for item in projections
                if item["strategy"] == strategy
                and item["contact_steps"] == steps
            )["projected_numerical_wall_seconds"]
            for strategy in STRATEGIES
        ]
        axes[1].bar(
            x + (offset - 1) * width,
            values,
            width,
            label=f"{steps} step" + ("s" if steps != 1 else ""),
        )
    axis_labels = []
    for strategy in STRATEGIES:
        value = next(item for item in projections if item["strategy"] == strategy)
        equivalent = (
            float(value["force_relative_l2"]) <= 1.0e-3
            and int(value["active_set_difference"]) == 0
        )
        axis_labels.append(
            _method_label(strategy, value)
            + ("\n(not equivalent)" if not equivalent else "")
        )
    axes[1].set_xticks(x, axis_labels)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("projected numerical wall time [s]")
    axes[1].set_title("one, ten, and 100 steps")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axial-divisions", type=int, default=60)
    parser.add_argument("--circumferential-divisions", type=int, default=80)
    parser.add_argument(
        "--circumferential-layout",
        choices=("uniform", "graded"),
        default="uniform",
    )
    parser.add_argument("--coarsening-factor", type=float, default=6.0)
    parser.add_argument("--warning-distance", type=float, default=0.02)
    parser.add_argument("--h-leaf-size", type=int, default=16)
    parser.add_argument("--h-eta", type=float, default=1.0)
    parser.add_argument("--h-max-rank", type=int, default=60)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "tyre_step_projection",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        help="reuse this mesh instead of the output directory's generated mesh",
    )
    parser.add_argument("--regenerate-mesh", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate reports from existing records.json",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records_path = output_root / "records.json"
    if args.report_only:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    else:
        mesh_path = (
            args.mesh.expanduser().resolve()
            if args.mesh is not None
            else output_root / "mesh" / "tyre.msh"
        )
        if args.regenerate_mesh or not mesh_path.is_file():
            generate_tyre_mesh(
                root / "geo_files" / "geometry_v2.geo",
                mesh_path,
                axial_divisions=args.axial_divisions,
                circumferential_divisions=args.circumferential_divisions,
                circumferential_layout=args.circumferential_layout,
                coarsening_factor=args.coarsening_factor,
                scale=0.001,
            )
        records = []
        total = args.repetitions * len(STRATEGIES)
        count = 0
        for repetition in range(1, args.repetitions + 1):
            order = list(STRATEGIES)
            if repetition % 2 == 0:
                order.reverse()
            for strategy in order:
                count += 1
                print(
                    f"[{count}/{total}] {strategy}, repetition {repetition}",
                    flush=True,
                )
                records.append(
                    _run_case(
                        root=root,
                        output_root=output_root,
                        mesh_path=mesh_path,
                        axial=args.axial_divisions,
                        circumferential=args.circumferential_divisions,
                        circumferential_layout=args.circumferential_layout,
                        coarsening_factor=args.coarsening_factor,
                        warning_distance=args.warning_distance,
                        h_leaf_size=args.h_leaf_size,
                        h_eta=args.h_eta,
                        h_max_rank=args.h_max_rank,
                        strategy=strategy,
                        repetition=repetition,
                    )
                )
                records_path.write_text(
                    json.dumps(records, indent=2) + "\n", encoding="utf-8"
                )

    summaries = _median_records(records)
    _audit_accuracy(summaries)
    projections = _project(summaries)
    _write_csv(output_root / "benchmark_runs.csv", records)
    _write_csv(output_root / "benchmark_summary.csv", summaries)
    _write_csv(output_root / "step_projection.csv", projections)
    _write_latex(output_root / "step_projection.tex", projections)
    _write_plot(output_root / "step_projection", projections)
    (output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "projection.json").write_text(
        json.dumps(projections, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote step projection under {output_root}")


if __name__ == "__main__":
    main()

"""Rebuild a local-symmetry H-matrix from candidate-only sample closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import matplotlib.pyplot as plt
import numpy as np

from cofebem.fenics.dihedral_compliance import (
    LocalDihedralComplianceEntrySource,
    RestrictedLocalDihedralComplianceEntrySource,
    restrict_local_dihedral_compliance_samples,
)
from cofebem.hmatrices import HMatrix
from cofebem.lcp import (
    LCP,
    RestrictedProjectedPreconditioner,
    SurfaceAreaDiagonalPreconditioner,
    solve,
)


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _sample_path(run_dir: Path) -> Path:
    archive_path = run_dir / "compliance.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        if "samples_file" in archive.files:
            path = Path(str(np.asarray(archive["samples_file"]).reshape(-1)[0]))
            return path if path.is_absolute() else archive_path.parent / path
    fallback = run_dir / "compliance_samples.npy"
    if not fallback.is_file():
        raise FileNotFoundError("could not locate full local compliance samples")
    return fallback


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            root
            / "results"
            / "tyre_step_projection_graded_300x118"
            / "hmatrix"
            / "repeat_1"
        ),
    )
    parser.add_argument("--fine-divisions", type=int, default=118)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-rank", type=int, default=50)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    result_path = run_dir / "contact_result_hmatrix.npz"
    with np.load(result_path, allow_pickle=False) as result:
        candidates = np.asarray(result["candidate_indices"], dtype=np.int64)
        points = np.asarray(result["contact_points"], dtype=float)
        gap = np.asarray(result["gap"], dtype=float)
        areas = np.asarray(result["contact_associated_area"], dtype=float)
        previous_force = np.asarray(result["force"], dtype=float)

    full_samples = np.load(
        _sample_path(run_dir), mmap_mode="r", allow_pickle=False
    )
    sector_step = np.deg2rad(60.0 / args.fine_divisions)
    start = perf_counter()
    start_cpu = process_time()
    source = restrict_local_dihedral_compliance_samples(
        full_samples,
        candidates,
        sector_step=sector_step,
    )
    restriction_seconds = perf_counter() - start
    restriction_cpu_seconds = process_time() - start_cpu
    restricted_path = run_dir / "potential_compliance_samples.npy"
    np.save(restricted_path, source.samples, allow_pickle=False)
    mapped = np.load(restricted_path, mmap_mode="r", allow_pickle=False)
    source = RestrictedLocalDihedralComplianceEntrySource(
        mapped,
        candidates,
        full_n_axial=source.full_n_axial,
        axial_indices=source.axial_indices,
        sector_deltas=source.sector_deltas,
        sector_step=source.sector_step,
    )

    # Selected blocks must be identical to the former full-patch source.
    rng = np.random.default_rng(1729)
    check_count = min(256, candidates.size)
    rows = np.sort(rng.choice(candidates.size, check_count, replace=False))
    columns = np.sort(rng.choice(candidates.size, check_count, replace=False))
    full_source = LocalDihedralComplianceEntrySource(
        full_samples, sector_step=sector_step
    )
    expected = full_source.get_block(candidates[rows], candidates[columns])
    actual = source.get_block(rows, columns)
    entry_error = float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), np.finfo(float).tiny)
    )
    del expected, actual, full_source

    source.reset_stats()
    start = perf_counter()
    start_cpu = process_time()
    matrix = HMatrix.from_entry_source(
        points[candidates],
        source,
        leaf_size=args.leaf_size,
        eta=args.eta,
        tol=args.tolerance,
        split="pca",
        lr_approx="aca_partial",
        symmetric=True,
        max_rank=args.max_rank,
    )
    build_seconds = perf_counter() - start
    build_cpu_seconds = process_time() - start_cpu
    h_stats = matrix.stats()
    query_stats = source.stats()

    full_preconditioner = SurfaceAreaDiagonalPreconditioner(areas)
    preconditioner = RestrictedProjectedPreconditioner(
        full_preconditioner, candidates
    )
    start = perf_counter()
    start_cpu = process_time()
    result = solve(
        LCP(matrix, gap[candidates]),
        method="ppcg",
        tol=1.0e-10,
        max_iter=5000,
        record_history=True,
        beta_method="pr_plus",
        preconditioner=preconditioner,
    )
    solve_seconds = perf_counter() - start
    solve_cpu_seconds = process_time() - start_cpu
    if not result.converged:
        raise RuntimeError(result.message)
    rebuilt_force = np.zeros_like(previous_force)
    rebuilt_force[candidates] = result.z
    force_error = float(
        np.linalg.norm(rebuilt_force - previous_force)
        / max(np.linalg.norm(previous_force), np.finfo(float).tiny)
    )

    full_entries = int(full_samples.size)
    restricted_entries = int(mapped.size)
    full_solves = 2 * int(full_samples.shape[1])
    restricted_solves = 2 * int(source.axial_indices.size)
    report = {
        "candidate_unknowns": int(candidates.size),
        "full_sample_shape": list(full_samples.shape),
        "restricted_sample_shape": list(mapped.shape),
        "full_sample_entries": full_entries,
        "restricted_sample_entries": restricted_entries,
        "sample_storage_ratio": restricted_entries / full_entries,
        "full_sampling_solves": full_solves,
        "restricted_sampling_solves": restricted_solves,
        "restriction_seconds": restriction_seconds,
        "restriction_cpu_seconds": restriction_cpu_seconds,
        "entry_relative_error": entry_error,
        "hmatrix_build_seconds": build_seconds,
        "hmatrix_build_cpu_seconds": build_cpu_seconds,
        "hmatrix_stored_entries": int(h_stats["memory_entries"]),
        "hmatrix_low_rank_blocks": int(h_stats["low_rank"]),
        "hmatrix_dense_blocks": int(h_stats["dense"]),
        "entry_source_query_calls": int(query_stats["query_calls"]),
        "entry_source_queried_entries": int(query_stats["queried_entries"]),
        "contact_solve_seconds": solve_seconds,
        "contact_solve_cpu_seconds": solve_cpu_seconds,
        "contact_iterations": int(result.iterations),
        "force_relative_error_vs_previous": force_error,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    report_path = run_dir / "potential_hmatrix_rebuild.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(
        run_dir / "potential_hmatrix_contact_result.npz",
        force=rebuilt_force,
        candidate_indices=candidates,
        residual=result.residual,
        iterations=result.iterations,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].bar(
        ["full patch", "potential closure"],
        [full_entries * 8 / 2**20, restricted_entries * 8 / 2**20],
        color=("C0", "C1"),
    )
    axes[0].set_ylabel("sample storage [MiB]")
    axes[0].set_title("local-symmetry samples")
    axes[1].bar(
        ["full patch", "potential closure"],
        [full_solves, restricted_solves],
        color=("C0", "C1"),
    )
    axes[1].set_ylabel("required FE back-solves")
    axes[1].set_title("reference source nodes")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "potential_hmatrix_rebuild.png", dpi=180)
    fig.savefig(run_dir / "potential_hmatrix_rebuild.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Compare saved tyre compliance-strategy runs on one physical case."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np


STRATEGIES = (
    "hmatrix",
    "fe_matrix_free",
    "fe_iterative",
    "mumps_schur",
)


def _scalar(archive, name, cast=float, default=None):
    if name not in archive.files:
        if default is not None:
            return default
        raise ValueError(f"result archive is missing field {name!r}")
    return cast(np.asarray(archive[name]).reshape(-1)[0])


def _load(path: Path, expected_strategy: str):
    archive = np.load(path, allow_pickle=False)
    strategy = str(np.asarray(archive["compliance_strategy"]).item())
    if strategy != expected_strategy:
        archive.close()
        raise ValueError(
            f"{path} contains strategy {strategy!r}, expected {expected_strategy!r}"
        )
    return archive


def _validate_same_case(reference, candidate, strategy: str) -> None:
    for field in ("gap", "candidate_indices"):
        left = np.asarray(reference[field])
        right = np.asarray(candidate[field])
        if left.shape != right.shape or not np.allclose(
            left, right, rtol=0.0, atol=1.0e-13
        ):
            raise ValueError(f"{strategy} does not use the same {field}")
    for field in (
        "axial_divisions",
        "circumferential_divisions",
        "young_modulus",
        "poisson_ratio",
        "inflation_pressure",
        "warning_distance",
    ):
        if field in reference.files and field in candidate.files:
            if not np.isclose(
                _scalar(reference, field),
                _scalar(candidate, field),
                rtol=1.0e-13,
                atol=1.0e-15,
            ):
                raise ValueError(f"{strategy} uses a different {field}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "results" / "tyre_dihedral"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="strategies to include (default: every available result)",
    )
    for strategy in STRATEGIES:
        parser.add_argument(
            "--" + strategy.replace("_", "-"),
            type=Path,
            default=default_dir / f"contact_result_{strategy}.npz",
        )
    args = parser.parse_args()
    paths = {
        strategy: getattr(args, strategy)
        for strategy in STRATEGIES
        if strategy in args.include and getattr(args, strategy).is_file()
    }
    if len(paths) < 2:
        parser.error("at least two strategy result archives must exist")

    with ExitStack() as stack:
        archives = {}
        for strategy, path in paths.items():
            archive = _load(path, strategy)
            stack.callback(archive.close)
            archives[strategy] = archive
        reference_name = (
            "fe_matrix_free" if "fe_matrix_free" in archives else next(iter(archives))
        )
        reference = archives[reference_name]
        for strategy, archive in archives.items():
            _validate_same_case(reference, archive, strategy)

        labels = {
            "hmatrix": "H-matrix",
            "fe_matrix_free": "FE direct",
            "fe_iterative": "FE iterative",
            "mumps_schur": "MUMPS Schur",
        }
        names = list(archives)
        timing_fields = (
            ("factorization", "factorization_seconds"),
            ("compliance load", "compliance_load_seconds"),
            ("compliance sampling", "compliance_sampling_seconds"),
            ("operator build", "operator_build_seconds"),
            ("contact solve", "contact_solve_seconds"),
            ("verification", "verification_seconds"),
            ("strategy total", "strategy_total_seconds"),
        )
        print("\nTiming comparison (seconds)")
        print(f"{'stage':<22}" + "".join(f"{labels[n]:>16}" for n in names))
        for label, field in timing_fields:
            print(
                f"{label:<22}"
                + "".join(
                    f"{_scalar(archives[n], field, default=0.0):>16.3f}"
                    for n in names
                )
            )

        reference_force = np.asarray(reference["force"], dtype=float)
        reference_clearance = np.asarray(reference["clearance"], dtype=float)
        force_norm = max(np.linalg.norm(reference_force), np.finfo(float).tiny)
        print(f"\nSolution agreement relative to {labels[reference_name]}")
        print(
            f"{'strategy':<18} {'force rel L2':>14} {'clearance Linf':>18} "
            f"{'active diff':>12} {'PPCG iters':>12}"
        )
        reference_active = reference_force > 0.0
        for strategy in names:
            archive = archives[strategy]
            force = np.asarray(archive["force"], dtype=float)
            clearance = np.asarray(archive["clearance"], dtype=float)
            print(
                f"{labels[strategy]:<18} "
                f"{np.linalg.norm(force-reference_force)/force_norm:>14.6e} "
                f"{np.linalg.norm(clearance-reference_clearance, ord=np.inf):>18.6e} "
                f"{np.count_nonzero((force > 0.0) != reference_active):>12d} "
                f"{_scalar(archive, 'iterations', int):>12d}"
            )

        print("\nOperator work/storage")
        for strategy in names:
            archive = archives[strategy]
            print(
                f"{labels[strategy]:<18} entries="
                f"{_scalar(archive, 'compliance_stored_entries', int, 0):d}, "
                f"FE/Schur solves={_scalar(archive, 'fe_linear_solves', int, 0):d}, "
                f"inner iterations="
                f"{_scalar(archive, 'fe_linear_iterations', int, 0):d}, "
                f"Schur estimate="
                f"{_scalar(archive, 'schur_estimated_memory_bytes', int, 0)/2**30:.3f} GiB, "
                f"peak RSS={_scalar(archive, 'peak_rss_bytes', int, 0)/2**30:.3f} GiB"
            )


if __name__ == "__main__":
    main()

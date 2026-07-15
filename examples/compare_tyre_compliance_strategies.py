"""Compare saved tyre H-matrix and flexibility-matrix-free contact runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _scalar(archive, name, cast=float):
    if name not in archive.files:
        raise ValueError(f"result archive is missing field {name!r}")
    return cast(np.asarray(archive[name]).reshape(-1)[0])


def _load(path: Path, expected_strategy: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    archive = np.load(path, allow_pickle=False)
    strategy = str(np.asarray(archive["compliance_strategy"]).item())
    if strategy != expected_strategy:
        archive.close()
        raise ValueError(
            f"{path} contains strategy {strategy!r}, expected {expected_strategy!r}"
        )
    return archive


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_dir = root / "results" / "tyre_dihedral"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hmatrix",
        type=Path,
        default=default_dir / "contact_result_hmatrix.npz",
    )
    parser.add_argument(
        "--matrix-free",
        type=Path,
        default=default_dir / "contact_result_fe_matrix_free.npz",
    )
    args = parser.parse_args()

    hmatrix = _load(args.hmatrix, "hmatrix")
    matrix_free = _load(args.matrix_free, "fe_matrix_free")
    try:
        force_h = np.asarray(hmatrix["force"], dtype=float)
        force_f = np.asarray(matrix_free["force"], dtype=float)
        gap_h = np.asarray(hmatrix["gap"], dtype=float)
        gap_f = np.asarray(matrix_free["gap"], dtype=float)
        clearance_h = np.asarray(hmatrix["clearance"], dtype=float)
        clearance_f = np.asarray(matrix_free["clearance"], dtype=float)
        for name, left, right in (
            ("force", force_h, force_f),
            ("gap", gap_h, gap_f),
            ("clearance", clearance_h, clearance_f),
        ):
            if left.shape != right.shape:
                raise ValueError(
                    f"{name} shapes differ: {left.shape} versus {right.shape}"
                )
        if not np.allclose(gap_h, gap_f, rtol=0.0, atol=1.0e-13):
            raise ValueError("runs do not use the same free gap")
        candidates_h = np.asarray(hmatrix["candidate_indices"], dtype=np.int64)
        candidates_f = np.asarray(
            matrix_free["candidate_indices"], dtype=np.int64
        )
        if not np.array_equal(candidates_h, candidates_f):
            raise ValueError("runs do not use the same potential contact zone")

        configuration_fields = (
            "axial_divisions",
            "circumferential_divisions",
            "scale",
            "indentation",
            "floor_level",
            "floor_grid_size",
            "floor_margin",
            "roughness_rms",
            "roughness_hurst",
            "roughness_k_low",
            "roughness_k_high",
            "roughness_seed",
            "roughness_plateau",
            "roughness_noise",
            "young_modulus",
            "poisson_ratio",
            "inflation_pressure",
            "warning_distance",
            "warning_verification_tol",
        )
        for field in configuration_fields:
            if field not in hmatrix.files or field not in matrix_free.files:
                continue
            left = _scalar(hmatrix, field)
            right = _scalar(matrix_free, field)
            if not np.isclose(left, right, rtol=1.0e-13, atol=1.0e-15):
                raise ValueError(
                    f"runs use different {field}: {left} versus {right}"
                )
        for field in (
            "contact_solver",
            "preconditioner",
            "boundary_condition_id",
            "floor_kind",
        ):
            if field not in hmatrix.files or field not in matrix_free.files:
                raise ValueError(
                    f"result archives do not both identify {field!r}"
                )
            left = str(np.asarray(hmatrix[field]).item())
            right = str(np.asarray(matrix_free[field]).item())
            if left != right:
                raise ValueError(
                    f"runs use different {field}: {left!r} versus {right!r}"
                )

        timing_fields = (
            ("floor build/output", "floor_build_seconds"),
            ("LU factorization", "factorization_seconds"),
            ("compliance loading", "compliance_load_seconds"),
            ("compliance sampling", "compliance_sampling_seconds"),
            ("operator build", "operator_build_seconds"),
            ("contact solve", "contact_solve_seconds"),
            ("verification", "verification_seconds"),
            ("strategy total", "strategy_total_seconds"),
        )
        print("\nTiming comparison (seconds)")
        print(f"{'stage':<24} {'H-matrix':>12} {'FE matrix-free':>16}")
        for label, field in timing_fields:
            print(
                f"{label:<24} {_scalar(hmatrix, field):>12.3f} "
                f"{_scalar(matrix_free, field):>16.3f}"
            )

        norm_force = max(np.linalg.norm(force_f), np.finfo(float).tiny)
        norm_force_inf = max(
            np.linalg.norm(force_f, ord=np.inf), np.finfo(float).tiny
        )
        active_h = force_h > 0.0
        active_f = force_f > 0.0
        objective_h = 0.5 * force_h @ (clearance_h - gap_h) + gap_h @ force_h
        objective_f = 0.5 * force_f @ (clearance_f - gap_f) + gap_f @ force_f

        print("\nSolution agreement")
        print(
            "force relative L2        "
            f"{np.linalg.norm(force_h - force_f) / norm_force:.6e}"
        )
        print(
            "force relative Linf      "
            f"{np.linalg.norm(force_h - force_f, ord=np.inf) / norm_force_inf:.6e}"
        )
        print(
            "clearance absolute Linf  "
            f"{np.linalg.norm(clearance_h - clearance_f, ord=np.inf):.6e}"
        )
        print(
            f"active nodes             {active_h.sum()} / {active_f.sum()} "
            f"(different: {np.count_nonzero(active_h != active_f)})"
        )
        print(f"total force              {force_h.sum():.9e} / {force_f.sum():.9e}")
        print(f"quadratic objective      {objective_h:.12e} / {objective_f:.12e}")

        print("\nRepresentation work/storage")
        print(
            "potential/global nodes   "
            f"{_scalar(hmatrix, 'potential_contact_unknowns', int)} / "
            f"{_scalar(hmatrix, 'global_contact_unknowns', int)}"
        )
        print(
            "PPCG iterations          "
            f"{_scalar(hmatrix, 'iterations', int)} / "
            f"{_scalar(matrix_free, 'iterations', int)}"
        )
        print(
            "compliance entries       "
            f"{_scalar(hmatrix, 'compliance_stored_entries', int)} / "
            f"{_scalar(matrix_free, 'compliance_stored_entries', int)}"
        )
        print(
            "FE back-solves in PPCG   "
            f"{_scalar(hmatrix, 'fe_linear_solves', int)} / "
            f"{_scalar(matrix_free, 'fe_linear_solves', int)}"
        )
    finally:
        hmatrix.close()
        matrix_free.close()


if __name__ == "__main__":
    main()

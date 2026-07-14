"""Compliance sampling for a tyre with discrete rotational symmetry.

The tyre axis is the global x axis.  A global road-normal (z) load does not
remain a z load when rotated to a reference meridian, so reconstruction needs
the 2x2 transverse (y/z) compliance tensor rather than a scalar cyclic shift.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from cofebem.hmatrices.entry_source import MatrixEntrySource


@dataclass(frozen=True)
class SectorOrdering:
    """Contact unknowns ordered by circumferential sector, then axial x."""

    scalar_dofs: np.ndarray
    parent_y_dofs: np.ndarray
    parent_z_dofs: np.ndarray
    points: np.ndarray
    sector_angle_error: float

    @property
    def n_sectors(self) -> int:
        return int(self.scalar_dofs.shape[0])

    @property
    def n_axial(self) -> int:
        return int(self.scalar_dofs.shape[1])


class FactorizedComplianceOperator:
    """Apply contact compliance through a reusable factorized FE solve.

    The operator maps nodal forces at ``force_dofs`` to displacements at
    ``response_dofs`` by solving ``A u = f`` with an already configured KSP.
    It is intended for serial direct-factorization workflows. No compliance
    matrix or H-matrix is constructed or stored.
    """

    def __init__(
        self,
        A: PETSc.Mat,
        ksp: PETSc.KSP,
        force_dofs: np.ndarray,
        response_dofs: np.ndarray | None = None,
        *,
        symmetric: bool = True,
    ) -> None:
        if A.comm.size != 1:
            raise NotImplementedError(
                "FactorizedComplianceOperator is currently serial"
            )
        force = np.asarray(force_dofs, dtype=np.int32).reshape(-1)
        response = (
            force.copy()
            if response_dofs is None
            else np.asarray(response_dofs, dtype=np.int32).reshape(-1)
        )
        if force.size == 0 or response.size == 0:
            raise ValueError("force_dofs and response_dofs must not be empty")
        matrix_rows, matrix_columns = A.getSize()
        if matrix_rows != matrix_columns:
            raise ValueError("A must be square")
        if np.any(force < 0) or np.any(force >= matrix_columns):
            raise IndexError("force DOF outside the FE matrix")
        if np.any(response < 0) or np.any(response >= matrix_rows):
            raise IndexError("response DOF outside the FE matrix")
        if np.unique(force).size != force.size:
            raise ValueError("force_dofs must be unique")
        if np.unique(response).size != response.size:
            raise ValueError("response_dofs must be unique")

        self.A = A
        self.ksp = ksp
        self.force_dofs = force.copy()
        self.response_dofs = response.copy()
        self.shape = (response.size, force.size)
        self.symmetric = bool(
            symmetric
            and self.shape[0] == self.shape[1]
            and np.array_equal(self.force_dofs, self.response_dofs)
        )
        self._rhs = A.createVecRight()
        self._displacement = A.createVecLeft()
        self._last_forces: np.ndarray | None = None
        self.operator_applications = 0
        self.linear_solves = 0
        self.cache_hits = 0
        self.zero_bypasses = 0
        self.solve_seconds = 0.0

    def _solve(self, forces: np.ndarray) -> None:
        if self._last_forces is not None and np.array_equal(
            forces, self._last_forces
        ):
            self.cache_hits += 1
            return
        if not np.any(forces):
            self._displacement.set(0.0)
            self.zero_bypasses += 1
        else:
            self._rhs.set(0.0)
            self._rhs.setValues(
                self.force_dofs,
                forces,
                addv=PETSc.InsertMode.INSERT_VALUES,
            )
            self._rhs.assemble()
            start = perf_counter()
            self.ksp.solve(self._rhs, self._displacement)
            self.solve_seconds += perf_counter() - start
            self.linear_solves += 1
            if int(self.ksp.getConvergedReason()) <= 0:
                raise RuntimeError(
                    "Factorized FE compliance solve failed with PETSc reason "
                    f"{self.ksp.getConvergedReason()}"
                )
        self._last_forces = forces.copy()

    def apply(
        self,
        forces: np.ndarray,
        *,
        response_dofs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the flexibility and extract selected displacement DOFs."""
        values = np.asarray(forces, dtype=np.float64).reshape(-1)
        if values.shape != (self.shape[1],):
            raise ValueError(
                f"forces must have shape ({self.shape[1]},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("forces contain NaN or infinite values")
        responses = (
            self.response_dofs
            if response_dofs is None
            else np.asarray(response_dofs, dtype=np.int32).reshape(-1)
        )
        if np.any(responses < 0) or np.any(responses >= self.A.getSize()[0]):
            raise IndexError("response DOF outside the FE matrix")
        self.operator_applications += 1
        self._solve(values)
        displacement = self._displacement.getArray(readonly=True)
        return np.array(displacement[responses], dtype=np.float64, copy=True)

    def __matmul__(self, forces: np.ndarray) -> np.ndarray:
        """Return contact displacement for one contact-force vector."""
        return self.apply(forces)

    def stats(self) -> dict[str, int | float]:
        """Return accumulated operator and factorized-solve counters."""
        return {
            "operator_applications": self.operator_applications,
            "linear_solves": self.linear_solves,
            "cache_hits": self.cache_hits,
            "zero_bypasses": self.zero_bypasses,
            "solve_seconds": self.solve_seconds,
        }


class DihedralComplianceEntrySource(MatrixEntrySource):
    """Global-z compliance entries derived from one sampled axial meridian.

    The stored data are the y/z responses at every surface node to y/z loads
    at the axial nodes of sector zero.  Rotation maps any requested global
    source/target pair back to that reference data.  Thus ``get_block`` can
    answer ACA cross queries without ever reconstructing the full ``S_c``.

    Matrix indices follow the sector-major, axial-minor ordering produced by
    :func:`order_contact_sectors`: rows are target nodes and columns are
    source nodes.
    """

    def __init__(self, samples: np.ndarray):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 5 or samples.shape[0] != 2 or samples.shape[2] != 2:
            raise ValueError(
                "samples must have shape (2, n_axial, 2, n_sectors, n_axial)"
            )
        _, n_axial, _, n_sectors, target_axial = samples.shape
        if target_axial != n_axial:
            raise ValueError("source and target axial sizes must match")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples contain NaN or infinite values")

        self.samples = samples
        self.n_axial = int(n_axial)
        self.n_sectors = int(n_sectors)
        n = self.n_axial * self.n_sectors
        self.shape = (n, n)
        self.query_calls = 0
        self.queried_entries = 0
        self.largest_query = (0, 0)

    def _evaluate(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.int64)
        source = np.asarray(source, dtype=np.int64)
        target_sector = target // self.n_axial
        target_axial = target % self.n_axial
        source_sector = source // self.n_axial
        source_axial = source % self.n_axial
        delta = (target_sector - source_sector) % self.n_sectors

        angle = source_sector * (2.0 * np.pi / self.n_sectors)
        q = (np.sin(angle), np.cos(angle))
        values = np.zeros(np.broadcast_shapes(target.shape, source.shape), dtype=float)
        for force_component in range(2):
            for response_component in range(2):
                values += (
                    q[force_component]
                    * self.samples[
                        force_component,
                        source_axial,
                        response_component,
                        delta,
                        target_axial,
                    ]
                    * q[response_component]
                )
        return values

    def get_block(
        self, row_indices: np.ndarray, column_indices: np.ndarray
    ) -> np.ndarray:
        """Return one Cartesian subblock reconstructed directly from samples."""
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        columns = np.asarray(column_indices, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("row index outside compliance matrix")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise IndexError("column index outside compliance matrix")
        self.query_calls += 1
        self.queried_entries += int(rows.size * columns.size)
        if rows.size * columns.size > self.largest_query[0] * self.largest_query[1]:
            self.largest_query = (int(rows.size), int(columns.size))
        return self._evaluate(rows[:, None], columns[None, :])

    def get_entries(
        self, row_indices: np.ndarray, column_indices: np.ndarray
    ) -> np.ndarray:
        """Return paired entries, useful for reciprocity diagnostics."""
        rows, columns = np.broadcast_arrays(
            np.asarray(row_indices, dtype=np.int64),
            np.asarray(column_indices, dtype=np.int64),
        )
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("row index outside compliance matrix")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise IndexError("column index outside compliance matrix")
        self.query_calls += 1
        self.queried_entries += int(rows.size)
        if rows.size > self.largest_query[0] * self.largest_query[1]:
            self.largest_query = (int(rows.size), 1)
        return self._evaluate(rows, columns)

    def reciprocity_error(
        self, *, sample_size: int = 4096, seed: int = 0
    ) -> float:
        """Estimate ``||S-S.T||/||S||`` from deterministic random pairs."""
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        rng = np.random.default_rng(seed)
        rows = rng.integers(0, self.shape[0], size=sample_size)
        columns = rng.integers(0, self.shape[1], size=sample_size)
        forward = self.get_entries(rows, columns)
        reverse = self.get_entries(columns, rows)
        return float(
            np.linalg.norm(forward - reverse)
            / max(np.linalg.norm(forward), np.finfo(float).tiny)
        )

    def stats(self) -> dict[str, int | tuple[int, int]]:
        """Return entry-query counters accumulated during construction."""
        return {
            "query_calls": self.query_calls,
            "queried_entries": self.queried_entries,
            "largest_query": self.largest_query,
        }

    def reset_stats(self) -> None:
        """Reset counters, for example after pre-construction diagnostics."""
        self.query_calls = 0
        self.queried_entries = 0
        self.largest_query = (0, 0)


def load_dihedral_compliance_archive(
    path: str | Path,
    points: np.ndarray,
    *,
    n_axial: int,
    n_sectors: int,
    young_modulus: float | None = None,
    poisson_ratio: float | None = None,
) -> np.ndarray:
    """Load and validate reference-meridian samples from ``compliance.npz``.

    A uniform translation of the saved contact points is accepted so that the
    same linear compliance can be reused at another indentation.  Inflation
    pressure is intentionally not checked because it changes the free
    displacement and effective gap, not the linear compliance operator.

    Archives written before elastic constants were added remain readable.  A
    warning makes clear that those legacy files cannot verify the material.
    """
    archive_path = Path(path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Compliance archive does not exist: {archive_path}")

    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            missing = {"samples", "points"}.difference(archive.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Compliance archive {archive_path} is missing: {names}"
                )
            samples = np.array(archive["samples"], dtype=float, copy=True)
            saved_points = np.array(archive["points"], dtype=float, copy=True)

            metadata: dict[str, object] = {}
            for name in (
                "axial_divisions",
                "circumferential_divisions",
                "young_modulus",
                "poisson_ratio",
            ):
                if name in archive.files:
                    value = np.asarray(archive[name])
                    if value.size != 1:
                        raise ValueError(
                            f"Compliance archive field {name!r} must be scalar"
                        )
                    metadata[name] = value.reshape(-1)[0].item()
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Compliance archive"):
            raise
        raise ValueError(
            f"Could not read compliance archive {archive_path}: {exc}"
        ) from exc

    source = DihedralComplianceEntrySource(samples)
    if source.n_axial != n_axial or source.n_sectors != n_sectors:
        raise ValueError(
            "Loaded compliance dimensions do not match the current contact surface: "
            f"archive={source.n_sectors} sectors x {source.n_axial} axial nodes, "
            f"current={n_sectors} x {n_axial}"
        )

    current_points = np.asarray(points, dtype=float)
    expected_shape = (source.shape[0], 3)
    if saved_points.shape != expected_shape or current_points.shape != expected_shape:
        raise ValueError(
            "Loaded compliance point dimensions do not match the current contact "
            f"surface: archive={saved_points.shape}, current={current_points.shape}, "
            f"expected={expected_shape}"
        )
    if not np.all(np.isfinite(saved_points)) or not np.all(np.isfinite(current_points)):
        raise ValueError("Compliance archive or current contact points are not finite")

    saved_centered = saved_points - saved_points.mean(axis=0)
    current_centered = current_points - current_points.mean(axis=0)
    diameter = max(float(np.ptp(current_points, axis=0).max()), 1.0e-12)
    if not np.allclose(
        saved_centered,
        current_centered,
        rtol=1.0e-9,
        atol=max(1.0e-12, 1.0e-9 * diameter),
    ):
        raise ValueError(
            "Loaded compliance contact geometry or sector-major ordering does not "
            "match the current mesh"
        )

    expected_divisions = {
        "axial_divisions": n_axial - 1,
        "circumferential_divisions": n_sectors,
    }
    for name, expected in expected_divisions.items():
        if name in metadata and int(metadata[name]) != expected:
            raise ValueError(
                f"Loaded compliance {name}={metadata[name]} does not match "
                f"the current value {expected}"
            )

    material = {
        "young_modulus": young_modulus,
        "poisson_ratio": poisson_ratio,
    }
    missing_material = []
    for name, expected in material.items():
        if expected is None:
            continue
        if name not in metadata:
            missing_material.append(name)
        elif not np.isclose(
            float(metadata[name]), float(expected), rtol=1.0e-12, atol=0.0
        ):
            raise ValueError(
                f"Loaded compliance {name}={metadata[name]} does not match "
                f"the current value {expected}"
            )
    if missing_material:
        warnings.warn(
            "Loaded legacy compliance archive has no "
            + "/".join(missing_material)
            + "; material compatibility could not be verified",
            UserWarning,
            stacklevel=2,
        )

    return samples


def infer_regular_sector_shape(
    points: np.ndarray,
    *,
    axis_yz: tuple[float, float] = (0.0, 0.0),
    angle_tol: float = 1.0e-8,
) -> tuple[int, int]:
    """Infer ``(n_sectors, nodes_per_sector)`` from a revolved point set.

    All points must lie on equally spaced meridians about the global x axis.
    The points need not be ordered, but every meridian must contain the same
    number of points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have nonzero shape (n, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain NaN or infinite values")
    if angle_tol <= 0.0:
        raise ValueError("angle_tol must be positive")

    y0, z0 = axis_yz
    relative_y = points[:, 1] - y0
    relative_z = points[:, 2] - z0
    radius = np.hypot(relative_y, relative_z)
    if np.any(radius <= np.finfo(float).eps):
        raise ValueError("cannot infer a meridian for points on the rotation axis")

    angles = np.sort(np.mod(np.arctan2(relative_z, relative_y), 2.0 * np.pi))
    clusters: list[list[float]] = [[float(angles[0])]]
    for angle in angles[1:]:
        if float(angle) - clusters[-1][-1] <= angle_tol:
            clusters[-1].append(float(angle))
        else:
            clusters.append([float(angle)])

    if len(clusters) > 1 and clusters[0][0] + 2.0 * np.pi - clusters[-1][-1] <= angle_tol:
        clusters[0] = [value - 2.0 * np.pi for value in clusters[-1]] + clusters[0]
        clusters.pop()

    counts = np.array([len(cluster) for cluster in clusters], dtype=int)
    if np.any(counts != counts[0]):
        raise ValueError(
            "cannot infer regular sectors because meridian point counts differ: "
            f"{sorted(np.unique(counts).tolist())}"
        )

    centers = np.mod(
        np.array([np.mean(cluster) for cluster in clusters]), 2.0 * np.pi
    )
    centers.sort()
    n_sectors = len(centers)
    if n_sectors < 2:
        raise ValueError("at least two meridians are required")
    gaps = np.diff(np.concatenate([centers, centers[:1] + 2.0 * np.pi]))
    expected_gap = 2.0 * np.pi / n_sectors
    if not np.allclose(gaps, expected_gap, rtol=0.0, atol=2.0 * angle_tol):
        raise ValueError(
            "point meridians are not equally spaced: maximum sector-gap error is "
            f"{np.max(np.abs(gaps - expected_gap)):.3e} rad"
        )
    return n_sectors, int(counts[0])


def potential_contact_indices(
    free_gap: np.ndarray, warning_distance: float
) -> np.ndarray:
    """Select nodes whose free gap lies within the warning distance."""
    gap = np.asarray(free_gap, dtype=float).reshape(-1)
    if gap.size == 0 or not np.all(np.isfinite(gap)):
        raise ValueError("free_gap must be a nonempty finite vector")
    if np.isnan(warning_distance) or warning_distance < 0.0:
        raise ValueError("warning_distance must be non-negative")
    indices = np.flatnonzero(gap <= warning_distance).astype(np.int64)
    if indices.size == 0:
        raise ValueError(
            "warning_distance selects no potential contact nodes; its value "
            f"is {warning_distance:g} while the minimum free gap is {gap.min():.6g}"
        )
    return indices


def dilate_sector_axial_mask(mask: np.ndarray, halo: int = 1) -> np.ndarray:
    """Dilate a sector/axial mask, periodically in the sector direction."""
    expanded = np.asarray(mask, dtype=bool)
    if expanded.ndim != 2 or expanded.size == 0:
        raise ValueError("mask must be a nonempty (n_sectors, n_axial) array")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    expanded = expanded.copy()
    for _ in range(halo):
        previous = expanded
        expanded = previous | np.roll(previous, 1, axis=0) | np.roll(
            previous, -1, axis=0
        )
        expanded[:, 1:] |= previous[:, :-1]
        expanded[:, :-1] |= previous[:, 1:]
    return expanded


def restricted_source_clearance(
    source: MatrixEntrySource,
    free_gap: np.ndarray,
    source_indices: np.ndarray,
    forces: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Evaluate full-surface clearance from restricted nonzero forces.

    Rows are queried in bounded chunks, so this verification never stores a
    full or rectangular global compliance matrix.
    """
    gap = np.asarray(free_gap, dtype=float).reshape(-1)
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    pressure = np.asarray(forces, dtype=float).reshape(-1)
    if source.shape != (gap.size, gap.size):
        raise ValueError("source shape must match free_gap size")
    if indices.shape != pressure.shape or indices.size == 0:
        raise ValueError("source_indices and forces must have equal nonzero size")
    if np.any(indices < 0) or np.any(indices >= gap.size):
        raise IndexError("source index outside compliance matrix")
    if not np.all(np.isfinite(gap)) or not np.all(np.isfinite(pressure)):
        raise ValueError("free_gap and forces must be finite")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    clearance = gap.copy()
    for start in range(0, gap.size, chunk_size):
        stop = min(start + chunk_size, gap.size)
        rows = np.arange(start, stop, dtype=np.int64)
        clearance[start:stop] += source.get_block(rows, indices) @ pressure
    return clearance


def order_contact_sectors(
    points: np.ndarray,
    scalar_dofs: np.ndarray,
    parent_y_dofs: np.ndarray,
    parent_z_dofs: np.ndarray,
    n_sectors: int,
    *,
    axis_yz: tuple[float, float] = (0.0, 0.0),
    angle_tol: float = 1.0e-8,
    geometry_tol: float = 1.0e-9,
) -> SectorOrdering:
    """Group a regular surface of revolution into equal angular sectors."""
    points = np.asarray(points, dtype=float)
    scalar_dofs = np.asarray(scalar_dofs, dtype=np.int32)
    parent_y_dofs = np.asarray(parent_y_dofs, dtype=np.int32)
    parent_z_dofs = np.asarray(parent_z_dofs, dtype=np.int32)
    n = len(points)
    if points.shape != (n, 3):
        raise ValueError("points must have shape (n, 3)")
    if any(array.shape != (n,) for array in (scalar_dofs, parent_y_dofs, parent_z_dofs)):
        raise ValueError("all DOF arrays must have shape (n,)")
    if n_sectors < 2:
        raise ValueError("n_sectors must be >= 2")

    y0, z0 = axis_yz
    angles = np.mod(np.arctan2(points[:, 2] - z0, points[:, 1] - y0), 2.0 * np.pi)
    dtheta = 2.0 * np.pi / n_sectors
    sectors = np.rint(angles / dtheta).astype(int) % n_sectors
    expected = sectors * dtheta
    angle_error = np.abs(np.arctan2(np.sin(angles - expected), np.cos(angles - expected)))
    max_angle_error = float(angle_error.max(initial=0.0))
    if max_angle_error > angle_tol:
        raise ValueError(
            "Contact nodes do not lie on the requested regular sectors: "
            f"maximum angular error is {max_angle_error:.3e} rad"
        )

    members = [np.flatnonzero(sectors == sector) for sector in range(n_sectors)]
    counts = {len(indices) for indices in members}
    if len(counts) != 1:
        raise ValueError(f"Sector contact-node counts differ: {sorted(counts)}")
    n_axial = counts.pop()
    if n_axial == 0:
        raise ValueError("No contact nodes were found")

    ordered_indices = np.empty((n_sectors, n_axial), dtype=np.int64)
    for sector, indices in enumerate(members):
        ordered_indices[sector] = indices[np.argsort(points[indices, 0])]

    ordered_points = points[ordered_indices]
    reference_x = ordered_points[0, :, 0]
    reference_radius = np.linalg.norm(
        ordered_points[0, :, 1:] - np.array([y0, z0]), axis=1
    )
    for sector in range(1, n_sectors):
        radius = np.linalg.norm(
            ordered_points[sector, :, 1:] - np.array([y0, z0]), axis=1
        )
        if not np.allclose(ordered_points[sector, :, 0], reference_x, atol=geometry_tol, rtol=0.0):
            raise ValueError(f"Axial coordinates differ in sector {sector}")
        if not np.allclose(radius, reference_radius, atol=geometry_tol, rtol=0.0):
            raise ValueError(f"Radial coordinates differ in sector {sector}")

    return SectorOrdering(
        scalar_dofs=scalar_dofs[ordered_indices],
        parent_y_dofs=parent_y_dofs[ordered_indices],
        parent_z_dofs=parent_z_dofs[ordered_indices],
        points=ordered_points,
        sector_angle_error=max_angle_error,
    )


def create_lu_solver(
    A: PETSc.Mat,
    comm: MPI.Comm,
    *,
    factor_solver_type: str | None = None,
) -> PETSc.KSP:
    """Create one reusable PETSc PREONLY+LU compliance-action solver."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    if factor_solver_type:
        pc.setFactorSolverType(factor_solver_type)
    ksp.setFromOptions()
    ksp.setUp()
    return ksp


def sample_reference_transverse_compliance(
    A: PETSc.Mat,
    ksp: PETSc.KSP,
    ordering: SectorOrdering,
    *,
    force_magnitude: float = 1.0,
    show_progress: bool = True,
) -> np.ndarray:
    """Sample y/z loads only on the axial nodes of reference sector zero.

    Returns an array with axes ``(force_component, source_axial,
    response_component, target_sector, target_axial)``. Components 0 and 1
    denote global y and z at the zero-angle reference meridian.
    """
    if A.comm.size != 1:
        raise NotImplementedError(
            "Reference compliance sampling is currently serial; distributed "
            "global DOF gathering has not been implemented."
        )
    if force_magnitude == 0.0:
        raise ValueError("force_magnitude must be nonzero")

    n_sectors, n_axial = ordering.scalar_dofs.shape
    samples = np.empty((2, n_axial, 2, n_sectors, n_axial), dtype=float)
    response_dofs = (ordering.parent_y_dofs.ravel(), ordering.parent_z_dofs.ravel())
    source_dofs = (ordering.parent_y_dofs[0], ordering.parent_z_dofs[0])

    rhs = A.createVecRight()
    displacement = A.createVecLeft()
    jobs = [(component, axial) for component in range(2) for axial in range(n_axial)]
    if show_progress:
        from tqdm import tqdm

        jobs = tqdm(jobs, desc="Sampling axial y/z compliance", unit="solve")

    for force_component, source_axial in jobs:
        rhs.set(0.0)
        rhs.setValue(
            int(source_dofs[force_component][source_axial]),
            PETSc.ScalarType(force_magnitude),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        rhs.assemble()
        ksp.solve(rhs, displacement)

        values = displacement.getArray(readonly=True)
        for response_component, dofs in enumerate(response_dofs):
            samples[force_component, source_axial, response_component] = (
                values[dofs].reshape(n_sectors, n_axial) / force_magnitude
            )

    return samples


def reconstruct_vertical_compliance(samples: np.ndarray) -> np.ndarray:
    """Rotate samples into dense ``S_c`` (small-problem validation only).

    Production contact examples should use
    :class:`DihedralComplianceEntrySource`; this helper intentionally
    reconstructs the global dense matrix and is retained for tests and
    numerical comparisons.
    """
    source = DihedralComplianceEntrySource(samples)
    indices = np.arange(source.shape[0], dtype=np.int64)
    return source.get_block(indices, indices)


def dihedral_reflection_error(samples: np.ndarray) -> float:
    """Return the relative D_n reflection mismatch of the sampled y/z tensors."""
    samples = np.asarray(samples, dtype=float)
    n_sectors = samples.shape[3]
    reflection = np.array([1.0, -1.0])
    mismatch_sq = 0.0
    norm_sq = 0.0
    for delta in range(n_sectors):
        mirrored = (-delta) % n_sectors
        expected = (
            samples[:, :, :, delta, :]
            * reflection[:, None, None, None]
            * reflection[None, None, :, None]
        )
        actual = samples[:, :, :, mirrored, :]
        mismatch_sq += float(np.linalg.norm(actual - expected) ** 2)
        norm_sq += float(np.linalg.norm(actual) ** 2)
    return float(np.sqrt(mismatch_sq / max(norm_sq, np.finfo(float).tiny)))

"""Compliance sampling for a tyre with discrete rotational symmetry.

The tyre axis is the global x axis.  A global road-normal (z) load does not
remain a z load when rotated to a reference meridian, so reconstruction needs
the 2x2 transverse (y/z) compliance tensor rather than a scalar cyclic shift.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """Create one reusable PETSc PREONLY+LU solver for compliance sampling."""
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

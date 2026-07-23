"""Compliance sampling for a tyre with discrete rotational symmetry.

The tyre axis is the global x axis. A global road-normal (z) load does not
remain a z load when rotated to a reference meridian. Reconstruction therefore
samples two auxiliary y/z load directions, but stores only the three tensor
combinations needed by the scalar normal compliance operator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time

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
    sector_angles: np.ndarray
    sector_angle_error: float
    structured: bool = True

    @property
    def n_sectors(self) -> int:
        return int(self.scalar_dofs.shape[0])

    @property
    def n_axial(self) -> int:
        return int(self.scalar_dofs.shape[1])


def order_unstructured_contact(
    points: np.ndarray,
    scalar_dofs: np.ndarray,
    parent_y_dofs: np.ndarray,
    parent_z_dofs: np.ndarray,
    *,
    axis_yz: tuple[float, float] = (0.0, 0.0),
) -> SectorOrdering:
    """Create deterministic flat ordering for an unstructured contact surface.

    The singleton leading dimension keeps the established force/DOF APIs
    compatible. It is not a physical circumferential sector and therefore must
    not be used by dihedral sampling or sector-spectral preconditioning.
    """
    points = np.asarray(points, dtype=float)
    scalar_dofs = np.asarray(scalar_dofs, dtype=np.int32).reshape(-1)
    parent_y_dofs = np.asarray(parent_y_dofs, dtype=np.int32).reshape(-1)
    parent_z_dofs = np.asarray(parent_z_dofs, dtype=np.int32).reshape(-1)
    n = len(points)
    if points.shape != (n, 3) or n == 0:
        raise ValueError("points must have nonzero shape (n, 3)")
    if any(
        values.shape != (n,)
        for values in (scalar_dofs, parent_y_dofs, parent_z_dofs)
    ):
        raise ValueError("all DOF arrays must have shape (n,)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain NaN or infinite values")
    y0, z0 = axis_yz
    angles = np.mod(
        np.arctan2(points[:, 2] - z0, points[:, 1] - y0),
        2.0 * np.pi,
    )
    radius = np.hypot(points[:, 1] - y0, points[:, 2] - z0)
    order = np.lexsort((radius, points[:, 0], angles))
    return SectorOrdering(
        scalar_dofs=scalar_dofs[order][None, :],
        parent_y_dofs=parent_y_dofs[order][None, :],
        parent_z_dofs=parent_z_dofs[order][None, :],
        points=points[order][None, :, :],
        sector_angles=np.empty(0, dtype=float),
        sector_angle_error=0.0,
        structured=False,
    )


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
        self.solve_cpu_seconds = 0.0
        self.linear_iterations = 0
        self.maximum_linear_iterations = 0

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
            iterations = int(self.ksp.getIterationNumber())
            self.linear_iterations += iterations
            self.maximum_linear_iterations = max(
                self.maximum_linear_iterations, iterations
            )
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


class FactorizedComplianceEntrySource(MatrixEntrySource):
    """Exact symmetric compliance entries from cached full-FE back-solves.

    Unlike the dihedral source, this implementation assumes no geometric
    symmetry.  It uses only elastic reciprocity: for each requested Cartesian
    block it solves whichever uncached set of source rows or columns is
    smaller, and caches the resulting contact-sized compliance columns.
    """

    def __init__(
        self,
        A: PETSc.Mat,
        ksp: PETSc.KSP,
        contact_dofs: np.ndarray,
    ) -> None:
        if A.comm.size != 1:
            raise NotImplementedError(
                "FactorizedComplianceEntrySource is currently serial"
            )
        dofs = np.asarray(contact_dofs, dtype=np.int32).reshape(-1)
        if dofs.size == 0 or np.unique(dofs).size != dofs.size:
            raise ValueError("contact_dofs must be non-empty and unique")
        if np.any(dofs < 0) or np.any(dofs >= A.getSize()[0]):
            raise IndexError("contact DOF outside the FE matrix")
        self.A = A
        self.ksp = ksp
        self.contact_dofs = dofs.copy()
        self.shape = (dofs.size, dofs.size)
        self._rhs = A.createVecRight()
        self._displacement = A.createVecLeft()
        self._columns: dict[int, np.ndarray] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset query/solve counters while retaining cached FE columns."""
        self.query_calls = 0
        self.queried_entries = 0
        self.largest_query = (0, 0)
        self.linear_solves = 0
        self.cache_hits = 0
        self.solve_seconds = 0.0
        self.solve_cpu_seconds = 0.0

    def _solve_column(self, source: int) -> None:
        if source in self._columns:
            self.cache_hits += 1
            return
        self._rhs.set(0.0)
        self._rhs.setValue(
            int(self.contact_dofs[source]),
            1.0,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        self._rhs.assemble()
        start = perf_counter()
        start_cpu = process_time()
        self.ksp.solve(self._rhs, self._displacement)
        self.solve_seconds += perf_counter() - start
        self.solve_cpu_seconds += process_time() - start_cpu
        self.linear_solves += 1
        if int(self.ksp.getConvergedReason()) <= 0:
            raise RuntimeError(
                "Full-FE compliance column solve failed with PETSc reason "
                f"{self.ksp.getConvergedReason()}"
            )
        values = self._displacement.getArray(readonly=True)
        self._columns[source] = np.array(
            values[self.contact_dofs], dtype=np.float64, copy=True
        )

    def get_block(
        self, row_indices: np.ndarray, column_indices: np.ndarray
    ) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        columns = np.asarray(column_indices, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("row index outside compliance source")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise IndexError("column index outside compliance source")
        self.query_calls += 1
        self.queried_entries += rows.size * columns.size
        self.largest_query = max(
            self.largest_query, (rows.size, columns.size), key=lambda x: x[0] * x[1]
        )
        missing_columns = sum(int(index) not in self._columns for index in columns)
        missing_rows = sum(int(index) not in self._columns for index in rows)
        if missing_columns <= missing_rows:
            for source in np.unique(columns):
                self._solve_column(int(source))
            return np.column_stack(
                [self._columns[int(source)][rows] for source in columns]
            )
        for source in np.unique(rows):
            self._solve_column(int(source))
        return np.row_stack(
            [self._columns[int(source)][columns] for source in rows]
        )

    def stats(self) -> dict[str, int | float | tuple[int, int]]:
        return {
            "query_calls": self.query_calls,
            "queried_entries": self.queried_entries,
            "largest_query": self.largest_query,
            "linear_solves": self.linear_solves,
            "cached_columns": len(self._columns),
            "cache_hits": self.cache_hits,
            "solve_seconds": self.solve_seconds,
            "solve_cpu_seconds": self.solve_cpu_seconds,
        }


class IterativeComplianceOperator(FactorizedComplianceOperator):
    """Compliance action backed by a strictly checked iterative FE solve."""

    def stats(self) -> dict[str, int | float]:
        values = super().stats()
        values.update(
            {
                "linear_iterations": self.linear_iterations,
                "maximum_linear_iterations": self.maximum_linear_iterations,
            }
        )
        return values


def probe_spd_operator(
    operator,
    *,
    seed: int = 1729,
) -> dict[str, float]:
    """Probe reciprocity and positive energy without materializing an operator."""
    rows, columns = operator.shape
    if rows != columns or not getattr(operator, "symmetric", False):
        raise ValueError("operator must declare a square symmetric action")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(columns)
    y = rng.standard_normal(columns)
    action_x = np.asarray(operator @ x, dtype=float)
    action_y = np.asarray(operator @ y, dtype=float)
    xy = float(x @ action_y)
    yx = float(y @ action_x)
    reciprocity = abs(xy - yx) / max(abs(xy), abs(yx), np.finfo(float).tiny)
    rayleigh_x = float(x @ action_x / (x @ x))
    rayleigh_y = float(y @ action_y / (y @ y))
    return {
        "reciprocity_error": reciprocity,
        "minimum_probed_rayleigh": min(rayleigh_x, rayleigh_y),
    }


class DihedralComplianceEntrySource(MatrixEntrySource):
    """Global-z normal compliance entries from one sampled axial meridian.

    New archives store only the three combinations used by the scalar normal
    operator: ``yy``, ``yz + zy``, and ``zz``. Legacy archives containing the
    complete 2x2 y/z tensor remain accepted. Rotation maps any requested
    global source/target pair back to that reference data, so ``get_block``
    can answer ACA cross queries without reconstructing the full ``S_c``.

    Matrix indices follow the sector-major, axial-minor ordering produced by
    :func:`order_contact_sectors`: rows are target nodes and columns are
    source nodes.
    """

    def __init__(self, samples: np.ndarray):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 4 and samples.shape[0] == 3:
            _, n_axial, n_sectors, target_axial = samples.shape
            self.compact_normal = True
        elif (
            samples.ndim == 5
            and samples.shape[0] == 2
            and samples.shape[2] == 2
        ):
            _, n_axial, _, n_sectors, target_axial = samples.shape
            self.compact_normal = False
        else:
            raise ValueError(
                "samples must have compact shape "
                "(3, n_axial, n_sectors, n_axial) or legacy shape "
                "(2, n_axial, 2, n_sectors, n_axial)"
            )
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
        if self.compact_normal:
            return (
                q[0] ** 2
                * self.samples[0, source_axial, delta, target_axial]
                + q[0]
                * q[1]
                * self.samples[1, source_axial, delta, target_axial]
                + q[1] ** 2
                * self.samples[2, source_axial, delta, target_axial]
            )

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


class LocalDihedralComplianceEntrySource(DihedralComplianceEntrySource):
    """Symmetry reconstruction restricted to an open regular angular patch.

    Unlike :class:`DihedralComplianceEntrySource`, this source does not wrap
    sector offsets around a full circle.  It assumes that translating a
    source/target pair between the equally spaced meridians of the tagged
    patch is a useful local approximation.  Entries below the diagonal are
    obtained by elastic reciprocity, so the resulting scalar operator is
    exactly symmetric even though the surrounding FE mesh need not possess
    the same rotational symmetry.

    This is an approximation whenever the structure outside the tagged patch
    is not rotationally invariant.  Call
    :func:`validate_local_dihedral_compliance` before using it in an SPD-only
    contact solver.
    """

    def __init__(self, samples: np.ndarray, *, sector_step: float):
        super().__init__(samples)
        if self.n_sectors < 2:
            raise ValueError("a local symmetry patch needs at least two meridians")
        if not np.isfinite(sector_step) or sector_step <= 0.0:
            raise ValueError("sector_step must be finite and positive")
        self.sector_step = float(sector_step)

    def _forward_evaluate(
        self, target: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
        """Evaluate pairs with target sector not before source sector."""
        target_sector = target // self.n_axial
        target_axial = target % self.n_axial
        source_sector = source // self.n_axial
        source_axial = source % self.n_axial
        delta = target_sector - source_sector
        angle = source_sector * self.sector_step
        qy = np.sin(angle)
        qz = np.cos(angle)
        if self.compact_normal:
            return (
                qy**2 * self.samples[0, source_axial, delta, target_axial]
                + qy
                * qz
                * self.samples[1, source_axial, delta, target_axial]
                + qz**2 * self.samples[2, source_axial, delta, target_axial]
            )

        values = np.zeros(np.broadcast_shapes(target.shape, source.shape), dtype=float)
        q = (qy, qz)
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

    def _evaluate(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        target, source = np.broadcast_arrays(
            np.asarray(target, dtype=np.int64),
            np.asarray(source, dtype=np.int64),
        )
        target_sector = target // self.n_axial
        source_sector = source // self.n_axial
        forward = target_sector >= source_sector
        values = np.empty(target.shape, dtype=float)
        if np.any(forward):
            values[forward] = self._forward_evaluate(
                target[forward], source[forward]
            )
        if np.any(~forward):
            # Maxwell-Betti reciprocity supplies the unsampled orientation.
            values[~forward] = self._forward_evaluate(
                source[~forward], target[~forward]
            )
        return values


class RestrictedLocalDihedralComplianceEntrySource(MatrixEntrySource):
    """Candidate-only local-symmetry compliance sample closure.

    The source stores only the axial indices and non-negative sector offsets
    needed by a fixed potential-contact set. Its matrix ordering is exactly
    the ordering of ``candidate_indices``; no full-patch principal-submatrix
    wrapper is required.
    """

    def __init__(
        self,
        samples: np.ndarray,
        candidate_indices: np.ndarray,
        *,
        full_n_axial: int,
        axial_indices: np.ndarray,
        sector_deltas: np.ndarray,
        sector_step: float,
    ) -> None:
        values = np.asarray(samples)
        candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
        axial = np.asarray(axial_indices, dtype=np.int64).reshape(-1)
        deltas = np.asarray(sector_deltas, dtype=np.int64).reshape(-1)
        if full_n_axial <= 0:
            raise ValueError("full_n_axial must be positive")
        if candidates.size == 0 or np.unique(candidates).size != candidates.size:
            raise ValueError("candidate_indices must be non-empty and unique")
        if axial.size == 0 or np.any(np.diff(axial) <= 0):
            raise ValueError("axial_indices must be sorted and unique")
        if deltas.size == 0 or deltas[0] != 0 or np.any(np.diff(deltas) <= 0):
            raise ValueError("sector_deltas must be sorted, unique, and include zero")
        if np.any(axial < 0) or np.any(axial >= full_n_axial):
            raise IndexError("axial index outside the full meridian")
        if np.any(candidates < 0):
            raise IndexError("candidate index must be non-negative")
        if values.shape != (3, axial.size, deltas.size, axial.size):
            raise ValueError(
                "samples must have shape "
                f"(3, {axial.size}, {deltas.size}, {axial.size})"
            )
        if not np.isfinite(sector_step) or sector_step <= 0.0:
            raise ValueError("sector_step must be finite and positive")

        candidate_axial = candidates % full_n_axial
        axial_lookup = np.full(full_n_axial, -1, dtype=np.int64)
        axial_lookup[axial] = np.arange(axial.size)
        if np.any(axial_lookup[candidate_axial] < 0):
            raise ValueError("candidate axial index is absent from sampled closure")
        delta_lookup = np.full(int(deltas[-1]) + 1, -1, dtype=np.int64)
        delta_lookup[deltas] = np.arange(deltas.size)

        self.samples = values
        self.candidate_indices = candidates.copy()
        self.full_n_axial = int(full_n_axial)
        self.axial_indices = axial.copy()
        self.sector_deltas = deltas.copy()
        self.sector_step = float(sector_step)
        self._candidate_sector = candidates // full_n_axial
        self._candidate_axial = candidate_axial
        self._axial_lookup = axial_lookup
        self._delta_lookup = delta_lookup
        self.shape = (candidates.size, candidates.size)
        self.reset_stats()

    def _forward_evaluate(
        self, target: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
        target_sector = self._candidate_sector[target]
        source_sector = self._candidate_sector[source]
        delta = target_sector - source_sector
        if np.any(delta < 0) or np.any(delta >= self._delta_lookup.size):
            raise IndexError("sector offset outside restricted sample closure")
        delta_local = self._delta_lookup[delta]
        if np.any(delta_local < 0):
            raise IndexError("sector offset was not sampled")
        source_axial = self._axial_lookup[self._candidate_axial[source]]
        target_axial = self._axial_lookup[self._candidate_axial[target]]
        angle = source_sector * self.sector_step
        qy = np.sin(angle)
        qz = np.cos(angle)
        return (
            qy**2 * self.samples[0, source_axial, delta_local, target_axial]
            + qy
            * qz
            * self.samples[1, source_axial, delta_local, target_axial]
            + qz**2 * self.samples[2, source_axial, delta_local, target_axial]
        )

    def _evaluate(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        target, source = np.broadcast_arrays(
            np.asarray(target, dtype=np.int64),
            np.asarray(source, dtype=np.int64),
        )
        target_sector = self._candidate_sector[target]
        source_sector = self._candidate_sector[source]
        forward = target_sector >= source_sector
        values = np.empty(target.shape, dtype=float)
        if np.any(forward):
            values[forward] = self._forward_evaluate(
                target[forward], source[forward]
            )
        if np.any(~forward):
            values[~forward] = self._forward_evaluate(
                source[~forward], target[~forward]
            )
        return values

    def get_block(
        self, row_indices: np.ndarray, column_indices: np.ndarray
    ) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        columns = np.asarray(column_indices, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("row index outside restricted compliance source")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise IndexError("column index outside restricted compliance source")
        self.query_calls += 1
        self.queried_entries += int(rows.size * columns.size)
        if rows.size * columns.size > self.largest_query[0] * self.largest_query[1]:
            self.largest_query = (int(rows.size), int(columns.size))
        return self._evaluate(rows[:, None], columns[None, :])

    def stats(self) -> dict[str, int | tuple[int, int]]:
        return {
            "query_calls": self.query_calls,
            "queried_entries": self.queried_entries,
            "largest_query": self.largest_query,
            "sampled_axial_nodes": int(self.axial_indices.size),
            "sampled_sector_deltas": int(self.sector_deltas.size),
            "sampled_entries": int(self.samples.size),
        }

    def reset_stats(self) -> None:
        self.query_calls = 0
        self.queried_entries = 0
        self.largest_query = (0, 0)


def restrict_local_dihedral_compliance_samples(
    samples: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    sector_step: float,
) -> RestrictedLocalDihedralComplianceEntrySource:
    """Copy the minimal rectangular sample closure for fixed candidates."""
    full = compact_normal_compliance_samples(samples)
    n_axial = int(full.shape[1])
    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if candidates.size == 0 or np.any(candidates >= full.shape[2] * n_axial):
        raise IndexError("candidate index outside full local-symmetry patch")
    sectors = np.unique(candidates // n_axial)
    axial = np.unique(candidates % n_axial)
    deltas = np.unique(np.abs(sectors[:, None] - sectors[None, :]))
    restricted = full[np.ix_(np.arange(3), axial, deltas, axial)].copy()
    return RestrictedLocalDihedralComplianceEntrySource(
        restricted,
        candidates,
        full_n_axial=n_axial,
        axial_indices=axial,
        sector_deltas=deltas,
        sector_step=sector_step,
    )

def validate_local_dihedral_compliance(
    A: PETSc.Mat,
    ksp: PETSc.KSP,
    ordering: SectorOrdering,
    source: LocalDihedralComplianceEntrySource,
    *,
    sample_columns: int = 4,
) -> dict[str, float | int]:
    """Compare selected reconstructed columns with direct FE back-solves."""
    if sample_columns <= 0:
        raise ValueError("sample_columns must be positive")
    if source.shape[0] != ordering.parent_z_dofs.size:
        raise ValueError("source and ordering sizes differ")
    count = min(int(sample_columns), source.shape[1])
    columns = np.unique(
        np.linspace(0, source.shape[1] - 1, count, dtype=np.int64)
    )
    exact_operator = FactorizedComplianceOperator(
        A,
        ksp,
        ordering.parent_z_dofs.ravel(),
    )
    rows = np.arange(source.shape[0], dtype=np.int64)
    error_sq = 0.0
    norm_sq = 0.0
    maximum_relative_column_error = 0.0
    for column in columns:
        unit = np.zeros(source.shape[1], dtype=float)
        unit[column] = 1.0
        exact = exact_operator @ unit
        approximate = source.get_block(rows, np.array([column]))[:, 0]
        error = np.linalg.norm(approximate - exact)
        norm = np.linalg.norm(exact)
        relative = float(error / max(norm, np.finfo(float).tiny))
        maximum_relative_column_error = max(
            maximum_relative_column_error, relative
        )
        error_sq += float(error**2)
        norm_sq += float(norm**2)
    return {
        "sample_columns": int(columns.size),
        "relative_frobenius_estimate": float(
            np.sqrt(error_sq / max(norm_sq, np.finfo(float).tiny))
        ),
        "maximum_relative_column_error": maximum_relative_column_error,
        "direct_fe_solves": int(exact_operator.linear_solves),
    }


def load_dihedral_compliance_archive(
    path: str | Path,
    points: np.ndarray,
    *,
    n_axial: int,
    n_sectors: int,
    circumferential_divisions: int | None = None,
    local_symmetry_tag: int | None = None,
    young_modulus: float | None = None,
    poisson_ratio: float | None = None,
    boundary_condition_id: str | None = None,
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
            missing = {"points"}.difference(archive.files)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Compliance archive {archive_path} is missing: {names}"
                )
            if "samples_file" in archive.files:
                stored_path = np.asarray(archive["samples_file"])
                if stored_path.size != 1:
                    raise ValueError(
                        "Compliance archive field 'samples_file' must be scalar"
                    )
                samples_path = Path(str(stored_path.reshape(-1)[0])).expanduser()
                if not samples_path.is_absolute():
                    samples_path = archive_path.parent / samples_path
                if not samples_path.is_file():
                    raise ValueError(
                        "Compliance sample sidecar does not exist: "
                        f"{samples_path.resolve()}"
                    )
                samples = np.load(samples_path, mmap_mode="r", allow_pickle=False)
            elif "samples" in archive.files:
                # Accessing an NPZ member already materializes a standalone
                # ndarray; an additional copy would double peak memory for
                # multi-gigabyte reference tensors.
                samples = np.asarray(archive["samples"], dtype=float)
            else:
                raise ValueError(
                    f"Compliance archive {archive_path} is missing: "
                    "samples or samples_file"
                )
            saved_points = np.array(archive["points"], dtype=float, copy=True)

            metadata: dict[str, object] = {}
            for name in (
                "axial_divisions",
                "circumferential_divisions",
                "young_modulus",
                "poisson_ratio",
                "boundary_condition_id",
                "local_symmetry_tag",
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
        "circumferential_divisions": (
            n_sectors
            if circumferential_divisions is None
            else circumferential_divisions
        ),
    }
    for name, expected in expected_divisions.items():
        if name in metadata and int(metadata[name]) != expected:
            raise ValueError(
                f"Loaded compliance {name}={metadata[name]} does not match "
                f"the current value {expected}"
            )

    if local_symmetry_tag is not None:
        if "local_symmetry_tag" not in metadata:
            raise ValueError(
                "Loaded compliance archive has no local_symmetry_tag metadata"
            )
        if int(metadata["local_symmetry_tag"]) != int(local_symmetry_tag):
            raise ValueError(
                "Loaded compliance local_symmetry_tag="
                f"{metadata['local_symmetry_tag']} does not match the current "
                f"value {local_symmetry_tag}"
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

    if boundary_condition_id is not None:
        saved_boundary_condition = metadata.get("boundary_condition_id")
        if saved_boundary_condition is None:
            raise ValueError(
                "Loaded compliance archive does not identify its boundary "
                "condition and cannot be reused safely"
            )
        if str(saved_boundary_condition) != boundary_condition_id:
            raise ValueError(
                "Loaded compliance boundary_condition_id="
                f"{saved_boundary_condition!r} does not match the current "
                f"value {boundary_condition_id!r}"
            )

    return samples


def compact_normal_compliance_samples(samples: np.ndarray) -> np.ndarray:
    """Return the three tensor combinations needed by normal contact.

    The compact component axis contains ``yy``, ``yz + zy``, and ``zz``.
    Contracting these values with ``(sin(theta), cos(theta))`` is exactly
    equivalent to contracting the complete legacy 2x2 tensor. No tangential
    contact operator is retained.
    """
    values = np.asarray(samples)
    if values.ndim == 4 and values.shape[0] == 3:
        if values.shape[1] != values.shape[3]:
            raise ValueError("source and target axial sizes must match")
        return values
    if (
        values.ndim != 5
        or values.shape[0] != 2
        or values.shape[2] != 2
        or values.shape[1] != values.shape[4]
    ):
        raise ValueError(
            "samples must have compact shape "
            "(3, n_axial, n_sectors, n_axial) or legacy shape "
            "(2, n_axial, 2, n_sectors, n_axial)"
        )

    compact = np.empty(
        (3, values.shape[1], values.shape[3], values.shape[4]),
        dtype=np.result_type(values.dtype, np.float64),
    )
    compact[0] = values[0, :, 0]
    np.add(values[0, :, 1], values[1, :, 0], out=compact[1])
    compact[2] = values[1, :, 1]
    return compact


def memory_map_compliance_samples(
    samples: np.ndarray,
    path: str | Path,
) -> np.memmap:
    """Persist compact normal samples as NPY and reopen them as a memmap."""
    values = np.asarray(samples)
    if values.ndim == 5:
        if (
            values.shape[0] != 2
            or values.shape[2] != 2
            or values.shape[1] != values.shape[4]
        ):
            compact_normal_compliance_samples(values)
        compact_values = None
        compact_shape = (3, values.shape[1], values.shape[3], values.shape[4])
    else:
        compact_values = compact_normal_compliance_samples(values)
        compact_shape = compact_values.shape
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    current_filename = getattr(samples, "filename", None)
    source_is_output = (
        current_filename is not None
        and Path(current_filename).resolve() == output
    )
    if (
        source_is_output
        and values.ndim == 4
        and values.dtype == np.float64
    ):
        mapped = np.load(output, mmap_mode="r", allow_pickle=False)
        if not isinstance(mapped, np.memmap):
            raise RuntimeError("NumPy did not reopen compliance samples as a memmap")
        return mapped

    destination = (
        output.with_name(f".{output.name}.compact.tmp")
        if source_is_output
        else output
    )
    mapped_output = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float64,
        shape=compact_shape,
    )
    if values.ndim == 5:
        mapped_output[0] = values[0, :, 0]
        np.add(values[0, :, 1], values[1, :, 0], out=mapped_output[1])
        mapped_output[2] = values[1, :, 1]
    else:
        mapped_output[:] = compact_values
    mapped_output.flush()
    del mapped_output
    if destination != output:
        destination.replace(output)
    mapped = np.load(output, mmap_mode="r", allow_pickle=False)
    if not isinstance(mapped, np.memmap):
        raise RuntimeError("NumPy did not reopen compliance samples as a memmap")
    return mapped


def infer_meridian_shape(
    points: np.ndarray,
    *,
    axis_yz: tuple[float, float] = (0.0, 0.0),
    angle_tol: float = 1.0e-8,
) -> tuple[np.ndarray, int]:
    """Infer sorted meridian angles and the common nodes-per-meridian count.

    The points need not be ordered or equally spaced, but every meridian must
    contain the same number of points.
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
            "cannot infer meridians because meridian point counts differ: "
            f"{sorted(np.unique(counts).tolist())}"
        )

    centers = np.mod(
        np.array([np.mean(cluster) for cluster in clusters]), 2.0 * np.pi
    )
    centers.sort()
    if len(centers) < 2:
        raise ValueError("at least two meridians are required")
    return centers, int(counts[0])


def infer_regular_sector_shape(
    points: np.ndarray,
    *,
    axis_yz: tuple[float, float] = (0.0, 0.0),
    angle_tol: float = 1.0e-8,
) -> tuple[int, int]:
    """Infer equal sector and axial-node counts from a revolved point set."""
    centers, nodes_per_sector = infer_meridian_shape(
        points, axis_yz=axis_yz, angle_tol=angle_tol
    )
    n_sectors = len(centers)
    gaps = np.diff(np.concatenate([centers, centers[:1] + 2.0 * np.pi]))
    expected_gap = 2.0 * np.pi / n_sectors
    if not np.allclose(gaps, expected_gap, rtol=0.0, atol=2.0 * angle_tol):
        raise ValueError(
            "point meridians are not equally spaced: maximum sector-gap error is "
            f"{np.max(np.abs(gaps - expected_gap)):.3e} rad"
        )
    return n_sectors, nodes_per_sector


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


def order_contact_meridians(
    points: np.ndarray,
    scalar_dofs: np.ndarray,
    parent_y_dofs: np.ndarray,
    parent_z_dofs: np.ndarray,
    sector_angles: np.ndarray,
    *,
    axis_yz: tuple[float, float] = (0.0, 0.0),
    angle_tol: float = 1.0e-8,
    geometry_tol: float | None = 1.0e-9,
) -> SectorOrdering:
    """Group a surface of revolution on prescribed angular meridians.

    Set ``geometry_tol=None`` for a locally structured but geometrically
    approximate patch whose axial/radial rows are paired only by sorted axial
    index. The regular full-circle path should retain the strict default.
    """
    points = np.asarray(points, dtype=float)
    scalar_dofs = np.asarray(scalar_dofs, dtype=np.int32)
    parent_y_dofs = np.asarray(parent_y_dofs, dtype=np.int32)
    parent_z_dofs = np.asarray(parent_z_dofs, dtype=np.int32)
    n = len(points)
    if points.shape != (n, 3):
        raise ValueError("points must have shape (n, 3)")
    if any(array.shape != (n,) for array in (scalar_dofs, parent_y_dofs, parent_z_dofs)):
        raise ValueError("all DOF arrays must have shape (n,)")
    expected_angles = np.mod(
        np.asarray(sector_angles, dtype=float).reshape(-1), 2.0 * np.pi
    )
    if expected_angles.size < 2 or not np.all(np.isfinite(expected_angles)):
        raise ValueError("sector_angles must contain at least two finite angles")
    expected_angles.sort()
    circular_gaps = np.diff(
        np.concatenate([expected_angles, expected_angles[:1] + 2.0 * np.pi])
    )
    if np.any(circular_gaps <= 2.0 * angle_tol):
        raise ValueError("sector_angles must be unique")
    n_sectors = expected_angles.size

    y0, z0 = axis_yz
    angles = np.mod(np.arctan2(points[:, 2] - z0, points[:, 1] - y0), 2.0 * np.pi)
    insertion = np.searchsorted(expected_angles, angles)
    right = insertion % n_sectors
    left = (insertion - 1) % n_sectors
    right_error = np.abs(
        np.arctan2(
            np.sin(angles - expected_angles[right]),
            np.cos(angles - expected_angles[right]),
        )
    )
    left_error = np.abs(
        np.arctan2(
            np.sin(angles - expected_angles[left]),
            np.cos(angles - expected_angles[left]),
        )
    )
    sectors = np.where(left_error <= right_error, left, right)
    expected = expected_angles[sectors]
    angle_error = np.abs(np.arctan2(np.sin(angles - expected), np.cos(angles - expected)))
    max_angle_error = float(angle_error.max(initial=0.0))
    if max_angle_error > angle_tol:
        raise ValueError(
            "Contact nodes do not lie on the requested meridians: "
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
    if geometry_tol is not None:
        if geometry_tol < 0.0:
            raise ValueError("geometry_tol must be non-negative or None")
        reference_x = ordered_points[0, :, 0]
        reference_radius = np.linalg.norm(
            ordered_points[0, :, 1:] - np.array([y0, z0]), axis=1
        )
        for sector in range(1, n_sectors):
            radius = np.linalg.norm(
                ordered_points[sector, :, 1:] - np.array([y0, z0]), axis=1
            )
            if not np.allclose(
                ordered_points[sector, :, 0],
                reference_x,
                atol=geometry_tol,
                rtol=0.0,
            ):
                raise ValueError(f"Axial coordinates differ in sector {sector}")
            if not np.allclose(
                radius, reference_radius, atol=geometry_tol, rtol=0.0
            ):
                raise ValueError(f"Radial coordinates differ in sector {sector}")

    return SectorOrdering(
        scalar_dofs=scalar_dofs[ordered_indices],
        parent_y_dofs=parent_y_dofs[ordered_indices],
        parent_z_dofs=parent_z_dofs[ordered_indices],
        points=ordered_points,
        sector_angles=expected_angles,
        sector_angle_error=max_angle_error,
    )


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
    if n_sectors < 2:
        raise ValueError("n_sectors must be >= 2")
    return order_contact_meridians(
        points,
        scalar_dofs,
        parent_y_dofs,
        parent_z_dofs,
        np.arange(n_sectors, dtype=float) * (2.0 * np.pi / n_sectors),
        axis_yz=axis_yz,
        angle_tol=angle_tol,
        geometry_tol=geometry_tol,
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


def create_iterative_solver(
    A: PETSc.Mat,
    comm: MPI.Comm,
    *,
    coordinates: np.ndarray | None = None,
    ksp_type: str = "cg",
    pc_type: str = "gamg",
    relative_tolerance: float = 1.0e-10,
    absolute_tolerance: float = 1.0e-14,
    max_iterations: int = 2000,
    options_prefix: str = "cofebem_fe_",
) -> PETSc.KSP:
    """Create a reusable SPD iterative solver for compliance actions.

    ``coordinates`` must contain one three-dimensional point per vector CG1
    node.  GAMG receives both these coordinates and PETSc's six rigid-body
    near-nullspace modes, which are important for elasticity coarsening.
    """
    if relative_tolerance <= 0.0 or not np.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and positive")
    if absolute_tolerance < 0.0 or not np.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not options_prefix or not options_prefix.endswith("_"):
        raise ValueError("options_prefix must be non-empty and end in '_'")

    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    A.setOption(PETSc.Mat.Option.SPD, True)
    coordinate_array = None
    near_nullspace = None
    if coordinates is not None:
        coordinate_array = np.asarray(coordinates, dtype=PETSc.RealType)
        if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
            raise ValueError("coordinates must have shape (n_nodes, 3)")
        if coordinate_array.size != A.getLocalSize()[0]:
            raise ValueError(
                "three coordinate values per node must match the local FE rows"
            )
        coordinate_vector = A.createVecRight()
        coordinate_vector.setBlockSize(3)
        coordinate_vector.getArray()[:] = coordinate_array.reshape(-1)
        near_nullspace = PETSc.NullSpace().createRigidBody(coordinate_vector)
        A.setNearNullSpace(near_nullspace)
        coordinate_vector.destroy()

    ksp = PETSc.KSP().create(comm)
    ksp.setOptionsPrefix(options_prefix)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setInitialGuessNonzero(False)
    ksp.setTolerances(
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_it=max_iterations,
    )
    pc = ksp.getPC()
    pc.setType(pc_type)
    if coordinate_array is not None:
        pc.setCoordinates(coordinate_array)
    ksp.setFromOptions()
    ksp.setUp()
    # PETSc objects retain the attached near-nullspace after setup.
    if near_nullspace is not None:
        near_nullspace.destroy()
    return ksp


def sample_reference_normal_compliance(
    A: PETSc.Mat,
    ksp: PETSc.KSP,
    ordering: SectorOrdering,
    *,
    force_magnitude: float = 1.0,
    show_progress: bool = True,
) -> np.ndarray:
    """Sample the compact scalar normal operator at one axial meridian.

    The returned axes are ``(normal_component, source_axial, target_sector,
    target_axial)``. The three normal components are ``yy``, ``yz + zy``, and
    ``zz``. Two auxiliary y/z load solves remain necessary because a fixed
    global-z road load rotates into both directions away from sector zero,
    but the unused antisymmetric part of the transverse response is discarded.
    """
    if A.comm.size != 1:
        raise NotImplementedError(
            "Reference compliance sampling is currently serial; distributed "
            "global DOF gathering has not been implemented."
        )
    if force_magnitude == 0.0:
        raise ValueError("force_magnitude must be nonzero")

    n_sectors, n_axial = ordering.scalar_dofs.shape
    samples = np.zeros((3, n_axial, n_sectors, n_axial), dtype=float)
    response_dofs = (ordering.parent_y_dofs.ravel(), ordering.parent_z_dofs.ravel())
    source_dofs = (ordering.parent_y_dofs[0], ordering.parent_z_dofs[0])

    rhs = A.createVecRight()
    displacement = A.createVecLeft()
    jobs = [(component, axial) for component in range(2) for axial in range(n_axial)]
    if show_progress:
        from tqdm import tqdm

        jobs = tqdm(jobs, desc="Sampling axial normal compliance", unit="solve")

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
        response_y = (
            values[response_dofs[0]].reshape(n_sectors, n_axial)
            / force_magnitude
        )
        response_z = (
            values[response_dofs[1]].reshape(n_sectors, n_axial)
            / force_magnitude
        )
        if force_component == 0:
            samples[0, source_axial] = response_y
            samples[1, source_axial] = response_z
        else:
            samples[1, source_axial] += response_y
            samples[2, source_axial] = response_z

    return samples


def sample_reference_transverse_compliance(
    A: PETSc.Mat,
    ksp: PETSc.KSP,
    ordering: SectorOrdering,
    *,
    force_magnitude: float = 1.0,
    show_progress: bool = True,
) -> np.ndarray:
    """Compatibility alias for :func:`sample_reference_normal_compliance`."""
    return sample_reference_normal_compliance(
        A,
        ksp,
        ordering,
        force_magnitude=force_magnitude,
        show_progress=show_progress,
    )


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
    """Return the relative D_n reflection mismatch of sampled normal data."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 4 and samples.shape[0] == 3:
        n_sectors = samples.shape[2]
        component_parity = np.array([1.0, -1.0, 1.0])
        sector_axis = 2
    elif samples.ndim == 5 and samples.shape[0] == 2 and samples.shape[2] == 2:
        n_sectors = samples.shape[3]
        reflection = np.array([1.0, -1.0])
        sector_axis = 3
    else:
        raise ValueError("invalid normal-compliance sample shape")
    mismatch_sq = 0.0
    norm_sq = 0.0
    for delta in range(n_sectors):
        mirrored = (-delta) % n_sectors
        if sector_axis == 2:
            expected = samples[:, :, delta, :] * component_parity[:, None, None]
            actual = samples[:, :, mirrored, :]
        else:
            expected = (
                samples[:, :, :, delta, :]
                * reflection[:, None, None, None]
                * reflection[None, None, :, None]
            )
            actual = samples[:, :, :, mirrored, :]
        mismatch_sq += float(np.linalg.norm(actual - expected) ** 2)
        norm_sq += float(np.linalg.norm(actual) ** 2)
    return float(np.sqrt(mismatch_sq / max(norm_sq, np.finfo(float).tiny)))

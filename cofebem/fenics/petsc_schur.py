"""Optional native PETSc/MUMPS selected-Schur compliance operator."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from petsc4py import PETSc


def selected_schur_memory_estimate(
    size: int, *, storage_copies: int = 3
) -> int:
    """Return a conservative byte estimate for a dense Schur factor."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if storage_copies < 1:
        raise ValueError("storage_copies must be positive")
    return int(storage_copies * size * size * np.dtype(PETSc.ScalarType).itemsize)


def _native_module():
    try:
        from cofebem.fenics import _petsc_schur
    except ImportError as exc:
        raise RuntimeError(
            "The mumps_schur strategy needs the optional native bridge. "
            "Build it inside fenicsx-env with "
            "COFEBEM_BUILD_PETSC_SCHUR=1 python -m pip install "
            "--no-build-isolation -e ."
        ) from exc
    return _petsc_schur


class MumpsSchurComplianceOperator:
    """Apply ``(K^{-1})[contact, contact]`` through a selected Schur solve.

    PETSc/MUMPS stores and factors the condensed contact stiffness.  The
    compliance itself is never explicitly inverted or materialized.
    """

    def __init__(
        self,
        A: PETSc.Mat,
        contact_dofs: np.ndarray,
        *,
        factor_type: str = "lu",
        max_memory_gib: float = 4.0,
    ) -> None:
        if A.comm.size != 1:
            raise NotImplementedError(
                "MumpsSchurComplianceOperator is currently serial"
            )
        dofs = np.asarray(contact_dofs, dtype=PETSc.IntType).reshape(-1)
        if dofs.size == 0:
            raise ValueError("contact_dofs must not be empty")
        if np.unique(dofs).size != dofs.size:
            raise ValueError("contact_dofs must be unique")
        rows, columns = A.getSize()
        if rows != columns:
            raise ValueError("A must be square")
        if np.any(dofs < 0) or np.any(dofs >= rows):
            raise IndexError("contact DOF outside the FE matrix")
        if factor_type not in {"lu", "cholesky"}:
            raise ValueError("factor_type must be 'lu' or 'cholesky'")
        if not np.isfinite(max_memory_gib) or max_memory_gib <= 0.0:
            raise ValueError("max_memory_gib must be finite and positive")

        estimated_bytes = selected_schur_memory_estimate(dofs.size)
        limit_bytes = int(max_memory_gib * 2**30)
        if estimated_bytes > limit_bytes:
            raise MemoryError(
                "Selected Schur factor is estimated to require "
                f"{estimated_bytes / 2**30:.3f} GiB for {dofs.size} unknowns, "
                f"above the configured {max_memory_gib:.3f} GiB limit. "
                "Reduce warning_distance, raise schur_max_memory_gib, or use "
                "fe_matrix_free/fe_iterative."
            )

        native = _native_module()
        schur_is = PETSc.IS().createGeneral(dofs, comm=A.comm)
        start = perf_counter()
        self._factor = native.create_factor(A, schur_is, factor_type)
        self.factorization_seconds = perf_counter() - start
        schur_is.destroy()

        self._native = native
        self.A = A
        self.contact_dofs = dofs.copy()
        self.shape = (dofs.size, dofs.size)
        self.symmetric = True
        self.estimated_memory_bytes = estimated_bytes
        self._schur_rhs = PETSc.Vec().createSeq(dofs.size, comm=PETSc.COMM_SELF)
        self._schur_solution = self._schur_rhs.duplicate()
        self._full_rhs = A.createVecRight()
        self._full_solution = A.createVecLeft()
        self._last_forces: np.ndarray | None = None
        self.operator_applications = 0
        self.schur_solves = 0
        self.full_solves = 0
        self.cache_hits = 0
        self.zero_bypasses = 0
        self.solve_seconds = 0.0

    def _set_schur_values(self, values: np.ndarray) -> None:
        array = self._schur_rhs.getArray()
        array[:] = values

    def _solve_schur(self, forces: np.ndarray) -> None:
        if self._last_forces is not None and np.array_equal(
            forces, self._last_forces
        ):
            self.cache_hits += 1
            return
        if not np.any(forces):
            self._schur_solution.set(0.0)
            self.zero_bypasses += 1
        else:
            self._set_schur_values(forces)
            start = perf_counter()
            self._native.solve_schur(
                self._factor, self._schur_rhs, self._schur_solution
            )
            self.solve_seconds += perf_counter() - start
            self.schur_solves += 1
        self._last_forces = forces.copy()

    def apply(
        self,
        forces: np.ndarray,
        *,
        response_dofs: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(forces, dtype=np.float64).reshape(-1)
        if values.shape != self.shape[1:]:
            raise ValueError(
                f"forces must have shape ({self.shape[1]},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("forces contain NaN or infinite values")
        self.operator_applications += 1
        if response_dofs is None or np.array_equal(
            np.asarray(response_dofs).reshape(-1), self.contact_dofs
        ):
            self._solve_schur(values)
            return np.array(
                self._schur_solution.getArray(readonly=True), copy=True
            )

        self._full_rhs.set(0.0)
        self._full_rhs.setValues(
            self.contact_dofs, values, addv=PETSc.InsertMode.INSERT_VALUES
        )
        self._full_rhs.assemble()
        self.solve(self._full_rhs, self._full_solution)
        indices = np.asarray(response_dofs, dtype=PETSc.IntType).reshape(-1)
        return np.array(
            self._full_solution.getArray(readonly=True)[indices], copy=True
        )

    def __matmul__(self, forces: np.ndarray) -> np.ndarray:
        return self.apply(forces)

    def solve(self, rhs: PETSc.Vec, solution: PETSc.Vec) -> None:
        """Solve the full FE system with the same MUMPS factor."""
        start = perf_counter()
        self._native.solve_full(self._factor, rhs, solution)
        self.solve_seconds += perf_counter() - start
        self.full_solves += 1

    def stats(self) -> dict[str, int | float]:
        return {
            "operator_applications": self.operator_applications,
            "linear_solves": self.schur_solves,
            "schur_solves": self.schur_solves,
            "full_solves": self.full_solves,
            "cache_hits": self.cache_hits,
            "zero_bypasses": self.zero_bypasses,
            "solve_seconds": self.solve_seconds,
            "factorization_seconds": self.factorization_seconds,
            "estimated_memory_bytes": self.estimated_memory_bytes,
        }

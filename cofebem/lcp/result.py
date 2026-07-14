from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .exceptions import InvalidLCPError
from .problem import Vector


class LCPStatus(str, Enum):
    """
    Termination status of an LCP solver.

    A ``str``/``Enum`` hybrid: members compare equal to their string
    ``value`` (e.g. ``LCPStatus.CONVERGED == "converged"``).

    Attributes
    ----------
    CONVERGED : str
        The solver's convergence criterion was satisfied.
    MAX_ITERATIONS : str
        The iteration limit was reached before the convergence criterion
        was satisfied.
    STAGNATION : str
        The iterate stopped changing (within tolerance) before the
        convergence criterion was satisfied.
    NUMERICAL_BREAKDOWN : str
        A numerical condition required for the algorithm to proceed (e.g.
        positive curvature, a feasible step) was violated.
    RAY_TERMINATION : str
        A pivoting method (e.g. :func:`cofebem.lcp.solvers.lemke`)
        terminated on a secondary ray: no solution is reachable from the
        chosen starting basis.
    """

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    STAGNATION = "stagnation"
    NUMERICAL_BREAKDOWN = "numerical_breakdown"
    RAY_TERMINATION = "ray_termination"


@dataclass(frozen=True, slots=True)
class LCPResult:
    """
    Outcome of running a solver on an LCP(M, q).

    Parameters
    ----------
    z : array_like of shape (n,)
        Candidate primal solution.
    w : array_like of shape (n,)
        Candidate complementary variable, ``w = M @ z + q``.
    status : LCPStatus
        How the iteration terminated.
    iterations : int
        Number of iterations performed by the solver.
    residual : float
        Final value of the solver's scalar convergence criterion, typically
        ``norm(min(z, w), inf)`` (some solvers report a normalized variant;
        see the solver's own docstring).
    residual_history : array_like of shape (k,), optional
        Per-iteration residual values, if requested from the solver via
        ``record_history=True``. ``None`` otherwise.
    message : str, optional
        Human-readable summary of the outcome.

    Attributes
    ----------
    z : ndarray of shape (n,)
    w : ndarray of shape (n,)
    status : LCPStatus
    iterations : int
    residual : float
    residual_history : ndarray of shape (k,) or None
    message : str

    Raises
    ------
    InvalidLCPError
        If ``z`` and ``w`` do not have the same shape, if ``iterations`` is
        negative, or if ``residual`` is negative.
    """

    z: Vector
    w: Vector
    status: LCPStatus
    iterations: int
    residual: float
    residual_history: Vector | None = None
    message: str = ""

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=np.float64).reshape(-1)
        w = np.asarray(self.w, dtype=np.float64).reshape(-1)

        if z.shape != w.shape:
            raise InvalidLCPError(
                f"z and w must have the same shape, got {z.shape} and {w.shape}."
            )

        if self.iterations < 0:
            raise InvalidLCPError(f"iterations must be >= 0, got {self.iterations}.")

        if self.residual < 0.0:
            raise InvalidLCPError(f"residual must be >= 0, got {self.residual}.")

        residual_history = self.residual_history
        if residual_history is not None:
            residual_history = np.asarray(residual_history, dtype=np.float64).reshape(
                -1
            )

        object.__setattr__(self, "z", z)
        object.__setattr__(self, "w", w)
        object.__setattr__(self, "residual", float(self.residual))
        object.__setattr__(self, "residual_history", residual_history)

    @property
    def size(self) -> int:
        """int: The number of variables ``n`` (i.e. ``z.shape[0]``)."""
        return self.z.shape[0]

    @property
    def converged(self) -> bool:
        """bool: Whether ``status`` is :attr:`LCPStatus.CONVERGED`."""
        return self.status is LCPStatus.CONVERGED

    @property
    def primal_violation(self) -> float:
        """float: ``max(0, -min(z))``, the violation of ``z >= 0``."""
        return max(0.0, -float(np.min(self.z)))

    @property
    def dual_violation(self) -> float:
        """float: ``max(0, -min(w))``, the violation of ``w >= 0``."""
        return max(0.0, -float(np.min(self.w)))

    @property
    def complementarity(self) -> float:
        """float: ``norm(z * w, inf)``, the violation of ``z_i * w_i = 0``."""
        return float(np.linalg.norm(self.z * self.w, ord=np.inf))

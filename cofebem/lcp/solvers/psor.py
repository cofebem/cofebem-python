from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..exceptions import InvalidSolverOptionError, UnsupportedMatrixError
from ..problem import LCP, Vector
from ..result import LCPResult, LCPStatus

__all__ = ["psor", "pgs"]


def psor(
    problem: LCP,
    *,
    omega: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 10000,
    z0: Vector | None = None,
    record_history: bool = False,
) -> LCPResult:
    """
    Solve an LCP(M, q) with Projected Successive Over-Relaxation.

    Each Gauss-Seidel sweep updates::

        z_i <- max(0, (1 - omega) * z_i + omega * (-q_i - sum_{j!=i} M_ij z_j) / M_ii)

    using the most recently updated components, and the iteration stops once
    ``norm(min(z, M @ z + q), inf) < tol``.

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem ``LCP(M, q)`` to solve. ``M``
        must have a nonzero diagonal.
    omega : float, optional
        Relaxation factor, must lie in ``(0, 2)``. ``omega = 1`` recovers
        Projected Gauss-Seidel; see :func:`pgs`.
    tol : float, optional
        Convergence tolerance on ``norm(min(z, w), inf)``, and on the
        relative change ``norm(z - z_prev, inf)`` used for stagnation
        detection.
    max_iter : int, optional
        Maximum number of sweeps.
    z0 : array_like of shape (n,), optional
        Initial guess. Negative entries are projected to zero. Defaults to
        the zero vector.
    record_history : bool, optional
        If True, record the residual after every sweep in
        ``result.residual_history``.

    Returns
    -------
    LCPResult
        The solver outcome. ``status`` is one of
        :attr:`LCPStatus.CONVERGED`, :attr:`LCPStatus.STAGNATION`, or
        :attr:`LCPStatus.MAX_ITERATIONS`.

    Raises
    ------
    InvalidSolverOptionError
        If ``omega`` is not in ``(0, 2)``, ``tol <= 0``, ``max_iter <= 0``,
        or ``z0`` does not have shape ``(n,)``.
    UnsupportedMatrixError
        If ``M`` has a zero diagonal entry.

    See Also
    --------
    pgs : Projected Gauss-Seidel, i.e. PSOR with ``omega = 1``.
    """
    if not (0.0 < omega < 2.0):
        raise InvalidSolverOptionError(f"omega must lie in (0, 2), got {omega}.")
    if tol <= 0.0:
        raise InvalidSolverOptionError(f"tol must be > 0, got {tol}.")
    if max_iter <= 0:
        raise InvalidSolverOptionError(f"max_iter must be > 0, got {max_iter}.")

    M = problem.M
    q = problem.q
    n = problem.size

    D = np.diag(M)
    if np.any(D == 0.0):
        raise UnsupportedMatrixError("PSOR requires a matrix with a nonzero diagonal.")

    if z0 is None:
        z = np.zeros(n, dtype=np.float64)
    else:
        z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
        if z0.shape != (n,):
            raise InvalidSolverOptionError(
                f"z0 must have shape ({n},), got {z0.shape}."
            )
        z = np.maximum(z0, 0.0)

    w = M @ z + q
    residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf))

    history: list[float] = []
    status = LCPStatus.MAX_ITERATIONS
    iterations = 0

    for k in range(max_iter):
        iterations = k + 1
        z_prev = z.copy()

        for i in range(n):
            s1 = M[i, :i] @ z[:i]  # already-updated values
            s2 = M[i, i + 1 :] @ z_prev[i + 1 :]  # not-yet-updated values
            z_gs = (-q[i] - s1 - s2) / D[i]
            z[i] = max(0.0, (1.0 - omega) * z_prev[i] + omega * z_gs)

        w = M @ z + q
        residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf))
        if record_history:
            history.append(residual)

        if residual < tol:
            status = LCPStatus.CONVERGED
            break

        if np.linalg.norm(z - z_prev, ord=np.inf) < tol * 1e-3:
            status = LCPStatus.STAGNATION
            break

    return LCPResult(
        z=z,
        w=w,
        status=status,
        iterations=iterations,
        residual=residual,
        residual_history=np.asarray(history) if record_history else None,
        message=f"PSOR {status.value} after {iterations} iteration(s) "
        f"(residual={residual:.3e}).",
    )


def pgs(
    problem: LCP,
    *,
    tol: float = 1e-10,
    max_iter: int = 10000,
    z0: Vector | None = None,
    record_history: bool = False,
) -> LCPResult:
    """
    Solve an LCP(M, q) with Projected Gauss-Seidel.

    This is :func:`psor` with ``omega = 1``.

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem ``LCP(M, q)`` to solve. ``M``
        must have a nonzero diagonal.
    tol : float, optional
        Convergence tolerance on ``norm(min(z, w), inf)``, and on the
        relative change ``norm(z - z_prev, inf)`` used for stagnation
        detection.
    max_iter : int, optional
        Maximum number of sweeps.
    z0 : array_like of shape (n,), optional
        Initial guess. Negative entries are projected to zero. Defaults to
        the zero vector.
    record_history : bool, optional
        If True, record the residual after every sweep in
        ``result.residual_history``.

    Returns
    -------
    LCPResult
        The solver outcome, with ``message`` referring to "PGS" instead of
        "PSOR". See :func:`psor` for the meaning of ``status``.

    Raises
    ------
    InvalidSolverOptionError
        If ``tol <= 0``, ``max_iter <= 0``, or ``z0`` does not have shape
        ``(n,)``.
    UnsupportedMatrixError
        If ``M`` has a zero diagonal entry.

    See Also
    --------
    psor : Projected Successive Over-Relaxation with a tunable ``omega``.
    """
    result = psor(
        problem,
        omega=1.0,
        tol=tol,
        max_iter=max_iter,
        z0=z0,
        record_history=record_history,
    )
    return replace(result, message=result.message.replace("PSOR", "PGS", 1))

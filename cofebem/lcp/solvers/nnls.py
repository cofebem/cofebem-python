from __future__ import annotations

import numpy as np

from ..exceptions import InvalidSolverOptionError, UnsupportedMatrixError
from ..problem import LCP, Matrix, Vector
from ..result import LCPResult, LCPStatus

__all__ = ["nnls", "nnls_lawson_hanson"]


def nnls_lawson_hanson(
    A: Matrix, b: Vector, tol: float = 1e-12, max_iter: int | None = None
) -> tuple[Vector, dict]:
    """
    Solve min_{x >= 0} ||A x - b||_2 with the Lawson-Hanson active-set algorithm.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Coefficient matrix.
    b : array_like of shape (m,)
        Right-hand side.
    tol : float, optional
        Tolerance on the dual variable ``A.T @ (b - A @ x)`` used to decide
        whether an inactive variable should enter the passive (free) set.
    max_iter : int, optional
        Maximum number of outer (active-set) iterations. Defaults to
        ``30 * n``.

    Returns
    -------
    x : ndarray of shape (n,)
        The non-negative least-squares solution.
    info : dict
        Diagnostics with keys:

        - ``outer_iterations`` : int
            Number of active-set updates performed.
        - ``inner_iterations`` : int
            Total number of unconstrained least-squares solves across all
            outer iterations.
        - ``passive_set`` : ndarray
            Indices of the variables in the final passive (free) set.
        - ``residual_norm`` : float
            ``norm(A @ x - b, 2)`` at the returned ``x``.

    Raises
    ------
    ValueError
        If ``A`` is not two-dimensional, ``b`` is not one-dimensional, or
        their shapes are incompatible.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible shapes between A and b.")

    m, n = A.shape
    if max_iter is None:
        max_iter = 30 * n

    P = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=np.float64)

    outer_iter = 0
    inner_iter_total = 0

    while outer_iter < max_iter:
        outer_iter += 1

        residual = b - A @ x
        w = A.T @ residual

        candidates = np.where(~P)[0]
        if candidates.size == 0:
            break

        t = candidates[np.argmax(w[candidates])]
        if w[t] <= tol:
            break

        P[t] = True

        while True:
            inner_iter_total += 1
            p_idx = np.where(P)[0]
            z = np.zeros(n, dtype=np.float64)

            A_P = A[:, p_idx]
            z_P, *_ = np.linalg.lstsq(A_P, b, rcond=None)
            z[p_idx] = z_P

            if np.all(z[p_idx] > -tol):
                z[p_idx] = np.maximum(z[p_idx], 0.0)
                x = z
                break

            negative = (z < 0) & P
            alpha = np.min(x[negative] / (x[negative] - z[negative]))

            x = x + alpha * (z - x)

            hit_zero = P & (x <= tol)
            x[hit_zero] = 0.0
            P[hit_zero] = False

    info = {
        "outer_iterations": outer_iter,
        "inner_iterations": inner_iter_total,
        "passive_set": np.where(P)[0],
        "residual_norm": float(np.linalg.norm(A @ x - b)),
    }
    return x, info


def _cholesky_factor(M: Matrix) -> Matrix:
    """
    Compute the lower Cholesky factor of a symmetric positive-definite matrix.

    Parameters
    ----------
    M : ndarray of shape (n, n)
        A symmetric matrix.

    Returns
    -------
    ndarray of shape (n, n)
        The lower-triangular factor ``L`` such that ``M = L @ L.T``.

    Raises
    ------
    UnsupportedMatrixError
        If ``M`` is not positive definite, i.e. the Cholesky factorization
        fails.
    """
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        raise UnsupportedMatrixError(
            "NNLS reduction requires a symmetric positive-definite M "
            "(Cholesky factorization failed)."
        ) from exc


def nnls(
    problem: LCP,
    *,
    tol: float = 1e-10,
    max_iter: int | None = None,
    check_symmetric: bool = True,
) -> LCPResult:
    """
    Solve an LCP(M, q) by reduction to a non-negative least squares
    problem, valid whenever M is symmetric positive definite::

        LCP(M, q)  <=>  min_{z >= 0} 0.5 z.T M z + q.T z
                   <=>  min_{z >= 0} ||A z - b||_2

    with M = L L.T (Cholesky), A = L.T and b = -L^{-1} q, solved with the
    Lawson-Hanson active-set algorithm (:func:`nnls_lawson_hanson`).

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem ``LCP(M, q)`` to solve. ``M``
        must be symmetric positive definite.
    tol : float, optional
        Convergence tolerance on ``norm(min(z, w), inf)``.
    max_iter : int, optional
        Maximum number of outer Lawson-Hanson iterations. Defaults to
        ``30 * n``; see :func:`nnls_lawson_hanson`.
    check_symmetric : bool, optional
        If True (default), verify that ``M`` is symmetric before computing
        its Cholesky factorization.

    Returns
    -------
    LCPResult
        The solver outcome. ``status`` is
        :attr:`LCPStatus.CONVERGED` if ``residual < tol``, otherwise
        :attr:`LCPStatus.MAX_ITERATIONS`.

    Raises
    ------
    InvalidSolverOptionError
        If ``tol <= 0`` or ``max_iter <= 0``.
    UnsupportedMatrixError
        If ``M`` is not symmetric (when ``check_symmetric`` is True), or if
        the Cholesky factorization of ``M`` fails (``M`` is not positive
        definite).

    See Also
    --------
    nnls_lawson_hanson : The underlying NNLS solver.
    """
    if tol <= 0.0:
        raise InvalidSolverOptionError(f"tol must be > 0, got {tol}.")
    if max_iter is not None and max_iter <= 0:
        raise InvalidSolverOptionError(f"max_iter must be > 0, got {max_iter}.")

    M, q = problem.M, problem.q

    if check_symmetric and not np.allclose(M, M.T, atol=1e-12, rtol=1e-12):
        raise UnsupportedMatrixError(
            "NNLS reduction requires a symmetric M; the LCP(M, q) <-> "
            "min_{z>=0} 0.5 z.T M z + q.T z equivalence does not hold otherwise."
        )

    L = _cholesky_factor(M)
    A = L.T
    b = np.linalg.solve(L, -q)

    z, solver_info = nnls_lawson_hanson(A, b, tol=tol, max_iter=max_iter)

    w = M @ z + q
    residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf))
    status = LCPStatus.CONVERGED if residual < tol else LCPStatus.MAX_ITERATIONS

    return LCPResult(
        z=z,
        w=w,
        status=status,
        iterations=solver_info["outer_iterations"],
        residual=residual,
        message=f"NNLS {status.value} (residual={residual:.3e}).",
    )

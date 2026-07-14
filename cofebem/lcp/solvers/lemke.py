from __future__ import annotations

import numpy as np

from ..exceptions import InvalidSolverOptionError
from ..problem import LCP, Matrix, Vector
from ..result import LCPResult, LCPStatus

__all__ = ["lemke"]

# Tags identifying which variable occupies a tableau column/row:
# basic/non-basic "w" or "z" components, the artificial driver "y" (z0),
# or the constant column "q".
_W, _Z, _Y, _Q = range(4)


class _LemkeTableau:
    """
    Pivoting tableau for Lemke's algorithm applied to LCP(M, q)::

        w = M z + q,  z >= 0,  w >= 0,  z.T w = 0.

    The tableau is augmented with an artificial variable z0, driven into the
    basis to restore feasibility, and with the constant column q::

        T = [ I | -M | -1 | q ]

    Complementary pivoting then drives z0 back out of the basis. The walk
    terminates either with z0 leaving the basis (a solution to the LCP), or
    with no eligible pivot row for the entering variable (a secondary ray,
    indicating the problem has no solution reachable from this starting
    basis).

    Parameters
    ----------
    M : ndarray of shape (n, n)
        The LCP matrix.
    q : ndarray of shape (n,)
        The LCP vector.

    Attributes
    ----------
    n : int
        The problem size.
    iterations : int
        The number of pivots performed so far.
    T : ndarray of shape (n, 2 * n + 2)
        The tableau, initialized to ``[I | -M | -1 | q]``.
    index : ndarray of shape (2, 2 * n + 2)
        For each tableau column, a ``(kind, index)`` pair identifying the
        variable currently occupying it (``_W``, ``_Z``, ``_Y`` or ``_Q``).
    w_pos, z_pos : ndarray of shape (n,)
        Current column index of each ``w_i`` / ``z_i`` variable.
    """

    def __init__(self, M: Matrix, q: Vector) -> None:
        n = q.size
        self.n = n
        self.iterations = 0

        self.T = np.hstack((np.eye(n), -M, -np.ones((n, 1)), q.reshape(n, 1)))

        self.w_pos = np.arange(n)
        self.z_pos = np.arange(n, 2 * n)

        basic = np.vstack((_W * np.ones(n, dtype=int), np.arange(n, dtype=int)))
        nonbasic = np.vstack((_Z * np.ones(n, dtype=int), np.arange(n, dtype=int)))
        driver = np.array([[_Y], [0]])
        rhs = np.array([[_Q], [0]])
        self.index = np.hstack((basic, nonbasic, driver, rhs))

    def solve(self, max_iter: int) -> tuple[Vector | None, int, str]:
        """
        Run Lemke's complementary pivoting algorithm.

        Parameters
        ----------
        max_iter : int
            Maximum number of complementary pivots after the initial
            (driving) pivot.

        Returns
        -------
        z : ndarray of shape (n,) or None
            The LCP solution, or ``None`` if no solution was found.
        exit_code : int
            ``0`` if a solution was found, ``1`` on a secondary ray, ``2``
            if ``max_iter`` was exceeded.
        detail : str
            Human-readable description of the outcome.
        """
        if not self._initialize():
            return np.zeros(self.n), 0, "trivial solution (q >= 0)"

        for _ in range(max_iter):
            stepped = self._step()
            if self.index[0, -2] == _Y:
                return (
                    self._extract_solution(),
                    0,
                    f"solution found after {self.iterations} pivot(s)",
                )
            if not stepped:
                return (
                    None,
                    1,
                    f"secondary ray found after {self.iterations} pivot(s)",
                )

        return None, 2, f"maximum iterations ({max_iter}) exceeded"

    def _initialize(self) -> bool:
        q = self.T[:, -1]
        row = np.argmin(q)
        if q[row] < 0.0:
            self._clear_driver_column(row)
            self._pivot(row)
            return True
        return False

    def _step(self) -> bool:
        q = self.T[:, -1]
        a = self.T[:, -2]

        eligible = np.where(a > 0.0)[0]
        if eligible.size == 0:
            return False

        row = eligible[np.argmin(q[eligible] / a[eligible])]
        self._clear_driver_column(row)
        self._pivot(row)
        return True

    def _extract_solution(self) -> Vector:
        z = np.zeros(self.n, dtype=np.float64)
        kind, var = self.index[:, : self.n]
        q = self.T[:, -1]
        is_z = kind == _Z
        z[var[is_z]] = q[is_z]
        return z

    def _partner_column(self, col: int) -> int | None:
        kind, ind = self.index[:, col]
        if kind == _W:
            return self.z_pos[ind]
        if kind == _Z:
            return self.w_pos[ind]
        return None

    def _pivot(self, row: int) -> None:
        partner = self._partner_column(row)
        if partner is not None:
            self._swap_columns(row, partner)
        self._swap_columns(row, -2)
        self.iterations += 1

    def _swap_columns(self, i: int, j: int) -> None:
        (kind_i, ind_i), (kind_j, ind_j) = self.index[:, i], self.index[:, j]
        self._set_position(kind_i, ind_i, j)
        self._set_position(kind_j, ind_j, i)

        self.index[:, [i, j]] = self.index[:, [j, i]]
        self.T[:, [i, j]] = self.T[:, [j, i]]

    def _set_position(self, kind: int, ind: int, new_col: int) -> None:
        new_col %= 2 * self.n + 2
        if kind == _W:
            self.w_pos[ind] = new_col
        elif kind == _Z:
            self.z_pos[ind] = new_col

    def _clear_driver_column(self, row: int) -> None:
        self.T[row] /= self.T[row, -2]
        multipliers = self.T[:, -2].copy()
        multipliers[row] = 0.0
        self.T -= np.outer(multipliers, self.T[row])


def lemke(problem: LCP, *, max_iter: int = 100, tol: float = 1e-10) -> LCPResult:
    """
    Solve an LCP(M, q) with Lemke's complementary pivoting algorithm.

    Lemke's method augments the tableau ``[I | -M | -1 | q]`` with an
    artificial variable z0 and performs complementary pivots until z0 leaves
    the basis (an exact solution of the LCP), no eligible pivot row remains
    for the entering variable (a secondary ray, reported as
    :attr:`LCPStatus.RAY_TERMINATION`), or ``max_iter`` pivots are exceeded.

    Unlike the iterative solvers in this package, a "solution found" outcome
    is, up to floating-point error, an exact vertex of the LCP -- ``residual``
    is reported purely as a diagnostic of that floating-point error.

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem ``LCP(M, q)`` to solve. ``M``
        may be non-symmetric and indefinite; Lemke's algorithm is not
        guaranteed to find a solution for arbitrary ``M``, but will report
        :attr:`LCPStatus.RAY_TERMINATION` if it cannot.
    max_iter : int, optional
        Maximum number of complementary pivots (after the initial driving
        pivot).
    tol : float, optional
        Tolerance used to classify a "solution found" outcome as
        :attr:`LCPStatus.CONVERGED` (``residual < tol``) versus
        :attr:`LCPStatus.NUMERICAL_BREAKDOWN`.

    Returns
    -------
    LCPResult
        The solver outcome. ``status`` is one of
        :attr:`LCPStatus.CONVERGED`, :attr:`LCPStatus.NUMERICAL_BREAKDOWN`,
        :attr:`LCPStatus.RAY_TERMINATION`, or
        :attr:`LCPStatus.MAX_ITERATIONS`. When no solution is found
        (``RAY_TERMINATION`` or ``MAX_ITERATIONS``), ``z`` is the zero
        vector and ``residual`` is computed from it.

    Raises
    ------
    InvalidSolverOptionError
        If ``max_iter <= 0`` or ``tol <= 0``.

    References
    ----------
    Lemke, C. E. (1965). "Bimatrix Equilibrium Points and Mathematical
    Programming". Management Science, 11(7), 681-689.
    """
    if max_iter <= 0:
        raise InvalidSolverOptionError(f"max_iter must be > 0, got {max_iter}.")
    if tol <= 0.0:
        raise InvalidSolverOptionError(f"tol must be > 0, got {tol}.")

    M, q = problem.M, problem.q
    n = problem.size

    tableau = _LemkeTableau(M, q)
    z, exit_code, detail = tableau.solve(max_iter)

    if z is None:
        z = np.zeros(n, dtype=np.float64)

    w = M @ z + q
    residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf))

    if exit_code == 0:
        status = (
            LCPStatus.CONVERGED if residual < tol else LCPStatus.NUMERICAL_BREAKDOWN
        )
    elif exit_code == 1:
        status = LCPStatus.RAY_TERMINATION
    else:
        status = LCPStatus.MAX_ITERATIONS

    return LCPResult(
        z=z,
        w=w,
        status=status,
        iterations=tableau.iterations,
        residual=residual,
        message=f"Lemke's algorithm: {detail} (residual={residual:.3e}).",
    )

from __future__ import annotations

import numpy as np

from ..exceptions import InvalidSolverOptionError, UnsupportedMatrixError
from ..problem import LCP, Vector
from ..result import LCPResult, LCPStatus

__all__ = ["ccg", "ccg_v2"]


def _residual(z: Vector, w: Vector) -> float:
    """float: The complementarity residual ``norm(min(z, w), inf)``."""
    return float(np.linalg.norm(np.minimum(z, w), ord=np.inf))


def _check_symmetric(M: object, check_symmetric: bool, name: str) -> None:
    """
    Raise :class:`UnsupportedMatrixError` if ``M`` is not symmetric.

    Parameters
    ----------
    M : ndarray of shape (n, n)
        The matrix to check.
    check_symmetric : bool
        If False, the check is skipped.
    name : str
        Name of the calling solver, used in the error message.

    Raises
    ------
    UnsupportedMatrixError
        If ``check_symmetric`` is True and ``M`` is not (numerically)
        symmetric.
    """
    if not check_symmetric:
        return
    if isinstance(M, np.ndarray):
        symmetric = np.allclose(M, M.T, atol=1e-12, rtol=1e-12)
    else:
        # Exhaustive verification would materialise the operator. Require an
        # explicit guarantee from the implementation instead.
        symmetric = getattr(M, "symmetric", False) is True
    if not symmetric:
        raise UnsupportedMatrixError(
            f"{name} requires a symmetric M; the LCP(M, q) <-> "
            "min_{z>=0} 0.5 z.T M z + q.T z equivalence does not hold otherwise."
        )


def _initial_iterate(q: Vector, n: int, z0: Vector | None, default: Vector) -> Vector:
    """
    Build a feasible (``z >= 0``) initial iterate.

    Parameters
    ----------
    q : ndarray of shape (n,)
        The LCP vector (unused other than for context; kept for a uniform
        signature).
    n : int
        The problem size.
    z0 : array_like of shape (n,) or None
        User-supplied initial guess, or ``None`` to use ``default``.
    default : ndarray of shape (n,)
        The initial iterate to use when ``z0 is None``.

    Returns
    -------
    ndarray of shape (n,)
        ``z0`` projected onto ``z >= 0``, or ``default`` if ``z0 is None``.

    Raises
    ------
    InvalidSolverOptionError
        If ``z0`` is not ``None`` and does not have shape ``(n,)``.
    """
    if z0 is None:
        return default
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    if z0.shape != (n,):
        raise InvalidSolverOptionError(f"z0 must have shape ({n},), got {z0.shape}.")
    return np.maximum(z0, 0.0)


def ccg(
    problem: LCP,
    *,
    tol: float = 1e-10,
    max_iter: int = 10000,
    z0: Vector | None = None,
    pressure_factor: float = 1e12,
    err_type: str = "displacement",
    record_history: bool = False,
    check_symmetric: bool = True,
) -> LCPResult:
    """
    Solve an LCP(M, q) with the Constrained Conjugate Gradient (CCG)
    method of Polonsky & Keer (1999), valid whenever M is symmetric positive
    definite.

    This is the active-set CG scheme commonly used for normal-contact
    pressure problems, where M is a compliance operator and q the (negated)
    initial gap, related through the LCP(M, q) convention ``w = M @ z + q``::

        z >= 0, w >= 0, z.T w = 0.

    On each iteration, the search direction ``t`` is the steepest-descent
    direction ``w`` restricted to the active set ``z > 0``, conjugated with
    the previous direction (Polak-Ribiere-style) unless the previous step
    activated a penetrating, zero-pressure point -- in which case the CG
    recursion restarts with a pure steepest-descent step. After the
    line-search step ``z <- max(z - tau * t, 0)``, any point that is both at
    zero pressure and still penetrating (``w < 0``) is corrected directly.

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem ``LCP(M, q)`` to solve. ``M``
        must be symmetric positive definite.
    tol : float, optional
        Convergence tolerance, interpreted according to ``err_type`` (and
        also checked against the standard complementarity residual; see
        Notes).
    max_iter : int, optional
        Maximum number of CG iterations.
    z0 : array_like of shape (n,), optional
        Initial guess. Negative entries are projected to zero. If not
        given, defaults to ``max(-q, 0) * pressure_factor``.
    pressure_factor : float, optional
        Scale factor used to build the default initial guess from the
        penetrating part of ``-q``. Must be positive.
    err_type : {"displacement", "mix", "nw"}, optional
        Selects the convergence test (see Notes).
    record_history : bool, optional
        If True, record the complementarity residual after every iteration
        in ``result.residual_history``.
    check_symmetric : bool, optional
        If True (default), verify that ``M`` is symmetric.

    Returns
    -------
    LCPResult
        The solver outcome. ``status`` is one of
        :attr:`LCPStatus.CONVERGED`, :attr:`LCPStatus.NUMERICAL_BREAKDOWN`,
        or :attr:`LCPStatus.MAX_ITERATIONS`.

    Raises
    ------
    InvalidSolverOptionError
        If ``tol <= 0``, ``max_iter <= 0``, ``pressure_factor <= 0``,
        ``err_type`` is not one of ``"displacement"``, ``"mix"`` or
        ``"nw"``, or ``z0`` does not have shape ``(n,)``.
    UnsupportedMatrixError
        If ``M`` is not symmetric (when ``check_symmetric`` is True).

    Notes
    -----
    ``err_type`` selects the convergence test:

    - ``"displacement"``: ``norm(w[z > 0]) / norm(w) < tol``
      (the complementary gap ``w`` should vanish on the active set);
    - ``"mix"``: ``sqrt(displacement_error * |w.z| / norm(w)) < tol``;
    - ``"nw"``: relative change in ``norm(w)`` between iterations < tol.

    Regardless of ``err_type``, the standard complementarity residual
    ``norm(min(z, w), inf)`` is also checked each iteration and is what is
    reported as ``result.residual``.

    References
    ----------
    J. J. Moré and G. Toraldo, “On the Solution of Large Quadratic Programming Problems with Bound Constraints,” SIAM Journal on Optimization, 1(1), pp. 93–113, 1991. DOI: 10.1137/0801008.
    M. Paggi, A. Bemporad, and related contributors, “Computational Methods for Contact Problems with Roughness,” in CISM International Centre for Mechanical Sciences, 2020.
    
    See Also
    --------
    ccg_v2 : A more robust active-set CG method with explicit active-set
        management and conjugacy restarts.
    """
    if tol <= 0.0:
        raise InvalidSolverOptionError(f"tol must be > 0, got {tol}.")
    if max_iter <= 0:
        raise InvalidSolverOptionError(f"max_iter must be > 0, got {max_iter}.")
    if pressure_factor <= 0.0:
        raise InvalidSolverOptionError(
            f"pressure_factor must be > 0, got {pressure_factor}."
        )
    if err_type not in ("displacement", "mix", "nw"):
        raise InvalidSolverOptionError(
            f"err_type must be 'displacement', 'mix' or 'nw', got {err_type!r}."
        )

    M, q = problem.M, problem.q
    n = problem.size

    _check_symmetric(M, check_symmetric, "CCG")

    z = _initial_iterate(q, n, z0, default=np.maximum(-q, 0.0) * pressure_factor)

    w = M @ z + q
    residual = _residual(z, w)

    history = [residual] if record_history else None
    status = LCPStatus.MAX_ITERATIONS
    iterations = 0

    t = w.copy()
    t_prev = np.zeros_like(w)
    restart_cg = True
    error = 1.0
    error_prev = 1.0

    if residual < tol:
        status = LCPStatus.CONVERGED
    else:
        for k in range(max_iter):
            iterations = k + 1

            if k > 0:
                active = z > 0.0
                beta = error / error_prev if error_prev != 0.0 else 0.0
                t = np.zeros_like(w)
                t[active] = w[active] + (beta if restart_cg else 0.0) * t_prev[active]

            if float(t @ t) == 0.0:
                status = LCPStatus.NUMERICAL_BREAKDOWN
                break

            curvature = float(t @ (M @ t))
            if curvature <= 0.0:
                status = LCPStatus.NUMERICAL_BREAKDOWN
                break

            tau = float(w @ t) / curvature
            z = np.maximum(z - tau * t, 0.0)

            zero_pressure = z == 0.0
            penetration = w < 0.0
            set_I = zero_pressure & penetration
            restart_cg = not np.any(set_I)
            if not restart_cg:
                z[set_I] -= tau * w[set_I]

            t_prev = t
            w = M @ z + q
            residual = _residual(z, w)
            if record_history:
                history.append(residual)

            nw = float(np.linalg.norm(w, ord=2))
            if nw > 0.0:
                active = z > 0.0
                displ_error = float(np.linalg.norm(w[active], ord=2)) / nw
                ort = abs(float(w @ z)) / nw
            else:
                displ_error = 0.0
                ort = 0.0

            error_prev = error
            if err_type == "displacement":
                error = displ_error
                converged = error < tol
            elif err_type == "mix":
                error = float(np.sqrt(displ_error * ort))
                converged = error < tol
            else:  # "nw"
                error = nw
                converged = (
                    error_prev != 0.0 and abs((error - error_prev) / error_prev) < tol
                )

            if converged or residual < tol:
                status = LCPStatus.CONVERGED
                break

    return LCPResult(
        z=z,
        w=w,
        status=status,
        iterations=iterations,
        residual=residual,
        residual_history=np.asarray(history) if record_history else None,
        message=f"CCG {status.value} after {iterations} iteration(s) "
        f"(residual={residual:.3e}).",
    )


def ccg_v2(
    problem: LCP,
    *,
    tol: float = 1e-10,
    max_iter: int = 10000,
    z0: Vector | None = None,
    record_history: bool = False,
    check_symmetric: bool = True,
) -> LCPResult:
    """
    Solve an LCP(M, q) with an active-set constrained
    conjugate-gradient method, valid whenever M is symmetric positive
    definite.

    The method alternates between:

    - a linear CG minimization of ``phi(z) = 0.5 z.T M z + q.T z`` on the
      current "free" face ``{i : not active[i]}``, where ``active[i]``
      means ``z_i`` is fixed at its lower bound 0;
    - active-set updates: a free variable that hits zero during a CG step
      is fixed (activated), and an active variable with a negative
      multiplier ``w_i < 0`` (which would decrease ``phi`` if freed) is
      released, restarting CG on the new face.

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem ``LCP(M, q)`` to solve. ``M``
        must be symmetric positive definite.
    tol : float, optional
        Convergence tolerance on the relative natural residual (see Notes).
    max_iter : int, optional
        Maximum total number of CG steps and active-set changes.
    z0 : array_like of shape (n,), optional
        Initial guess. Negative entries are projected to zero. Defaults to
        the zero vector.
    record_history : bool, optional
        If True, record the relative natural residual after every CG step
        in ``result.residual_history``.
    check_symmetric : bool, optional
        If True (default), verify that ``M`` is symmetric.

    Returns
    -------
    LCPResult
        The solver outcome. ``status`` is one of
        :attr:`LCPStatus.CONVERGED`, :attr:`LCPStatus.NUMERICAL_BREAKDOWN`,
        or :attr:`LCPStatus.MAX_ITERATIONS`. ``iterations`` counts CG steps
        only; the number of active-set changes is reported in ``message``.
        ``residual`` is the relative natural residual (see Notes), not the
        raw ``norm(min(z, w), inf)``.

    Raises
    ------
    InvalidSolverOptionError
        If ``tol <= 0``, ``max_iter <= 0``, or ``z0`` does not have shape
        ``(n,)``.
    UnsupportedMatrixError
        If ``M`` is not symmetric (when ``check_symmetric`` is True).

    Notes
    -----
    Convergence is measured by the relative natural residual::

        norm(min(z, w), inf) / (1 + norm(z, inf) + norm(w, inf)) < tol.

    See Also
    --------
    ccg : The classic Polonsky-Keer CCG iteration.
    """
    if tol <= 0.0:
        raise InvalidSolverOptionError(f"tol must be > 0, got {tol}.")
    if max_iter <= 0:
        raise InvalidSolverOptionError(f"max_iter must be > 0, got {max_iter}.")

    M, q = problem.M, problem.q
    n = problem.size
    eps = np.finfo(np.float64).eps

    _check_symmetric(M, check_symmetric, "CCG (v2)")

    z = _initial_iterate(q, n, z0, default=np.zeros(n, dtype=np.float64))
    w = M @ z + q

    history: list[float] = []

    # active[i] = True means z_i is fixed at its lower bound 0. A variable at
    # zero is active only when w_i >= 0; if w_i < 0, increasing z_i decreases
    # the objective, so that variable must stay free.
    zero_tol = 100.0 * eps * max(1.0, float(np.linalg.norm(z, ord=np.inf)))
    active = (z <= zero_tol) & (w >= 0.0)
    z[active] = 0.0

    cg_iterations = 0
    active_set_changes = 0
    work_count = 0
    status = LCPStatus.MAX_ITERATIONS

    while work_count < max_iter:
        w = M @ z + q
        scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
        residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf)) / scale

        if residual <= tol:
            status = LCPStatus.CONVERGED
            break

        free = ~active

        # CG residual on the free face: grad(phi) = M z + q = w, so r = -w
        # restricted to the free variables.
        r = np.zeros_like(z)
        r[free] = -w[free]

        if np.linalg.norm(r, ord=np.inf) <= tol * scale:
            # The free face is minimized; release the active variable with
            # the most negative multiplier, since w_j < 0 implies increasing
            # z_j decreases phi.
            violated = np.flatnonzero(active & (w < -tol * scale))
            if violated.size == 0:
                status = LCPStatus.CONVERGED
                break

            j = violated[np.argmin(w[violated])]
            active[j] = False
            active_set_changes += 1
            work_count += 1
            continue

        d = r.copy()
        rr = float(r @ r)

        # Inner CG loop on the current fixed face.
        while work_count < max_iter:
            Md = M @ d
            curvature = float(d @ Md)
            if not np.isfinite(curvature) or curvature <= 0.0:
                status = LCPStatus.NUMERICAL_BREAKDOWN
                break

            # alpha_CG = -(w.d) / (d.M.d) = (r.r) / (d.M.d) for linear CG.
            alpha_cg = rr / curvature

            # Largest step keeping z + alpha*d >= 0.
            decreasing = d < 0.0
            if np.any(decreasing):
                alpha_max = float(np.min(-z[decreasing] / d[decreasing]))
            else:
                alpha_max = np.inf

            alpha = min(alpha_cg, alpha_max)
            if alpha < 0.0 or not np.isfinite(alpha):
                status = LCPStatus.NUMERICAL_BREAKDOWN
                break

            z = z + alpha * d

            zero_tol = 1000.0 * eps * max(1.0, float(np.linalg.norm(z, ord=np.inf)))
            tiny_negative = (z < 0.0) & (z >= -zero_tol)
            z[tiny_negative] = 0.0

            if np.any(z < -zero_tol):
                status = LCPStatus.NUMERICAL_BREAKDOWN
                break

            w = M @ z + q
            cg_iterations += 1
            work_count += 1

            if record_history:
                scale = (
                    1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
                )
                history.append(
                    float(np.linalg.norm(np.minimum(z, w), ord=np.inf)) / scale
                )

            hit_boundary = np.isfinite(alpha_max) and alpha_max <= alpha_cg * (
                1.0 + 1e-14
            )
            if hit_boundary:
                hit = (z <= zero_tol) & (d < 0.0)
                z[hit] = 0.0
                active[hit] = True
                active_set_changes += 1
                # The feasible face changed; old directions are no longer
                # conjugate, so restart CG.
                break

            free = ~active
            r_new = np.zeros_like(z)
            r_new[free] = -w[free]

            scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
            if np.linalg.norm(r_new, ord=np.inf) <= tol * scale:
                # The current face is minimized; return to the outer loop to
                # check whether an active variable must be released.
                break

            rr_new = float(r_new @ r_new)
            beta = rr_new / rr  # Fletcher-Reeves / linear-CG coefficient.

            d = r_new + beta * d
            d[active] = 0.0

            r = r_new
            rr = rr_new

        if status == LCPStatus.NUMERICAL_BREAKDOWN:
            break

    w = M @ z + q
    scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
    residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf)) / scale

    if status == LCPStatus.MAX_ITERATIONS and residual <= tol:
        status = LCPStatus.CONVERGED

    return LCPResult(
        z=z,
        w=w,
        status=status,
        iterations=cg_iterations,
        residual=residual,
        residual_history=np.asarray(history) if record_history else None,
        message=f"CCG (v2) {status.value} after {cg_iterations} CG step(s) and "
        f"{active_set_changes} active-set change(s) (residual={residual:.3e}).",
    )

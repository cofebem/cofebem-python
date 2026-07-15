"""Projected preconditioned conjugate gradient for contact LCPs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..exceptions import InvalidSolverOptionError
from ..problem import LCP, Vector
from ..result import LCPResult, LCPStatus
from .ccg import _check_symmetric, _initial_iterate

ProjectedPreconditioner = Callable[[Vector, np.ndarray], Vector]

__all__ = ["ppcg", "ProjectedPreconditioner"]


def _natural_residual(z: Vector, w: Vector) -> float:
    scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
    return float(np.linalg.norm(np.minimum(z, w), ord=np.inf) / scale)


def _apply_preconditioner(
    projected_gradient: Vector,
    free: np.ndarray,
    preconditioner: ProjectedPreconditioner | None,
) -> Vector:
    if preconditioner is None:
        result = projected_gradient.copy()
    else:
        try:
            result = np.asarray(
                preconditioner(projected_gradient, free), dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise InvalidSolverOptionError(
                f"preconditioner application failed: {exc}"
            ) from exc
        if result.shape != projected_gradient.shape:
            raise InvalidSolverOptionError(
                "preconditioner must return a vector with shape "
                f"{projected_gradient.shape}, got {result.shape}."
            )
        if not np.all(np.isfinite(result)):
            raise InvalidSolverOptionError(
                "preconditioner returned NaN or infinite values."
            )
        result = result.copy()
    result[~free] = 0.0
    return result


def ppcg(
    problem: LCP,
    *,
    tol: float = 1.0e-10,
    max_iter: int = 10000,
    z0: Vector | None = None,
    preconditioner: ProjectedPreconditioner | None = None,
    beta_method: str = "pr_plus",
    record_history: bool = False,
    check_symmetric: bool = True,
) -> LCPResult:
    """Solve an SPD contact LCP by projected preconditioned CG.

    This is the indentation-controlled counterpart of the Polonsky--Keer
    projected PCG scheme. At each iteration it builds the projected free set
    ``(z > 0) | (w < 0)``, so every penetrating zero-pressure node can enter
    in the same update. The masked preconditioned gradient is conjugated with
    PR+ (default) or Fletcher--Reeves, followed by an exact quadratic line
    search, projection onto ``z >= 0``, and overlap correction.

    The preconditioner must implement ``preconditioner(gradient, free_mask)``
    and represent an SPD map on the masked subspace. It is never given the
    matrix explicitly, so hierarchical operator-only execution is preserved.

    Parameters
    ----------
    problem : LCP
        ``LCP(M, q)`` with symmetric positive-definite ``M``.
    tol : float
        Tolerance on the relative natural residual.
    max_iter : int
        Maximum number of projected steps.
    z0 : array_like, optional
        Feasible warm start; negative entries are projected to zero.
    preconditioner : callable, optional
        Mask-aware SPD inverse approximation.
    beta_method : {"pr_plus", "fletcher_reeves"}
        Nonlinear conjugacy update. Conjugacy restarts whenever the projected
        free set changes.
    record_history : bool
        Record the relative natural residual at every outer evaluation.
    check_symmetric : bool
        Require a symmetric dense matrix or an operator declaring
        ``symmetric=True``.
    """
    if tol <= 0.0:
        raise InvalidSolverOptionError(f"tol must be > 0, got {tol}.")
    if max_iter <= 0:
        raise InvalidSolverOptionError(f"max_iter must be > 0, got {max_iter}.")
    if beta_method not in {"pr_plus", "fletcher_reeves"}:
        raise InvalidSolverOptionError(
            "beta_method must be 'pr_plus' or 'fletcher_reeves', "
            f"got {beta_method!r}."
        )
    if preconditioner is not None and not callable(preconditioner):
        raise InvalidSolverOptionError("preconditioner must be callable or None.")

    M, q = problem.M, problem.q
    n = problem.size
    _check_symmetric(M, check_symmetric, "PPCG")

    z = _initial_iterate(q, n, z0, default=np.zeros(n, dtype=np.float64))
    direction = np.zeros(n, dtype=np.float64)
    previous_gradient = np.zeros(n, dtype=np.float64)
    previous_free: np.ndarray | None = None
    previous_inner_product = 1.0
    force_restart = True
    eps = np.finfo(np.float64).eps
    history: list[float] = []
    status = LCPStatus.MAX_ITERATIONS
    iterations = 0
    operator_applications = 0

    for _ in range(max_iter):
        w = M @ z + q
        operator_applications += 1
        residual = _natural_residual(z, w)
        if record_history:
            history.append(residual)
        if residual <= tol:
            status = LCPStatus.CONVERGED
            break

        zero_tolerance = 100.0 * eps * max(
            1.0, float(np.linalg.norm(z, ord=np.inf))
        )
        free = (z > zero_tolerance) | (w < 0.0)
        projected_gradient = np.where(free, w, 0.0)
        preconditioned = _apply_preconditioner(
            projected_gradient, free, preconditioner
        )
        inner_product = float(projected_gradient @ preconditioned)
        if not np.isfinite(inner_product) or inner_product <= 0.0:
            status = LCPStatus.NUMERICAL_BREAKDOWN
            break

        free_set_changed = previous_free is None or not np.array_equal(
            free, previous_free
        )
        if free_set_changed or force_restart:
            beta = 0.0
        elif beta_method == "pr_plus":
            beta = max(
                0.0,
                float(preconditioned @ (projected_gradient - previous_gradient))
                / previous_inner_product,
            )
        else:
            beta = inner_product / previous_inner_product

        direction = np.where(free, preconditioned + beta * direction, 0.0)
        numerator = float(projected_gradient @ direction)
        if not np.isfinite(numerator) or numerator <= 0.0:
            # A stale nonlinear-CG direction lost descent: restart with the
            # preconditioned projected gradient.
            direction = preconditioned.copy()
            numerator = inner_product

        matrix_direction = M @ direction
        operator_applications += 1
        curvature = float(direction @ matrix_direction)
        if not np.isfinite(curvature) or curvature <= 0.0:
            status = LCPStatus.NUMERICAL_BREAKDOWN
            break

        step = numerator / curvature
        unprojected = z - step * direction
        z_new = np.maximum(unprojected, 0.0)
        zero_tolerance = 1000.0 * eps * max(
            1.0, float(np.linalg.norm(z_new, ord=np.inf))
        )
        z_new[z_new <= zero_tolerance] = 0.0

        # If projection leaves a currently penetrating point at zero, give it
        # the reference method's local steepest-descent overlap correction.
        overlap = (z_new == 0.0) & (w < 0.0)
        z_new[overlap] = -step * w[overlap]
        force_restart = bool(np.any(overlap))

        previous_gradient = projected_gradient
        previous_inner_product = inner_product
        previous_free = free
        z = z_new
        iterations += 1

    else:
        w = M @ z + q
        operator_applications += 1
        residual = _natural_residual(z, w)
        if residual <= tol:
            status = LCPStatus.CONVERGED

    return LCPResult(
        z=z,
        w=w,
        status=status,
        iterations=iterations,
        residual=residual,
        residual_history=np.asarray(history) if record_history else None,
        message=f"PPCG {status.value} after {iterations} projected step(s) and "
        f"{operator_applications} operator application(s) "
        f"(residual={residual:.3e}).",
    )

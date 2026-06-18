from __future__ import annotations

from typing import Callable

from .exceptions import UnsupportedSolverError
from .problem import LCP
from .result import LCPResult
from .solvers import ccg, ccg_v2, lemke, nnls, pgs, psor

__all__ = ["solve", "SOLVERS", "DEFAULT_METHOD"]

SOLVERS: dict[str, Callable[..., LCPResult]] = {
    "psor": psor,
    "pgs": pgs,
    "nnls": nnls,
    "lemke": lemke,
    "ccg": ccg,
    "ccg_v2": ccg_v2,
}

DEFAULT_METHOD = "lemke"


def solve(problem: LCP, method: str = DEFAULT_METHOD, **options: object) -> LCPResult:
    """
    Solve a linear complementarity problem ``LCP(M, q)``.

    This is a thin dispatcher over :mod:`cofebem.lcp.solvers`: it looks up
    ``method`` in :data:`SOLVERS` and calls the corresponding solver with
    ``problem`` and any extra keyword arguments.

    Parameters
    ----------
    problem : LCP
        The linear complementarity problem to solve.
    method : str, optional
        Name of the solver to use. One of ``"psor"``, ``"pgs"``,
        ``"nnls"``, ``"lemke"``, ``"ccg"``, or ``"ccg_v2"``. Defaults to
        ``"lemke"``, which is exact (up to floating-point error) and places
        no restriction on ``M``.
    **options
        Extra keyword arguments forwarded to the chosen solver, e.g.
        ``tol``, ``max_iter``, ``z0``. See the individual solver's
        docstring in :mod:`cofebem.lcp.solvers` for the supported options.

    Returns
    -------
    LCPResult
        The solver outcome.

    Raises
    ------
    UnsupportedSolverError
        If ``method`` is not one of the available solvers.
    InvalidSolverOptionError
        If an option is rejected by the chosen solver.
    UnsupportedMatrixError
        If ``problem.M`` is not compatible with the chosen solver.

    See Also
    --------
    cofebem.lcp.solvers.psor, cofebem.lcp.solvers.pgs,
    cofebem.lcp.solvers.nnls, cofebem.lcp.solvers.lemke,
    cofebem.lcp.solvers.ccg, cofebem.lcp.solvers.ccg_v2

    Examples
    --------
    >>> import numpy as np
    >>> problem = LCP(M=np.array([[2.0, 1.0], [1.0, 2.0]]), q=np.array([-1.0, -1.0]))
    >>> result = solve(problem)
    >>> result.converged
    True
    """
    try:
        solver = SOLVERS[method]
    except KeyError as exc:
        available = ", ".join(sorted(SOLVERS))
        raise UnsupportedSolverError(
            f"Unknown LCP solver {method!r}. Available solvers: {available}."
        ) from exc

    return solver(problem, **options)

"""
Linear complementarity problem (LCP) definitions, results and solvers.

This subpackage provides the shared vocabulary used by every LCP solver in
:mod:`cofebem.lcp.solvers`:

- :class:`LCP` -- a validated, immutable container for ``(M, q)``.
- :class:`LCPResult` / :class:`LCPStatus` -- the outcome of a solver call.
- :func:`solve` -- a frontend that dispatches ``LCP(M, q)`` to one of the
  solvers in :mod:`cofebem.lcp.solvers` by name.
- The exception hierarchy in :mod:`cofebem.lcp.exceptions`, rooted at
  :class:`LCPError`.

See :mod:`cofebem.lcp.solvers` for the available solvers (``psor``, ``pgs``,
``nnls``, ``lemke``, ``ccg``, ``ccg_v2``, ``ppcg``).
"""

from __future__ import annotations

from .exceptions import (
    InvalidLCPError,
    InvalidSolverOptionError,
    LCPError,
    LCPNumericalError,
    UnsupportedMatrixError,
    UnsupportedSolverError,
)
from .problem import LCP
from .preconditioners import (
    RestrictedProjectedPreconditioner,
    SectorSurfaceSpectralPreconditioner,
)
from .result import LCPResult, LCPStatus
from .solve import DEFAULT_METHOD, SOLVERS, solve

__all__ = [
    "LCP",
    "SectorSurfaceSpectralPreconditioner",
    "RestrictedProjectedPreconditioner",
    "LCPResult",
    "LCPStatus",
    "solve",
    "SOLVERS",
    "DEFAULT_METHOD",
    "LCPError",
    "InvalidLCPError",
    "InvalidSolverOptionError",
    "UnsupportedSolverError",
    "UnsupportedMatrixError",
    "LCPNumericalError",
]

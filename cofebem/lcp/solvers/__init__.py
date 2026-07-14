"""
Solvers for :class:`cofebem.lcp.problem.LCP`.

Each solver takes an :class:`~cofebem.lcp.problem.LCP` plus
keyword-only options and returns an
:class:`~cofebem.lcp.result.LCPResult`. Available solvers:

- :func:`psor` / :func:`pgs` -- Projected (Successive-Over-Relaxation /
  Gauss-Seidel) iteration. Simple, matrix-free-friendly, linear convergence.
- :func:`nnls` -- exact reduction to a non-negative least-squares problem
  via Cholesky factorization, for symmetric positive-definite ``M``.
- :func:`lemke` -- Lemke's complementary pivoting algorithm, an exact
  (up to floating-point error) direct method for general ``M``.
- :func:`ccg` / :func:`ccg_v2` -- constrained conjugate-gradient methods for
  symmetric positive-definite ``M``, as commonly used for normal-contact
  pressure problems.
- :func:`ppcg` -- projected preconditioned CG with simultaneous projected
  active-set updates.
"""

from __future__ import annotations

from .ccg import ccg, ccg_v2
from .lemke import lemke
from .nnls import nnls
from .psor import pgs, psor
from .ppcg import ppcg

__all__ = ["psor", "pgs", "nnls", "lemke", "ccg", "ccg_v2", "ppcg"]

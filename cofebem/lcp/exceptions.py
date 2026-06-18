"""
Exception hierarchy for the :mod:`cofebem.lcp` subpackage.

All exceptions raised by :mod:`cofebem.lcp` and :mod:`cofebem.lcp.solvers`
inherit from :class:`LCPError`, so callers can catch every package-specific
error through a single type while still distinguishing cases (invalid
problem data, invalid solver options, unsupported matrices, ...) via the
more specific subclasses.
"""

from __future__ import annotations

__all__ = [
    "LCPError",
    "InvalidLCPError",
    "InvalidSolverOptionError",
    "UnsupportedSolverError",
    "UnsupportedMatrixError",
    "LCPNumericalError",
]


class LCPError(Exception):
    """
    Base exception for errors raised by the LCP subpackage.

    All package-specific exceptions inherit from this class, allowing users
    to catch every CoFeBEM LCP error through a single exception type.
    """


class InvalidLCPError(LCPError, ValueError):
    """
    Raised when an LCP definition or input is invalid.

    Examples include:

    - a non-square matrix;
    - incompatible dimensions between ``M`` and ``q``;
    - invalid initial-guess dimensions;
    - NaN or infinite coefficients;
    - unsupported numerical data types.
    """


class InvalidSolverOptionError(LCPError, ValueError):
    """
    Raised when a solver option has an invalid value.

    Examples include:

    - a non-positive tolerance;
    - a non-positive maximum iteration count;
    - a PSOR relaxation factor outside the supported interval;
    - an invalid restart frequency.
    """


class UnsupportedSolverError(LCPError, ValueError):
    """
    Raised when the requested LCP solution method is unavailable.

    This may occur when the solver name is unknown or when a requested
    implementation backend is not available.
    """


class UnsupportedMatrixError(LCPError, TypeError):
    """
    Raised when a solver cannot operate on the supplied matrix format.

    For example, Lemke's method may require an explicit dense matrix, while
    a matrix-vector-based solver may accept a hierarchical matrix operator.
    """


class LCPNumericalError(LCPError, RuntimeError):
    """
    Raised when a numerical operation cannot be completed safely.

    This exception should be reserved for exceptional failures, such as an
    invalid factorization or an unrecoverable arithmetic condition.

    Ordinary solver outcomes such as reaching the maximum iteration count,
    stagnation, numerical breakdown detected by the algorithm, or Lemke ray
    termination should normally be represented through ``LCPStatus`` in the
    returned ``LCPResult`` rather than raised as exceptions.
    """

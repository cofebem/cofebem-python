from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .exceptions import InvalidLCPError

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LCP:
    """
    A linear complementarity problem LCP(M, q).

    Defines the problem of finding ``z`` such that::

        z >= 0,
        w = M @ z + q >= 0,
        z.T @ w = 0.

    Instances are immutable. ``M`` and ``q`` are validated and converted to
    ``float64`` NumPy arrays in ``__post_init__``.

    Parameters
    ----------
    M : array_like of shape (n, n)
        The LCP matrix.
    q : array_like of shape (n,)
        The LCP vector.

    Attributes
    ----------
    M : ndarray of shape (n, n)
        The validated, real-valued ``float64`` LCP matrix.
    q : ndarray of shape (n,)
        The validated, real-valued ``float64`` LCP vector.

    Raises
    ------
    InvalidLCPError
        If ``M`` is complex, is not two-dimensional, is not square, does not
        contain finite values, or is empty; or if ``q`` is not
        one-dimensional, does not contain finite values, or has a size that
        does not match ``M``.

    Examples
    --------
    >>> import numpy as np
    >>> problem = LCP(M=np.array([[2.0, 1.0], [1.0, 2.0]]), q=np.array([-1.0, -1.0]))
    >>> problem.size
    2
    """

    M: Matrix
    q: Vector

    def __post_init__(self) -> None:
        # Check for complex before converting: np.asarray(..., dtype=float64)
        # silently discards imaginary parts, making the check below a no-op.
        if np.iscomplexobj(self.M):
            raise InvalidLCPError(
                "M must be real-valued; complex matrices are not supported."
            )

        M = np.asarray(self.M, dtype=np.float64)
        q = np.asarray(self.q, dtype=np.float64)

        if M.ndim != 2:
            raise InvalidLCPError("M must be two-dimensional.")

        if M.shape[0] != M.shape[1]:
            raise InvalidLCPError(f"M must be square, got shape {M.shape}.")

        if q.ndim != 1:
            raise InvalidLCPError("q must be one-dimensional.")

        if M.shape[0] != q.size:
            raise InvalidLCPError(
                f"Incompatible dimensions: M has shape {M.shape}, "
                f"while q has size {q.size}."
            )

        if M.shape[0] == 0:
            raise InvalidLCPError("M must not be empty.")

        if not np.all(np.isfinite(M)):
            raise InvalidLCPError("M contains NaN or infinite values.")

        if not np.all(np.isfinite(q)):
            raise InvalidLCPError("q contains NaN or infinite values.")

        try:
            M = np.asarray(M, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidLCPError(
                "M could not be converted to a float64 NumPy array."
            ) from exc

        object.__setattr__(self, "M", M)
        object.__setattr__(self, "q", q)

    @property
    def size(self) -> int:
        """int: The number of variables ``n`` (i.e. ``q.size``)."""
        return self.q.size

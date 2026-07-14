from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .exceptions import InvalidLCPError

Vector = NDArray[np.float64]


class MatrixOperator(Protocol):
    """Minimal matrix operator accepted by iterative LCP solvers."""

    shape: tuple[int, int]

    def __matmul__(self, vector: Vector) -> Vector: ...


Matrix = NDArray[np.float64] | MatrixOperator


@dataclass(frozen=True, slots=True)
class LCP:
    """
    A linear complementarity problem LCP(M, q).

    Defines the problem of finding ``z`` such that::

        z >= 0,
        w = M @ z + q >= 0,
        z.T @ w = 0.

    Instances are immutable. Dense ``M`` and ``q`` are converted to
    ``float64`` NumPy arrays. A matrix operator with ``shape`` and ``@`` is
    retained without materialisation, allowing hierarchical LCP solves.

    Parameters
    ----------
    M : array_like or matrix operator of shape (n, n)
        The LCP matrix. Operators are supported by iterative solvers only.
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
        q = np.asarray(self.q, dtype=np.float64)

        is_operator = (
            not isinstance(self.M, np.ndarray)
            and hasattr(self.M, "shape")
            and hasattr(self.M, "__matmul__")
        )
        if is_operator:
            M = self.M
            try:
                shape = tuple(M.shape)
            except (TypeError, ValueError) as exc:
                raise InvalidLCPError("M operator has an invalid shape.") from exc
            if len(shape) != 2:
                raise InvalidLCPError("M must be two-dimensional.")
            if shape[0] != shape[1]:
                raise InvalidLCPError(f"M must be square, got shape {shape}.")
        else:
            # Check before converting: dtype=float64 silently discards an
            # imaginary part.
            if np.iscomplexobj(self.M):
                raise InvalidLCPError(
                    "M must be real-valued; complex matrices are not supported."
                )
            try:
                M = np.asarray(self.M, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise InvalidLCPError(
                    "M could not be converted to a float64 NumPy array."
                ) from exc
            shape = M.shape
            if M.ndim != 2:
                raise InvalidLCPError("M must be two-dimensional.")
            if shape[0] != shape[1]:
                raise InvalidLCPError(f"M must be square, got shape {shape}.")

        if q.ndim != 1:
            raise InvalidLCPError("q must be one-dimensional.")

        if shape[0] != q.size:
            raise InvalidLCPError(
                f"Incompatible dimensions: M has shape {shape}, "
                f"while q has size {q.size}."
            )

        if shape[0] == 0:
            raise InvalidLCPError("M must not be empty.")

        if not is_operator and not np.all(np.isfinite(M)):
            raise InvalidLCPError("M contains NaN or infinite values.")

        if not np.all(np.isfinite(q)):
            raise InvalidLCPError("q contains NaN or infinite values.")

        object.__setattr__(self, "M", M)
        object.__setattr__(self, "q", q)

    @property
    def size(self) -> int:
        """int: The number of variables ``n`` (i.e. ``q.size``)."""
        return self.q.size

    @property
    def uses_operator(self) -> bool:
        """Whether ``M`` is an implicit matrix operator rather than an array."""
        return not isinstance(self.M, np.ndarray)

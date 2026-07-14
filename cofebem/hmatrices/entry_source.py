"""Entry sources used to build H-matrices without a global dense matrix."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MatrixEntrySource(Protocol):
    """Protocol for blockwise access to an implicitly represented matrix.

    Implementations must return the Cartesian product of the requested row
    and column indices.  H-matrix construction uses this method for dense
    near-field leaves and for the individual rows and columns selected by
    partial ACA; it never needs to request the global matrix.
    """

    shape: tuple[int, int]

    def get_block(
        self, row_indices: np.ndarray, column_indices: np.ndarray
    ) -> np.ndarray:
        """Return ``A[np.ix_(row_indices, column_indices)]``."""

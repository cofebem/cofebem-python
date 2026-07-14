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


class IndexedEntrySource:
    """Principal submatrix view of another entry source.

    Local row and column indices are mapped through ``indices`` before the
    underlying source is queried. This lets ACA construct only a potential
    contact submatrix without materialising or querying the excluded rows and
    columns.
    """

    def __init__(self, source: MatrixEntrySource, indices: np.ndarray) -> None:
        if not isinstance(source, MatrixEntrySource):
            raise TypeError("source must implement MatrixEntrySource")
        if source.shape[0] != source.shape[1]:
            raise ValueError("source must be square for a principal submatrix")
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise ValueError("indices must not be empty")
        if np.any(selected < 0) or np.any(selected >= source.shape[0]):
            raise IndexError("selected index outside source matrix")
        if np.unique(selected).size != selected.size:
            raise ValueError("indices must be unique")
        self.source = source
        self.indices = selected.copy()
        self.shape = (selected.size, selected.size)

    def get_block(
        self, row_indices: np.ndarray, column_indices: np.ndarray
    ) -> np.ndarray:
        """Return a Cartesian block in the restricted local ordering."""
        rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        columns = np.asarray(column_indices, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("row index outside restricted matrix")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise IndexError("column index outside restricted matrix")
        return self.source.get_block(self.indices[rows], self.indices[columns])

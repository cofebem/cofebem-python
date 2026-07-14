"""Partially pivoted ACA driven only by selected matrix rows and columns."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..entry_source import MatrixEntrySource


def aca_partial_entry(
    source: MatrixEntrySource,
    row_indices: np.ndarray,
    column_indices: np.ndarray,
    tol: float = 1.0e-6,
    k_max: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate a block using only adaptively selected rows and columns.

    Unlike dense ACA routines, this function never materialises its input
    block.  At rank ``r`` it has requested at most ``r`` complete rows and
    ``r`` complete columns of that block (plus rows tried after a zero
    pivot).  The returned factors satisfy ``A_block ~= U @ V.T``.
    """
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    columns = np.asarray(column_indices, dtype=np.int64).reshape(-1)
    m, n = rows.size, columns.size
    if m == 0 or n == 0:
        return np.zeros((m, 0)), np.zeros((n, 0))
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if k_max <= 0:
        raise ValueError("k_max must be positive")

    rank_limit = min(m, n, k_max)
    U_cols: list[np.ndarray] = []
    V_cols: list[np.ndarray] = []
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    approximation_norm_sq = 0.0
    scale = 0.0
    next_row = 0

    for _ in range(rank_limit):
        # A zero residual row need not imply a zero block. Try unused rows
        # until a viable cross is found, without requesting the full block.
        residual_row = None
        pivot_column = None
        pivot = 0.0
        candidates = [next_row] + [i for i in range(m) if i not in used_rows and i != next_row]
        for i in candidates:
            if i in used_rows:
                continue
            row = source.get_block(rows[i : i + 1], columns)[0].copy()
            for u, v in zip(U_cols, V_cols):
                row -= u[i] * v
            scale = max(scale, float(np.max(np.abs(row), initial=0.0)))
            available = np.ones(n, dtype=bool)
            if used_columns:
                available[list(used_columns)] = False
            if not np.any(available):
                break
            masked = np.where(available, np.abs(row), -1.0)
            j = int(np.argmax(masked))
            candidate_pivot = float(row[j])
            used_rows.add(i)
            if abs(candidate_pivot) > tol * max(scale, np.finfo(float).tiny):
                next_row = i
                residual_row = row
                pivot_column = j
                pivot = candidate_pivot
                break

        if residual_row is None or pivot_column is None:
            break

        residual_column = source.get_block(rows, columns[pivot_column : pivot_column + 1])[:, 0].copy()
        for u, v in zip(U_cols, V_cols):
            residual_column -= v[pivot_column] * u

        # Both queried crosses contain the pivot. Averaging is unnecessary:
        # the row value is used so the new term interpolates that whole row.
        u_new = residual_column / pivot
        v_new = residual_row

        term_norm_sq = float(np.dot(u_new, u_new) * np.dot(v_new, v_new))
        cross = 0.0
        for u, v in zip(U_cols, V_cols):
            cross += float(np.dot(u_new, u) * np.dot(v_new, v))
        approximation_norm_sq = max(
            0.0, approximation_norm_sq + 2.0 * cross + term_norm_sq
        )

        U_cols.append(u_new)
        V_cols.append(v_new)
        used_columns.add(pivot_column)

        remaining_rows = np.ones(m, dtype=bool)
        if used_rows:
            remaining_rows[list(used_rows)] = False
        if not np.any(remaining_rows):
            break
        next_row = int(
            np.argmax(np.where(remaining_rows, np.abs(residual_column), -1.0))
        )

        if term_norm_sq <= tol**2 * max(
            approximation_norm_sq, np.finfo(float).tiny
        ):
            break

    if not U_cols:
        return np.zeros((m, 0)), np.zeros((n, 0))
    return np.column_stack(U_cols), np.column_stack(V_cols)

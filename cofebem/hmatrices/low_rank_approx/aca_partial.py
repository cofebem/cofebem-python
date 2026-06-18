from typing import Tuple
import numpy as np


def aca_partial(
    A: np.ndarray, tol: float = 1.0e-6, k_max: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Partial Adaptive Cross Approximation without materialising the full residual.

    Only individual rows and columns of the residual are evaluated, keeping
    memory at O((m+n)·r).  Pivots are chosen greedily by tracking the largest
    entry in the current residual column; the next column is determined by the
    largest entry in the current residual row.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Matrix to approximate.
    tol : float
        Relative stopping tolerance.  The algorithm stops when the new
        outer-product norm squared falls below ``tol² * ||approx||_F²``, or
        when the pivot magnitude drops below ``tol * max|A|``.
    k_max : int
        Maximum number of cross terms (rank cap).

    Returns
    -------
    U : ndarray of shape (m, r)
        Column factors stacked as columns.
    V : ndarray of shape (n, r)
        Row factors stacked as columns.

    Notes
    -----
    The approximation satisfies ``A ≈ U @ V.T``.
    Returns zero-column matrices when *A* is the zero matrix.
    """
    m, n = A.shape
    max_A = float(np.abs(A).max())
    if max_A == 0.0:
        return np.zeros((m, 0)), np.zeros((n, 0))

    U_cols: list[np.ndarray] = []
    V_cols: list[np.ndarray] = []
    norm_sq = 0.0  # running ||approx||_F^2 (cross-terms dropped)

    # start with the column whose maximum entry is largest
    j_curr = int(np.abs(A).max(axis=0).argmax())

    for _ in range(k_max):
        # residual column j_curr without materializing R
        r_col = A[:, j_curr].copy()
        for u_l, v_l in zip(U_cols, V_cols):
            r_col -= v_l[j_curr] * u_l

        i_piv = int(np.abs(r_col).argmax())
        delta = r_col[i_piv]

        if abs(delta) <= tol * max_A:
            break

        u_k = r_col  # shape (m,)

        # residual row i_piv
        r_row = A[i_piv, :].copy()
        for u_l, v_l in zip(U_cols, V_cols):
            r_row -= u_l[i_piv] * v_l
        v_k = r_row / delta  # shape (n,)

        nk_sq = float(np.dot(u_k, u_k) * np.dot(v_k, v_k))
        norm_sq += nk_sq

        U_cols.append(u_k)
        V_cols.append(v_k)

        if nk_sq <= tol**2 * norm_sq:
            break

        j_curr = int(np.abs(v_k).argmax())

    if not U_cols:
        return np.zeros((m, 0)), np.zeros((n, 0))

    return np.column_stack(U_cols), np.column_stack(V_cols)

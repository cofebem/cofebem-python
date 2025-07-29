from typing import Tuple
import numpy as np


def aca_partial(
    A: np.ndarray, tol: float = 1.0e-6, k_max: int = 50
) -> Tuple[np.ndarray, np.ndarray]:

    m, n = A.shape
    R = A.copy()

    max_A = np.abs(A).max()
    norm_A2 = np.linalg.norm(A, 2)

    for j_start in range(n):
        if np.any(np.abs(R[:, j_start]) > 1.0e-15):
            j_curr = j_start
            break
    else:
        return np.zeros_like(A), np.zeros_like(A)

    U_cols: list[np.ndarray] = []
    V_rows: list[np.ndarray] = []
    k = 0

    while True:
        i_piv = np.abs(R[:, j_curr]).argmax()
        delta = R[i_piv, j_curr]

        if abs(delta) <= tol * max_A:
            break

        u_k = R[:, j_curr].copy()
        v_k = R[i_piv, :].copy() / delta

        R -= np.outer(u_k, v_k)

        U_cols.append(u_k)
        V_rows.append(v_k)
        k += 1

        if np.linalg.norm(u_k) * np.linalg.norm(v_k) <= tol * norm_A2:
            break

        if k_max is not None and k >= k_max:
            break

        j_curr = np.abs(v_k).argmax()

    U = np.column_stack(U_cols)
    V = np.vstack(V_rows)

    return U, V

from typing import Tuple
import numpy as np


def aca_full(
    A: np.ndarray, tol: float = 1.0e-6, k_max: int = 50
) -> Tuple[np.ndarray, np.ndarray]:

    m, n = A.shape
    R = A.copy()

    norm_A2 = np.linalg.norm(A, 2)
    max_A = np.abs(A).max()

    U_cols: list[np.ndarray] = []
    V_rows: list[np.ndarray] = []

    k = 0
    while True:
        flat_idx = np.abs(R).argmax()
        i_piv, j_piv = divmod(flat_idx, n)
        delta = R[i_piv, j_piv]

        if abs(delta) <= tol * max_A:
            break

        u_k = R[:, j_piv].copy()
        v_k = R[i_piv, :].copy() / delta

        R -= np.outer(u_k, v_k)

        U_cols.append(u_k)
        V_rows.append(v_k)
        k += 1

        if np.linalg.norm(u_k) * np.linalg.norm(v_k) <= tol * norm_A2:
            break

        if k_max is not None and k >= k_max:
            break

    U = np.column_stack(U_cols)
    V = np.column_stack(V_rows)

    return U, V

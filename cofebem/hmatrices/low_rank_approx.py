from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray


def truncated_svd(
    A: NDArray[np.floating],
    tol: float = 1.0e-6,
    max_rank: int = 50,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:

    U_full, S, Vh = np.linalg.svd(A, full_matrices=False)

    r = int(np.sum(S > tol * S[0]))
    r = min(r, max_rank)

    sqrt_S = np.sqrt(S[:r])

    U = U_full[:, :r] * sqrt_S
    V = (Vh[:r, :].T) * sqrt_S

    return U, V


def randomized_svd(
    A: NDArray[np.floating],
    rank: int,
    *,
    oversample: int = 10,
    n_iter: int = 2,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:

    m, n = A.shape
    r = min(rank, m, n)
    p = max(0, oversample)

    rng = np.random.default_rng(random_state)

    Omega = rng.standard_normal(size=(n, r + p))
    Y = A @ Omega  # m × (r+p)

    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y, mode="reduced")  # Q  :  m × (r+p)

    B = Q.T @ A
    Ub, S, Vh = np.linalg.svd(B, full_matrices=False)

    Ub = Ub[:, :r]
    S = S[:r]
    Vh = Vh[:r, :]

    sqrt_S = np.sqrt(S)
    U = (Q @ Ub) * sqrt_S
    V = (Vh.T) * sqrt_S

    return U, V


def aca(
    A: np.ndarray,
    tol: float = 1e-6,
    max_rank: Optional[int] = None,
    rel: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    A = np.asarray(A)
    m, n = A.shape
    if max_rank is None:
        max_rank = min(m, n)

    U = np.empty((m, 0), dtype=A.dtype)
    V = np.empty((n, 0), dtype=A.dtype)

    normA = np.linalg.norm(A, "fro")
    stop_val = tol * normA if rel else tol

    row_used: set[int] = set()
    i = 0
    r = 0
    while r < max_rank:
        found = False
        loops = 0
        while not found and loops < m:
            if i in row_used:
                i = (i + 1) % m
                loops += 1
                continue

            row = A[i, :]
            if r:
                row = row - U[i, :] @ V.T
            j = int(np.argmax(np.abs(row)))
            pivot = row[j]

            if abs(pivot) > stop_val:
                found = True
            else:
                row_used.add(i)
                i = (i + 1) % m
                loops += 1

        if not found:
            break

        col = A[:, j]
        if r:
            col = col - U @ V[j, :]

        u = col.copy()
        v = row / pivot
        if abs(pivot) * np.linalg.norm(u) * np.linalg.norm(v) <= stop_val:
            break

        U = np.column_stack((U, u))
        V = np.column_stack((V, v))
        r += 1
        row_used.add(i)
        i = (i + 1) % m

    return U, V

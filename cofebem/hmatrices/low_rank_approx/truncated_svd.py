from typing import Tuple

import numpy as np


def truncated_svd(
    A: np.ndarray,
    tol: float = 1.0e-6,
    k_max: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:

    U_full, S, Vh = np.linalg.svd(A, full_matrices=False)

    r = int(np.sum(S > tol * S[0]))
    r = min(r, k_max)

    sqrt_S = np.sqrt(S[:r])

    U = U_full[:, :r] * sqrt_S
    V = (Vh[:r, :].T) * sqrt_S

    return U, V

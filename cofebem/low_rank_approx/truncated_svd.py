import numpy as np
from typing import Tuple


def truncated_svd(
    A: np.ndarray,
    tol: float = 1.0e-6,
    k_max: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:

    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    r = min(int(np.sum(S > tol * S[0])), k_max)

    sqrt_S = np.diag(np.sqrt(S[:r]))

    Ut, Vt = U[:, :r] @ sqrt_S, sqrt_S @ Vh[:r, :]

    return Ut, Vt.T

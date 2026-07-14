import numpy as np
from typing import Tuple


def truncated_svd(
    A: np.ndarray,
    tol: float = 1.0e-6,
    k_max: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a low-rank approximation via truncated SVD.

    Computes the full SVD of *A* and retains only the singular triplets whose
    singular value satisfies ``S[i] > tol * S[0]``, up to ``k_max`` terms.
    The result factorises as ``A ≈ U @ V.T``.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Matrix to approximate.
    tol : float
        Relative truncation threshold applied to the singular values.
    k_max : int
        Maximum number of singular triplets to retain.

    Returns
    -------
    U : ndarray of shape (m, r)
        Left factor, columns scaled by ``sqrt(S[:r])``.
    V : ndarray of shape (n, r)
        Right factor, columns scaled by ``sqrt(S[:r])``.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    r = min(int(np.sum(S > tol * S[0])), k_max)

    sqrt_S = np.diag(np.sqrt(S[:r]))

    Ut, Vt = U[:, :r] @ sqrt_S, sqrt_S @ Vh[:r, :]

    return Ut, Vt.T

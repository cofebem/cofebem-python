from typing import Tuple
import numpy as np

from .aca_partial import aca_partial


def aca_plus(
    A: np.ndarray, tol: float = 1.0e-6, k_max: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """ACA+ — partial ACA followed by a QR-SVD recompression step.

    Runs :func:`aca_partial` with a relaxed tolerance (``10 * tol``) to obtain
    an initial cross approximation, then recompresses it via QR decompositions
    and a compact SVD of the small ``r×r`` core, truncating to numerical rank.
    This typically yields a tighter low-rank representation than plain ACA at
    comparable cost.

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Matrix to approximate.
    tol : float
        Relative truncation tolerance for the final SVD step.
    k_max : int
        Maximum rank passed to the inner :func:`aca_partial` call.

    Returns
    -------
    U : ndarray of shape (m, s)
        Left factor, columns scaled by ``sqrt(S[:s])``.
    V : ndarray of shape (n, s)
        Right factor, columns scaled by ``sqrt(S[:s])``.

    Notes
    -----
    The approximation satisfies ``A ≈ U @ V.T``.
    ``s ≤ r`` where ``r`` is the rank returned by the inner ACA.
    """
    m = A.shape[0]

    # initial cross approximation with a looser tolerance
    U, V = aca_partial(A, tol=tol * 10.0, k_max=k_max)

    r = U.shape[1]
    if r == 0:
        return U, V

    # QR decompositions of the tall factors (m×r and n×r)
    Q_U, R_U = np.linalg.qr(U)  # Q_U: (m, r), R_U: (r, r)
    Q_V, R_V = np.linalg.qr(V)  # Q_V: (n, r), R_V: (r, r)

    # compact SVD of the small r×r core
    W, S, Zt = np.linalg.svd(R_U @ R_V.T, full_matrices=False)

    # truncate to numerical rank
    s = max(1, int(np.sum(S > tol * S[0])))
    sqrt_S = np.sqrt(S[:s])

    U_out = Q_U @ (W[:, :s] * sqrt_S)       # (m, s)
    V_out = Q_V @ (Zt[:s, :].T * sqrt_S)    # (n, s)

    return U_out, V_out

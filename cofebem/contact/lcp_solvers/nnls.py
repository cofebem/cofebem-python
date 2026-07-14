import numpy as np
from scipy.optimize import nnls


def nnls_lawson_hanson(A, b, tol=1e-12, max_iter=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)

    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("Incompatible shapes between A and b.")

    m, n = A.shape
    if max_iter is None:
        max_iter = 30 * n

    P = np.zeros(n, dtype=bool)
    x = np.zeros(n, dtype=float)

    outer_iter = 0
    inner_iter_total = 0

    while outer_iter < max_iter:
        outer_iter += 1

        residual = b - A @ x
        w = A.T @ residual

        candidates = np.where(~P)[0]
        if candidates.size == 0:
            break

        t = candidates[np.argmax(w[candidates])]
        if w[t] <= tol:
            break

        P[t] = True

        while True:
            inner_iter_total += 1
            p_idx = np.where(P)[0]
            z = np.zeros(n, dtype=float)

            A_P = A[:, p_idx]

            z_P, *_ = np.linalg.lstsq(A_P, b, rcond=None)
            z[p_idx] = z_P

            if np.all(z[p_idx] > -tol):
                z[p_idx] = np.maximum(z[p_idx], 0.0)
                x = z
                break

            negative = (z < 0) & P
            alpha = np.min(x[negative] / (x[negative] - z[negative]))

            x = x + alpha * (z - x)

            hit_zero = P & (x <= tol)
            x[hit_zero] = 0.0
            P[hit_zero] = False

    info = {
        "outer_iterations": outer_iter,
        "inner_iterations": inner_iter_total,
        "passive_set": np.where(P)[0],
        "residual_norm": np.linalg.norm(A @ x - b),
    }
    return x, info


def lcp_to_nnls_data(M, q, check_spd=True):
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix.")
    if q.ndim != 1 or q.shape[0] != M.shape[0]:
        raise ValueError("q must be a vector with length equal to M.shape[0].")

    if check_spd and not np.allclose(M, M.T, atol=1e-12, rtol=1e-12):
        raise ValueError("M must be symmetric to use this LCP -> QP -> NNLS route.")

    try:
        L = np.linalg.cholesky(M)  # M = L L^T
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "M is not positive definite. This NNLS reduction requires SPD M."
        ) from e

    A = L.T
    b = np.linalg.solve(L, -q)
    return A, b


def lawson_hanson_nnls_lcp(M, q, tol=1e-10, max_iter=None, check_spd=True):
    A, b = lcp_to_nnls_data(M, q, check_spd=check_spd)
    z, nnls_info = nnls_lawson_hanson(A, b, tol=tol, max_iter=max_iter)
    w = M @ z + q

    comp = z * w
    info = {
        **nnls_info,
        "min_z": np.min(z),
        "min_w": np.min(w),
        "complementarity_inf": np.linalg.norm(comp, ord=np.inf),
        "feasibility_violation": max(0.0, -np.min(z), -np.min(w)),
        "kkt_residual_inf": np.linalg.norm(np.minimum(z, w), ord=np.inf),
    }
    return z, w, info


def scipy_nnls_lcp(M, q, tol=1e-10, max_iter=None, check_spd=True):
    A, b = lcp_to_nnls_data(M, q, check_spd=check_spd)
    z, rnorm = nnls(A, b, maxiter=max_iter, atol=tol)
    w = M @ z + q

    comp = z * w
    info = {
        "residual_norm": rnorm,
        "min_z": np.min(z),
        "min_w": np.min(w),
        "complementarity_inf": np.linalg.norm(comp, ord=np.inf),
        "feasibility_violation": max(0.0, -np.min(z), -np.min(w)),
        "kkt_residual_inf": np.linalg.norm(np.minimum(z, w), ord=np.inf),
    }
    return z, w, info


# NNLS : min x>0 norm(Ax-b)^2 = min x>0 *(Ax-b).T(Ax-b) = min x>0 x.T A.T A x - 2 b.T A x + b.T b
# LCP : find z>=0, w>=0, w = Mz + q, z.T w = 0
# LCP -> QP : min z>=0 0.5 z.T M z + q.T * z
# M = A.T A = Q*Q.T
# q = -A.T b

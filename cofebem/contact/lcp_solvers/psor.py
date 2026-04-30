import numpy as np


def psor_lcp(M, q, omega=1.0, tol=1e-10, max_iter=10000, z0=None):
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(q)

    if z0 is None:
        z = np.zeros(n, dtype=float)
    else:
        z = np.maximum(0.0, np.asarray(z0, dtype=float).copy())

    diag = np.diag(M)
    if np.any(np.abs(diag) < 1e-15):
        raise ValueError("M has a zero diagonal entry; PSOR update is not defined.")

    history = []

    for k in range(max_iter):
        z_old = z.copy()

        for i in range(n):
            s1 = np.dot(M[i, :i], z[:i])  # new values
            s2 = np.dot(M[i, i + 1 :], z_old[i + 1 :])  # old values

            z_gs = (-q[i] - s1 - s2) / diag[i]
            z_relaxed = (1.0 - omega) * z_old[i] + omega * z_gs
            z[i] = max(0.0, z_relaxed)

        w = M @ z + q

        # Complementarity residual
        comp_res = np.linalg.norm(np.minimum(z, w), ord=np.inf)

        # Feasibility violations
        neg_z = max(0.0, -np.min(z))
        neg_w = max(0.0, -np.min(w))

        # Iteration change
        step_res = np.linalg.norm(z - z_old, ord=np.inf)

        res = max(comp_res, neg_z, neg_w, step_res)
        history.append(res)

        if res < tol:
            return z, w, k + 1, history

    return z, M @ z + q, max_iter, history

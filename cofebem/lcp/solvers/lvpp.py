import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg


def lvpp(
    M,
    q,
    z0=None,
    alpha=1.0,
    growth=2.0,
    tol=1e-8,
    max_iter=100,
    max_newton=50,
    cg_tol=1e-11,
    max_cg=None,
    scales=None,
):
    if np.iscomplexobj(q):
        raise ValueError("q must be real.")
    q = np.asarray(q, dtype=float)
    M = aslinearoperator(M)
    if np.issubdtype(M.dtype, np.complexfloating):
        raise ValueError("M must be real.")
    n = q.size
    if q.ndim != 1 or n == 0 or M.shape != (n, n):
        raise ValueError("M must be square and q must be a nonempty vector.")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite.")
    if not np.all(np.isfinite([alpha, growth, tol, cg_tol])):
        raise ValueError("Solver parameters must be finite.")
    if alpha <= 0 or growth < 1 or tol <= 0 or not 0 < cg_tol < 1:
        raise ValueError("Require alpha > 0, growth >= 1, tol > 0, 0 < cg_tol < 1.")
    if max_iter < 1 or max_newton < 1:
        raise ValueError("Iteration limits must be positive.")

    matvecs = 0

    def multiply(v):
        nonlocal matvecs
        matvecs += 1
        return np.asarray(M @ v).reshape(n)

    if scales is None:
        w_scale = np.linalg.norm(q, np.inf) or 1.0
        m_scale = np.linalg.norm(multiply(np.ones(n)), np.inf)
        if not np.isfinite(m_scale) or m_scale <= 0:
            raise ValueError("M must represent a finite SPD operator.")
        z_scale = w_scale / m_scale
    else:
        z_scale, w_scale = scales
    if not np.all(np.isfinite([z_scale, w_scale])) or min(z_scale, w_scale) <= 0:
        raise ValueError("scales must contain positive finite z and w scales.")

    def apply(v):
        return (z_scale / w_scale) * multiply(v)

    b = q / w_scale
    x = np.ones(n) if z0 is None else np.asarray(z0, dtype=float) / z_scale
    if x.shape != (n,) or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("z0 must be a finite, strictly positive vector.")
    psi = np.log(x)
    w = apply(x) + b
    residual = np.linalg.norm(np.minimum(x, w), np.inf)
    history = [float(residual)]
    newton_iterations = 0
    cg_iterations = 0
    status = "max_iter"
    outer = 0
    inner_tol = 0.01 * tol

    def count_cg(_):
        nonlocal cg_iterations
        cg_iterations += 1

    for outer in range(1, max_iter + 1):
        if residual <= tol:
            status = "converged"
            outer -= 1
            break
        previous = psi.copy()
        F = w + (psi - previous) / alpha

        for _ in range(max_newton):
            if np.linalg.norm(F, np.inf) <= inner_tol:
                break
            root = np.exp(0.5 * psi)
            H = LinearOperator(
                (n, n),
                matvec=lambda v: v + alpha * root * apply(root * v),
                dtype=float,
            )
            y, flag = cg(
                H,
                -alpha * root * F,
                rtol=cg_tol,
                atol=0.0,
                maxiter=max_cg,
                callback=count_cg,
            )
            if flag != 0 or not np.all(np.isfinite(y)):
                status = "cg_failed"
                break

            # This equals y / root without dividing by tiny pressures.
            direction = -alpha * (F + apply(root * y))
            step = 1.0
            norm_F = np.linalg.norm(F)
            for _ in range(50):
                trial_psi = psi + step * direction
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    trial_x = np.exp(trial_psi)
                if not np.all(np.isfinite(trial_x)):
                    step *= 0.5
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    trial_w = apply(trial_x) + b
                    trial_F = trial_w + (trial_psi - previous) / alpha
                if (
                    np.all(np.isfinite(trial_F))
                    and np.linalg.norm(trial_F) <= (1 - 1e-4 * step) * norm_F
                ):
                    break
                step *= 0.5
            else:
                status = "line_search_failed"
                break

            psi, x, w, F = trial_psi, trial_x, trial_w, trial_F
            newton_iterations += 1
        else:
            if np.linalg.norm(F, np.inf) > inner_tol:
                status = "newton_failed"

        residual = np.linalg.norm(np.minimum(x, w), np.inf)
        history.append(float(residual))
        if residual <= tol:
            status = "converged"
            break
        if status != "max_iter":
            break
        alpha *= growth

    info = {
        "converged": status == "converged",
        "status": status,
        "iterations": outer,
        "newton_iterations": newton_iterations,
        "cg_iterations": cg_iterations,
        "matvecs": matvecs,
        "residual": float(residual),
        "history": np.asarray(history),
        "scales": (float(z_scale), float(w_scale)),
    }
    return z_scale * x, w_scale * w, info

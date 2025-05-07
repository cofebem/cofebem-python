import numpy as np


class PCG:
    def __init__(
        S,
        error_type,
        gap,
        max_iter,
        tolerance,
        pressure_factor=1e12,
        initial_pressure=None,
    ):
        error_history = np.zeros((max_iter, 3))
        ub = -gap
        # Warmed start does not work well
        if initial_pressure is not None:
            # p = initial_pressure
            # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
            # p[gap>0] = 0
            p = np.maximum(-gap, 0) * pressure_factor
        else:
            p = np.zeros_like(ub)
            p = np.maximum(-gap, 0) * pressure_factor

        w = np.inner(S, p) - ub
        # w -= np.mean(w) #new
        t = w
        t_ = np.zeros_like(w)
        d = 0
        error = 1
        error_ = 1
        for iter in range(max_iter):
            if iter > 0:
                t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
                t[p <= 0] = 0
            q = np.inner(S, t)
            tau = np.inner(w, t) / np.inner(t, q)
            p = p - tau * t
            p = np.maximum(p, 0)
            zero_pressure = np.where(p == 0)[0]
            penetration = np.where(w < 0)[0]
            set_I = np.intersect1d(zero_pressure, penetration)
            if len(set_I) == 0:
                d = 1
            else:
                d = 0
                p[set_I] -= tau * w[set_I]
            t_ = t

            w = np.inner(S, p) - ub
            nw = np.linalg.norm(w, 2)

            error_ = error
            displ_error = np.linalg.norm(w[p > 0], 2) / nw
            ort = np.abs(np.dot(w, p) / nw)

            if error_type == "displacement":
                error = displ_error
            elif error_type == "mix":
                error = np.sqrt(displ_error * ort)
            elif error_type == "nw":
                error = nw
                if abs((error - error_) / error_) < tolerance:
                    error_history[iter, 0] = displ_error
                    error_history[iter, 1] = abs((error - error_) / error_)
                    error_history[iter, 2] = ort
                    return p, np.inner(S, p), error_history[: iter + 1]
            error_history[iter, 0] = displ_error
            error_history[iter, 1] = error
            error_history[iter, 2] = ort
            if error < tolerance:
                break
        return p, np.inner(S, p), error_history[: iter + 1]

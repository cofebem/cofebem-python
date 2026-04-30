import numpy as np


class CCG:
    def __init__(
        self,
        Sc,
        g,
        max_iter,
        tol,
        err_type="displacement",
        pfactor=1e12,
        p0=None,
    ):

        self.Sc = Sc
        self.err_type = err_type
        self.g = g
        self.max_iter = max_iter
        self.tol = tol
        self.pfactor = pfactor
        self.p0 = p0

    def solve(self):
        ub = -self.g
        error_history = np.zeros((self.max_iter, 3))

        if self.p0 is not None:
            p = np.full_like(ub, self.p0)
        else:
            p = np.zeros_like(ub)
            p = np.maximum(-self.g, 0) * self.pfactor

        w = np.inner(self.Sc, p) - ub
        # w -= np.mean(w) #new
        t = w
        t_ = np.zeros_like(w)
        d = 0
        error = 1
        error_ = 1
        for iter in range(self.max_iter):
            if iter > 0:
                t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
                t[p <= 0] = 0
            q = np.inner(self.Sc, t)
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

            w = np.inner(self.Sc, p) - ub
            nw = np.linalg.norm(w, 2)

            error_ = error
            displ_error = np.linalg.norm(w[p > 0], 2) / nw
            ort = np.abs(np.dot(w, p) / nw)

            if self.err_type == "displacement":
                error = displ_error
            elif self.err_type == "mix":
                error = np.sqrt(displ_error * ort)
            elif self.err_type == "nw":
                error = nw
                if abs((error - error_) / error_) < self.tol:
                    error_history[iter, 0] = displ_error
                    error_history[iter, 1] = abs((error - error_) / error_)
                    error_history[iter, 2] = ort
                    return p, np.inner(self.Sc, p), error_history[: iter + 1]
            error_history[iter, 0] = displ_error
            error_history[iter, 1] = error
            error_history[iter, 2] = ort
            if error < self.tol:
                break
        return p, np.inner(self.Sc, p), error_history[: iter + 1]

    def solve_hm(self):
        ub = -self.g
        error_history = np.zeros((self.max_iter, 3))

        if self.p0 is not None:
            # p = initial_pressure
            # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
            # p[gap>0] = 0
            p = np.maximum(-self.g, 0) * self.pfactor
        else:
            p = np.zeros_like(ub)
            p = np.maximum(-self.g, 0) * self.pfactor

        w = self.Sc @ p - ub

        t = w
        t_ = np.zeros_like(w)
        d = 0
        error = 1
        error_ = 1
        for iter in range(self.max_iter):
            if iter > 0:
                t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
                t[p <= 0] = 0
            q = self.Sc @ t
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

            w = self.Sc @ p - ub
            nw = np.linalg.norm(w, 2)

            error_ = error
            displ_error = np.linalg.norm(w[p > 0], 2) / nw
            ort = np.abs(np.dot(w, p) / nw)

            if self.err_type == "displacement":
                error = displ_error
            elif self.err_type == "mix":
                error = np.sqrt(displ_error * ort)
            elif self.err_type == "nw":
                error = nw
                if abs((error - error_) / error_) < self.tol:
                    error_history[iter, 0] = displ_error
                    error_history[iter, 1] = abs((error - error_) / error_)
                    error_history[iter, 2] = ort
                    return p, self.Sc @ p, error_history[: iter + 1]
            error_history[iter, 0] = displ_error
            error_history[iter, 1] = error
            error_history[iter, 2] = ort
            if error < self.tol:
                break
        return (
            p,
            self.Sc @ p,
            error_history[: iter + 1],
        )

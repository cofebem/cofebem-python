import logging
import numpy as np
import numba as nb


LOGGER_NAME = "flexcontact.constrained_cg"
_LOGGER = logging.getLogger(LOGGER_NAME)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
_LOGGER.setLevel(logging.INFO)


def _log_info(message, *args, **kwargs):
    _LOGGER.info(message, *args, **kwargs)

def _log_warning(message, *args, **kwargs):
    _LOGGER.warning(message, *args, **kwargs)

"""
    Active set solver using NNLS solver from scipy.optimize
"""
from scipy.optimize import nnls

def as_nnls_stable(
    K, g0, node_of_elem, *,
    use_minus=True,            # True: g = g0 - K p; False: g = g0 + K p
    max_outer=80,
    tau_add=None,              # add to active if g < -tau_add
    tau_rem=None,              # remove if g >  tau_rem and not supported by pressure
    ptol=0.0,                  # facet pressure threshold; if 0 -> 1e-12 abs
    hold_iters=3,              # persistence before release
    lam=0.0,                   # Tikhonov (can be tiny)
    lam_bump=1e-6,             # increase lam when a 2-cycle is detected
    cont_every=3,              # continuation trigger (stable iters)
    cont_factor=0.5,           # shrink taus and lam by this
    verbose=False
):
    n, m = K.shape
    g0 = np.asarray(g0, float)
    assert g0.shape == (n,)
    sign = -1.0 if use_minus else +1.0  # g = g0 + sign*K p

    # ---- column scaling for NNLS conditioning ----
    col_scale = np.linalg.norm(K, axis=0) + 1e-30
    Ks = K / col_scale

    def gap_from_p(p):
        return g0 + sign * (K @ p)  # always return physical g

    # thresholds
    g_scale = max(1.0, np.linalg.norm(g0, np.inf))
    if tau_add is None: tau_add = 1e-4 * g_scale
    if tau_rem is None: tau_rem = 5.0 * tau_add
    if ptol <= 0.0:     ptol    = 1e-12

    # init
    p = np.zeros(m)
    g = gap_from_p(p)
    A = np.where(g < -tau_add)[0]
    hold = np.zeros(n, dtype=int)
    A_prev = None
    stable_counter = 0

    if verbose:
        print(f"[init] use_minus={use_minus}, |A|={len(A)}, min(g)={g.min():.3e}, "
              f"tau_add={tau_add:.2e}, tau_rem={tau_rem:.2e}, lam={lam:.1e}")

    for it in range(max_outer):
        if len(A) == 0:
            return p, g, {"iters": it, "use_minus": use_minus, "tau_add": tau_add, "lam": lam}

        # --- NNLS on active rows (scaled variables x = diag(col_scale)*p) ---
        KA = Ks[A, :]                                 # (|A|, m)
        rhs = (-g0[A]) if (sign > 0) else (g0[A])     # because g = g0 + sign*K p

        if lam > 0.0:
            L = np.sqrt(lam) * np.eye(m)
            KA_aug  = np.vstack([KA, L])              # (|A|+m, m)
            rhs_aug = np.concatenate([rhs, np.zeros(m)])
            x, _ = nnls(KA_aug, rhs_aug)
        else:
            x, _ = nnls(KA, rhs)

        # back-scale and prune tiny pressures
        p = x / col_scale
        p[p < ptol] = 0.0

        # --- debias on current support S: LS on frozen set (no nonnegativity) ---
        S = np.where(p > ptol)[0]
        if S.size:
            KA_true_S = K[A, :][:, S]                 # unscaled operator on A×S
            ATA = KA_true_S.T @ KA_true_S
            if lam > 0.0:
                ATA.flat[::ATA.shape[0]+1] += lam
            b = (g0[A]) if use_minus else (-g0[A])    # want K_A,S p_S ≈ g0_A (if use_minus)
            rhs_ls = KA_true_S.T @ b
            try:
                p_S = np.linalg.solve(ATA, rhs_ls)
            except np.linalg.LinAlgError:
                p_S = np.linalg.lstsq(ATA, rhs_ls, rcond=None)[0]
            # clip negative components and optionally re-solve once
            neg = p_S < 0
            if np.any(neg):
                keep = ~neg
                S2 = S[keep]
                if S2.size:
                    KA2 = K[A, :][:, S2]
                    ATA2 = KA2.T @ KA2
                    if lam > 0.0:
                        ATA2.flat[::ATA2.shape[0]+1] += lam
                    rhs2 = KA2.T @ b
                    try:
                        p_S2 = np.linalg.solve(ATA2, rhs2)
                    except np.linalg.LinAlgError:
                        p_S2 = np.linalg.lstsq(ATA2, rhs2, rcond=None)[0]
                    p[:] = 0.0
                    p[S2] = np.maximum(p_S2, 0.0)
                else:
                    p[:] = 0.0
            else:
                p[:] = 0.0
                p[S] = np.maximum(p_S, 0.0)

        # update gaps
        g = gap_from_p(p)

        # --- keep nodes touched by positive pressure facets ---
        node_press = np.zeros(n, dtype=bool)
        for e, nodes in enumerate(node_of_elem):
            if p[e] > ptol:
                node_press[nodes] = True

        # --- persistence counters ---
        cand_release = (g > tau_rem) & (~node_press)
        hold[cand_release] += 1
        hold[~cand_release] = 0

        # --- new active set ---
        A_pen  = np.where(g < -tau_add)[0]
        A_keep = np.where(node_press | (hold < hold_iters))[0]
        A_new  = np.union1d(A_pen, A_keep)

        # cycle detection ⇒ merge + bump lam
        if A_prev is not None and np.array_equal(A_new, A_prev):
            A_new = np.union1d(A_new, A)
            lam   = max(lam, lam_bump)

        # stability tracking
        if np.array_equal(A_new, A):
            stable_counter += 1
        else:
            stable_counter = 0

        if verbose:
            print(f"[{it:02d}] |A|={len(A_new):4d}  min(g)={g.min():.3e}  max(p)={p.max():.3e}  "
                  f"tau=({tau_add:.1e},{tau_rem:.1e})  lam={lam:.1e}  stable={stable_counter}")

        # continuation
        if stable_counter >= cont_every and (tau_add > 1e-9 * g_scale or lam > 0.0):
            tau_add  *= cont_factor
            tau_rem   = max(tau_rem * cont_factor, 2.0 * tau_add)
            lam      *= cont_factor
            stable_counter = 0

        # stop if stable and gaps nonnegative within tolerance
        if stable_counter >= 2 and g.min() >= -tau_add:
            break

        A_prev = A
        A = A_new

    # --- final refinement on frozen A ---
    if len(A) > 0:
        KA = Ks[A, :]
        rhs = (-g0[A]) if (sign > 0) else (g0[A])
        if lam > 0.0:
            L = np.sqrt(lam) * np.eye(m)
            KA_aug  = np.vstack([KA, L])
            rhs_aug = np.concatenate([rhs, np.zeros(m)])
            x, _ = nnls(KA_aug, rhs_aug, atol=1e-8)
        else:
            x, _ = nnls(KA, rhs, atol=1e-8)
        p = x / col_scale
        p[p < ptol] = 0.0
        # one last debias on its support
        S = np.where(p > ptol)[0]
        if S.size:
            KA_true_S = K[A, :][:, S]
            ATA = KA_true_S.T @ KA_true_S
            b   = (g0[A]) if use_minus else (-g0[A])
            rhs_ls = KA_true_S.T @ b
            try:
                p_S = np.linalg.solve(ATA, rhs_ls)
            except np.linalg.LinAlgError:
                p_S = np.linalg.lstsq(ATA, rhs_ls, rcond=None)[0]
            p[:] = 0.0
            p[S] = np.maximum(p_S, 0.0)

    g = gap_from_p(p)
    return p, g, {"iters": it + 1, "use_minus": use_minus, "tau_add": tau_add, "lam": lam}


def _augment_rhs(rhs, m, lam):
    """Helper: build augmented RHS for Tikhonov NNLS [K; sqrt(lam) I] x = [rhs; 0]."""
    L = np.zeros(m)  # zeros correspond to regularization part
    return np.concatenate([rhs, L])















def constrained_CG_p0p1(
    K,               # (n x m) nodal gap influence from element pressures
    g0,              # (n,)
    tol,
    p0,              # (m,)
    node_of_elem,    # list of arrays of node indices per element (0..n-1)
    max_iter=200,
    use_minus=None,  # None = auto-detect, True => g = g0 - Kp, False => g = g0 + Kp
    neg_gap_tol=1e-10,
    warm_alpha=1.0
):
    """
    Projected CG for complementarity with p on elements and g on nodes.
    Enforces g >= 0, p >= 0, and p ⊥ g in practice via active nodes.

    Returns
    -------
    p : (m,) optimal pressures
    u : (n,) displacement u = ±Kp consistent with chosen sign
    info : dict
        {'eta_disp', 'eta_ort', 'max_pen', 'iters', 'use_minus'}
    """
    n, m = K.shape
    assert g0.shape[0] == n
    p = np.maximum(np.asarray(p0, dtype=float), 0.0).copy()

    # --- helpers -------------------------------------------------------------
    def gap_plus(p):   # g = g0 + Kp
        return g0 + K @ p

    def gap_minus(p):  # g = g0 - Kp
        return g0 - K @ p

    def elem_avg_gap(g):
        ge = np.zeros(m)
        for e, nodes in enumerate(node_of_elem):
            ge[e] = g[nodes].mean()
        return ge

    def grad_from_gap(g, stick_nodes):
        gA = np.zeros_like(g)
        gA[stick_nodes] = g[stick_nodes]
        return K.T @ gA * sign  # gradient wrt p of 0.5*||g_A||^2

    # --- sign auto-detection -------------------------------------------------
    if use_minus is None:
        g_plus  = gap_plus(p)
        g_minus = gap_minus(p)
        pen_plus  = float(-np.minimum(g_plus,  0.0).sum())
        pen_minus = float(-np.minimum(g_minus, 0.0).sum())
        if pen_plus <= 1e-16 and pen_minus <= 1e-16:
            # no penetration either way; trivial solution
            u = np.zeros_like(g0)
            _log_info("constrained_CG_p0p1: no penetration detected with initial guess; returning zeros.")
            return p*0.0, u, {'eta_disp': 0.0, 'eta_ort': 0.0, 'max_pen': 0.0, 'iters': 0, 'use_minus': False}

        if abs(pen_plus - pen_minus) > 1e-16:
            # prefer the sign that yields smaller penetration for the current guess
            use_minus = (pen_minus < pen_plus)
        else:
            # fallback: look at gradient tendency from the raw gap
            stick0 = (g0 < -neg_gap_tol)
            if np.any(p > 0):
                touched = np.zeros_like(stick0)
                for e, nodes in enumerate(node_of_elem):
                    if p[e] > 0.0:
                        touched[nodes] = True
                stick0 |= touched
            gA = np.zeros_like(g0)
            gA[stick0] = g0[stick0]
            w_plus = K.T @ gA  # assumes sign = +1
            w_minus = -w_plus  # sign = -1 reverses gradient
            pot_plus = float(np.maximum(-w_plus, 0.0).sum())
            pot_minus = float(np.maximum(-w_minus, 0.0).sum())
            use_minus = (pot_minus > pot_plus)
        _log_info(
            "constrained_CG_p0p1: auto-detected sign (pen_plus=%.3e, pen_minus=%.3e, use_minus=%s)",
            pen_plus,
            pen_minus,
            use_minus,
        )

    gap = gap_minus if use_minus else gap_plus
    sign = -1.0 if use_minus else +1.0  # ∂g/∂p = sign * K
    _log_info(f"constrained_CG_p0p1: start (use_minus={use_minus}, tol={tol:.3e}, max_iter={max_iter})")

    # --- initialize ----------------------------------------------------------
    g = gap(p)

    # seed stick nodes from penetration first
    stick_nodes = (g < -neg_gap_tol)

    # also add nodes touched by elements with p>0
    if np.any(p > 0):
        touched = np.zeros(n, dtype=bool)
        for e, nodes in enumerate(node_of_elem):
            if p[e] > 0.0:
                touched[nodes] = True
        stick_nodes |= touched

    # if still empty, use element-averaged penetration
    if not stick_nodes.any():
        ge = elem_avg_gap(g)
        for e, nodes in enumerate(node_of_elem):
            if ge[e] < -neg_gap_tol:
                stick_nodes[nodes] = True

    # warm start if p is zero OR stick_nodes nonempty but gradient is zero
    w = grad_from_gap(g, stick_nodes)
    if np.all(p == 0.0) or (np.linalg.norm(w) == 0 and stick_nodes.any()):
        p = np.maximum(-warm_alpha * w, 0.0)
        g = gap(p)
        if np.any(p > 0):
            touched = np.zeros(n, dtype=bool)
            for e, nodes in enumerate(node_of_elem):
                if p[e] > 0.0:
                    touched[nodes] = True
            stick_nodes |= touched
        w = grad_from_gap(g, stick_nodes)

    t = -w
    w_prev = w.copy()

    # quick exit guard: only if no penetration and p==0
    def diagnostics(p, g):
        ge = elem_avg_gap(g)
        eta_disp = np.linalg.norm(g[stick_nodes]) / (np.linalg.norm(g) + 1e-30) if stick_nodes.any() else 0.0
        denom = (np.linalg.norm(p) * np.linalg.norm(ge)) + 1e-30
        eta_ort  = abs(np.dot(p, ge)) / denom
        max_pen  = float(-g.min()) if g.size else 0.0
        return eta_disp, eta_ort, max_pen

    eta_disp, eta_ort, max_pen = diagnostics(p, g)
    performed_step = False
    iter_count = 0

    for it in range(max_iter):
        iter_count = it + 1
        # project direction onto feasible cone at boundary: if p_i==0, forbid t_i<0
        t_eff = t.copy()
        t_eff[(p <= 0.0) & (t_eff < 0.0)] = 0.0

        # if direction is null, restart from steepest descent and enlarge stick set if needed
        if not np.any(t_eff != 0.0):
            pen_nodes = (g < -neg_gap_tol)
            if pen_nodes.any():
                stick_nodes_before = stick_nodes.copy()
                stick_nodes |= pen_nodes
                if np.any(stick_nodes_before != stick_nodes):
                    w = grad_from_gap(g, stick_nodes)
            t = -w
            t_eff = t.copy()
            t_eff[(p <= 0.0) & (t_eff < 0.0)] = 0.0
            if not np.any(t_eff != 0.0):
                # nothing to do; break to avoid infinite loop
                break

        # quadratic model on active nodes: q = H t with H ≈ K^T P_A K
        Kt = K @ t_eff * sign
        KtA = np.zeros(n)
        KtA[stick_nodes] = Kt[stick_nodes]
        q = K.T @ KtA                                      # (m,)
        q *= sign  # ensure positive-definite Hessian irrespective of use_minus

        denom = float(np.dot(t_eff, q))
        num   = -float(np.dot(w,     t_eff))
        # safe step
        alpha_star = num/denom if denom > 1e-30 else 0.0

        # blocking step to keep p>=0
        neg = t_eff < 0
        alpha_block = np.min(-p[neg] / t_eff[neg]) if np.any(neg) else np.inf

        # choose alpha
        if not np.isfinite(alpha_star) or alpha_star <= 0:
            alpha = alpha_block if np.isfinite(alpha_block) else 0.0
        else:
            alpha = min(alpha_star, alpha_block)

        if alpha <= 0:
            # fallback: pure boundary step if available
            if np.isfinite(alpha_block) and alpha_block > 0:
                alpha = alpha_block
            else:
                # no progress possible; break
                break

        # line search (Armijo-like) to guarantee decrease on current stick nodes
        gA = g[stick_nodes]
        f_curr = 0.5 * float(np.dot(gA, gA))
        alpha_try = alpha
        accepted = False
        for _ in range(12):
            p_trial = np.maximum(p + alpha_try * t_eff, 0.0)
            if not np.any(p_trial != p):
                alpha_try = 0.0
                break
            g_trial = gap(p_trial)
            gA_trial = g_trial[stick_nodes]
            f_trial = 0.5 * float(np.dot(gA_trial, gA_trial))
            if f_trial <= f_curr + 1e-14:
                accepted = True
                break
            alpha_try *= 0.5

        if not accepted or alpha_try <= 0.0:
            # failed to find a productive step
            break

        # take step and project
        p = np.maximum(p + alpha_try * t_eff, 0.0)
        performed_step = True

        # update residual and stick set
        g = g_trial

        stick_prev = stick_nodes.copy()
        stick_nodes[:] = False
        if np.any(p > 0):
            for e, nodes in enumerate(node_of_elem):
                if p[e] > 0.0:
                    stick_nodes[nodes] = True
        stick_nodes |= (g < -neg_gap_tol)

        w_prev = w.copy()
        w = grad_from_gap(g, stick_nodes)

        # CG update
        if np.any(stick_prev != stick_nodes):
            beta = 0.0
        else:
            denom_beta = float(np.dot(w_prev, w_prev)) + 1e-30
            beta = float(np.dot(w, w)) / denom_beta
        t = -w + beta * t

        w_prev = w.copy()

        # diagnostics and stopping
        eta_disp, eta_ort, max_pen = diagnostics(p, g)
        _log_info(
            "iter %04d: eta_disp=%.3e eta_ort=%.3e max_pen=%.3e ||p||=%.3e alpha=%.3e active_elems=%d active_nodes=%d",
            it,
            eta_disp,
            eta_ort,
            max_pen,
            float(np.linalg.norm(p)),
            alpha_try,
            int(np.count_nonzero(p > 0.0)),
            int(np.count_nonzero(stick_nodes)),
        )
        if (eta_disp < tol and eta_ort < tol and max_pen < neg_gap_tol):
            break

    # final outputs
    u = K @ p * (-1.0 if use_minus else +1.0)  # so that g = g0 + u
    _log_info(
        "constrained_CG_p0p1: finish (iters=%d, eta_disp=%.3e, eta_ort=%.3e, max_pen=%.3e, ||p||=%.3e, use_minus=%s)",
        iter_count if performed_step else 0,
        eta_disp,
        eta_ort,
        max_pen,
        float(np.linalg.norm(p)),
        use_minus,
    )
    return p, u, {'eta_disp': eta_disp, 'eta_ort': eta_ort, 'max_pen': max_pen,
                  'iters': (iter_count if performed_step else 0), 'use_minus': use_minus}

"""
   Solver for pressure defined at nodes
"""
# Constrained CG python
def constrained_CG_p1p1(K, error_type, coord, dofs, gap, max_iter, tolerance, pressure_factor=1e12, initial_pressure=None):
    error_history = np.zeros((max_iter,3))
    ub = -gap
    # print(" {0:10s}   {1:10s}   {2:10s}  {3:10s}".format("Iteration", "Error sqrt(R1*R2)", "Displ. Error, R1 ", "Orthogonality, R2"))
    # Warmed start does not work well
    if initial_pressure is not None:        
        # p = initial_pressure
        # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
        # p[gap>0] = 0
        p = np.maximum(-gap, 0) * pressure_factor
    else:
        p = np.zeros_like(ub)
        p = np.maximum(-gap, 0) * pressure_factor

    w = np.inner(K, p) - ub
    # w -= np.mean(w) #new
    t  = w
    t_ = np.zeros_like(w)
    d = 0
    error  = 1
    error_ = 1
    for iter in range(max_iter):
        if iter > 0:
            t[p>0] = w[p>0] + d * error/error_ * t_[p>0]
            t[p<=0] = 0
        q = np.inner(K, t)
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

        w = np.inner(K, p) - ub
        nw = np.linalg.norm(w, 2)

        error_ = error
        displ_error = np.linalg.norm(w[p>0], 2) / nw
        ort = np.abs(np.dot(w,p)/nw) 
        
        if error_type == "displacement":
            error = displ_error
        elif error_type == "mix":
            error = np.sqrt(displ_error * ort)
        elif error_type == "nw":
            error = nw
            if abs((error - error_)/error_) < tolerance:
                error_history[iter,0] = displ_error
                error_history[iter,1] = abs((error - error_)/error_)
                error_history[iter,2] = ort
                return p, np.inner(K,p), error_history[:iter+1]
        error_history[iter,0] = displ_error
        error_history[iter,1] = error
        error_history[iter,2] = ort
        if error < tolerance:
            break
    return p, np.inner(K,p), error_history[:iter+1]

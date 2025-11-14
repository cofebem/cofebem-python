import numpy as np
from .kernels import kelvin_H


def gl_interval(n, a, b):
    """Gauss–Legendre nodes/weights on [a,b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    # map from [-1,1] to [a,b]
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    wm = 0.5 * (b - a) * w
    return xm, wm


def reorder_triangle_for_corner(elem, sing_corner):

    if sing_corner == 0:
        return elem.copy()
    elif sing_corner == 1:
        # (1,2,0)
        return np.vstack([elem[1], elem[2], elem[0]])
    else:  # sing_corner == 2
        # (2,0,1)
        return np.vstack([elem[2], elem[0], elem[1]])


def triangle_edge_vectors(elem_loc0):

    p0, p1, p2 = elem_loc0
    a = p1 - p0
    b = p2 - p0
    cx = np.cross(a, b)
    J_geo = np.linalg.norm(cx)
    if J_geo == 0.0:
        raise ValueError("Degenerate triangle in CPV routine (zero area).")
    n_tri = cx / J_geo
    return a, b, J_geo, n_tri


def A_theta(a, b, theta):

    c, s = np.cos(theta), np.sin(theta)
    Av = a * c + b * s  # 3-vector
    return np.linalg.norm(Av), Av


# ----------------------------- Guiggiani CPV for H on a single P1 triangle -----------------------------
def guiggiani_H_triangle(
    xc,
    elem,
    normal,
    sing_corner,
    loc_idx,
    mu,
    nu,
    n_theta=16,
    n_rho=16,
):
    """
    Direct CPV evaluation of the traction kernel H for a single P1 triangle
    when the collocation is at a triangle vertex (Guiggiani & Gigante, 1990, Eq. (20e)).

    Returns: 3x3 matrix, or None if this is not the singular basis.
    """

    if loc_idx != sing_corner:
        return None

    E = reorder_triangle_for_corner(elem, sing_corner)
    p0, p1, p2 = E[0], E[1], E[2]

    a, b, J_geo, _ = triangle_edge_vectors(E)

    n = normal / np.linalg.norm(normal)

    factor = -1.0 / (8.0 * np.pi * (1.0 - nu))
    I3 = np.eye(3)
    one_minus_2nu = 1.0 - 2.0 * nu

    thetas, wtheta = gl_interval(n_theta, 0.0, 0.5 * np.pi)

    I_1D = np.zeros((3, 3))
    I_2D = np.zeros((3, 3))

    tiny = 1e-14

    for th, wth in zip(thetas, wtheta):
        c, s = np.cos(th), np.sin(th)
        denom = c + s
        if denom < tiny:
            continue  # avoid blowup right at the axes

        A_val, Av = A_theta(a, b, th)
        if A_val < 1e-15:
            continue  # degenerate direction

        rhat = Av / A_val
        n_dot_Av = float(np.dot(n, Av))

        outer_rhat = np.outer(rhat, rhat)
        term_leading = n_dot_Av * (
            one_minus_2nu * I3 + 3.0 * outer_rhat
        ) - one_minus_2nu * (np.outer(Av, n) - np.outer(n, Av))
        f_theta = factor * (J_geo / (A_val**3)) * term_leading  # 3x3

        rho_max = 1.0 / denom

        I_1D += f_theta * np.log(max(tiny, rho_max * A_val)) * wth

        rhos, wrho = gl_interval(n_rho, 0.0, rho_max)
        accum_rho = np.zeros((3, 3))
        for rho, wr in zip(rhos, wrho):
            if rho < tiny:
                continue

            xi = rho * c
            eta = rho * s

            y = map_to_physical_3d(E, xi, eta)

            N_local = 1.0 - xi - eta
            if N_local < 0.0:
                N_local = 0.0

            Hmat = kelvin_H(xc, y, n, mu, nu)  # 3x3

            F = Hmat * (N_local * J_geo * rho)

            accum_rho += (F - (f_theta / rho)) * wr

        I_2D += accum_rho * wth

    return I_1D + I_2D

import meshio
import numpy as np
from tqdm import tqdm
from numba import njit


# ------------------------------------------------read mesh and extract boundary ------------------------------------
def mesh_dim(mesh):
    dim_map = {
        "vertex": 0,
        "line": 1,
        "line2": 1,
        "triangle": 2,
        "triangle6": 2,
        "quad": 2,
        "quad8": 2,
        "tetra": 3,
        "tetra10": 3,
        "hexahedron": 3,
        "hexahedron20": 3,
        "wedge": 3,
        "pyramid": 3,
    }
    dims = [dim_map[cell.type] for cell in mesh.cells]
    return max(dims)


mesh = meshio.read("cube2bem.msh")

cell_type = (
    "triangle"
    if "triangle" in mesh.cells_dict
    else ("quad" if "quad" in mesh.cells_dict else None)
)
if cell_type is None:
    raise RuntimeError("No surface triangle/quad cells found in the mesh.")

F = mesh.cells_dict[cell_type]

unique_ids, inv = np.unique(F.ravel(), return_inverse=True)

boundary_points = mesh.points[unique_ids]
boundary_cells_conn = inv.reshape(F.shape)

boundary_mesh = meshio.Mesh(
    points=boundary_points,
    cells=[(cell_type, boundary_cells_conn)],
)
meshio.write("cube_boundary.vtk", boundary_mesh)

# ------------------------------------material parameters---------------------------------------------

E = 1.0e9
nu = 0.3

lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# -----------------------------------------Kelvin fundamental solutions-------------------------------------


def kelvin_G(x, y, normal, mu, nu):
    x = np.asarray(x)
    y = np.asarray(y)
    r = y - x
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: x and y coincide (r = 0).")

    I = np.eye(3)

    factor = 1.0 / (16 * np.pi * mu * (1 - nu) * r_norm)

    G = factor * ((3 - 4 * nu) * I + np.outer(r, r) / r_norm**2)

    return G


def kelvin_H(x, y, normal, mu, nu):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    normal = np.asarray(normal, dtype=float)

    r = y - x
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: x and y coincide (r = 0).")

    I = np.eye(3)

    factor = -1 / (8 * np.pi * (1 - nu) * r_norm**2)
    term1 = (
        np.dot(r, normal) * ((1 - 2 * nu) * I + 3 * np.outer(r, r) / r_norm**2) / r_norm
    )
    term2 = (1 - 2 * nu) * (np.outer(r, normal) - np.outer(normal, r)) / r_norm

    H = factor * (term1 - term2)

    return H


# ----------------------------------- Shape functions for triangular elements----------------------------------
def shape_functions(xi1, xi2):
    N1 = 1 - xi1 - xi2
    N2 = xi1
    N3 = xi2
    return np.array([N1, N2, N3])


def shape_function_derivatives(xi1, xi2):
    dN1_dxi1 = -1
    dN1_dxi2 = -1
    dN2_dxi1 = 1
    dN2_dxi2 = 0
    dN3_dxi1 = 0
    dN3_dxi2 = 1
    return np.array(
        [
            [dN1_dxi1, dN1_dxi2],
            [dN2_dxi1, dN2_dxi2],
            [dN3_dxi1, dN3_dxi2],
        ]
    )


def map_to_physical_3d(element, xi1, xi2):
    N_vals = shape_functions(xi1, xi2)
    point = np.dot(N_vals, element)
    return point


def jacobian_determinant_3d(element, xi1, xi2):
    dN = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)  # shape (3,2)

    J = element.T @ dN
    return np.linalg.norm(np.cross(J[:, 0], J[:, 1]))


# -----------------------------Numerical integration-----------------------------------
def dunavant_rule(degree: int):
    if degree <= 1:
        pts = np.array([[1 / 3, 1 / 3]], dtype=float)
        w = np.array([0.5], dtype=float)

    elif degree == 2:
        a = 1.0 / 6.0
        b = 2.0 / 3.0
        pts = np.array([[a, a], [b, a], [a, b]], dtype=float)
        w = np.array([1 / 6, 1 / 6, 1 / 6], dtype=float)

    elif degree == 3:
        pts = np.array(
            [[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]], dtype=float
        )
        w = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96], dtype=float)

    elif degree == 4:
        a = 0.445948490915965
        b = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        pts = np.array(
            [
                [a, a],
                [1 - 2 * a, a],
                [a, 1 - 2 * a],
                [b, b],
                [1 - 2 * b, b],
                [b, 1 - 2 * b],
            ],
            dtype=float,
        )
        w = np.array([w1, w1, w1, w2, w2, w2], dtype=float)

    elif degree == 5:
        a = 0.101286507323456
        b = 0.470142064105115
        w1 = 0.0629695902724136
        w2 = 0.066197076394253
        pts = np.array(
            [
                [a, a],
                [1 - 2 * a, a],
                [a, 1 - 2 * a],
                [b, b],
                [1 - 2 * b, b],
                [b, 1 - 2 * b],
                [1 / 3, 1 / 3],
            ],
            dtype=float,
        )
        w = np.array([w1, w1, w1, w2, w2, w2, 0.1125], dtype=float)

    elif degree == 13:
        pts = np.array(
            [
                [0.333333333333333333333333333333, 0.333333333333333333333333333333],
                [0.950275662924105565450352089520, 0.024862168537947217274823955239],
                [0.024862168537947217274823955239, 0.950275662924105565450352089520],
                [0.024862168537947217274823955239, 0.024862168537947217274823955239],
                [0.171614914923835347556304795551, 0.414192542538082326221847602214],
                [0.414192542538082326221847602214, 0.171614914923835347556304795551],
                [0.414192542538082326221847602214, 0.414192542538082326221847602214],
                [0.539412243677190440263092985511, 0.230293878161404779868453507244],
                [0.230293878161404779868453507244, 0.539412243677190440263092985511],
                [0.230293878161404779868453507244, 0.230293878161404779868453507244],
                [0.772160036676532561750285570113, 0.113919981661733719124857214943],
                [0.113919981661733719124857214943, 0.772160036676532561750285570113],
                [0.113919981661733719124857214943, 0.113919981661733719124857214943],
                [0.009085399949835353883572964740, 0.495457300025082323058213517632],
                [0.495457300025082323058213517632, 0.009085399949835353883572964740],
                [0.495457300025082323058213517632, 0.495457300025082323058213517632],
                [0.062277290305886993497083640527, 0.468861354847056503251458179727],
                [0.468861354847056503251458179727, 0.062277290305886993497083640527],
                [0.468861354847056503251458179727, 0.468861354847056503251458179727],
                [0.022076289653624405142446876931, 0.851306504174348550389457672223],
                [0.022076289653624405142446876931, 0.126617206172027096933163647918],
                [0.851306504174348550389457672223, 0.022076289653624405142446876931],
                [0.126617206172027096933163647918, 0.851306504174348550389457672223],
                [0.851306504174348550389457672223, 0.126617206172027096933163647918],
                [0.126617206172027096933163647918, 0.022076289653624405142446876931],
                [0.018620522802520968955913511549, 0.689441970728591295496647976487],
                [0.018620522802520968955913511549, 0.291937506468887771754472382212],
                [0.689441970728591295496647976487, 0.018620522802520968955913511549],
                [0.291937506468887771754472382212, 0.689441970728591295496647976487],
                [0.689441970728591295496647976487, 0.291937506468887771754472382212],
                [0.291937506468887771754472382212, 0.018620522802520968955913511549],
                [0.096506481292159228736516560903, 0.635867859433872768286976979827],
                [0.096506481292159228736516560903, 0.267625659273967961282458816185],
                [0.635867859433872768286976979827, 0.096506481292159228736516560903],
                [0.267625659273967961282458816185, 0.635867859433872768286976979827],
                [0.635867859433872768286976979827, 0.267625659273967961282458816185],
                [0.267625659273967961282458816185, 0.096506481292159228736516560903],
            ]
        )

        w_raw = np.array(
            [
                0.051739766065744133555179145422,
                0.008007799555564801597804123460,
                0.008007799555564801597804123460,
                0.008007799555564801597804123460,
                0.046868898981821644823226732071,
                0.046868898981821644823226732071,
                0.046868898981821644823226732071,
                0.046590940183976487960361770070,
                0.046590940183976487960361770070,
                0.046590940183976487960361770070,
                0.031016943313796381407646220131,
                0.031016943313796381407646220131,
                0.031016943313796381407646220131,
                0.010791612736631273623178240136,
                0.010791612736631273623178240136,
                0.010791612736631273623178240136,
                0.032195534242431618819414482205,
                0.032195534242431618819414482205,
                0.032195534242431618819414482205,
                0.015445834210701583817692900053,
                0.015445834210701583817692900053,
                0.015445834210701583817692900053,
                0.015445834210701583817692900053,
                0.015445834210701583817692900053,
                0.015445834210701583817692900053,
                0.017822989923178661888748319485,
                0.017822989923178661888748319485,
                0.017822989923178661888748319485,
                0.017822989923178661888748319485,
                0.017822989923178661888748319485,
                0.017822989923178661888748319485,
                0.037038683681384627918546472190,
                0.037038683681384627918546472190,
                0.037038683681384627918546472190,
                0.037038683681384627918546472190,
                0.037038683681384627918546472190,
                0.037038683681384627918546472190,
            ]
        )
        w = 0.5 * w_raw

    else:
        return dunavant_rule(5)

    return pts, w


def telles01(s0, t):  # t in [0,1]
    z = 2 * t - 1
    a = 2 * s0 - 1
    s = 0.5 * (z**3 - a * z**2 + a + 1.0)
    ds_dz = 0.5 * (3 * z**2 - 2 * a * z)
    ds_dt = 2 * ds_dz
    return s, ds_dt


def closest_point_on_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (1.0, 0.0, 0.0), np.linalg.norm(p - a)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0), np.linalg.norm(p - b)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        return q, (1.0 - v, v, 0.0), np.linalg.norm(p - q)

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0), np.linalg.norm(p - c)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        return q, (1.0 - w, 0.0, w), np.linalg.norm(p - q)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + w * (c - b)
        return q, (0.0, 1.0 - w, w), np.linalg.norm(p - q)

    # Inside face region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    q = u * a + v * b + w * c
    return q, (u, v, w), np.linalg.norm(p - q)


def duffy_map_corner(corner, u, v):

    if corner == 0:
        xi1 = u * (1 - v)
        xi2 = u * v
        J = u
    elif corner == 1:
        xi1 = 1.0 - u
        xi2 = u * v
        J = u
    else:  # corner == 2
        xi1 = u * v
        xi2 = 1.0 - u
        J = u
    return xi1, xi2, J


gauss_pts, gauss_wts = np.polynomial.legendre.leggauss(3)

degree = 5
quad_pts, quad_wts = dunavant_rule(degree)


def legendre_on_01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


GL_N = 10
t_1d, w_1d = legendre_on_01(GL_N)


# ----------------------------- Strong Sing quadrature -----------------------------
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


def integrate_G_const(
    kernel, xc, elem, normal, reg, xi_eta_star=None, sing_corner=None
):
    """
    Integrate kernel(xc, y, normal, ...) over a single triangle with
    a P0 (constant) basis function, i.e. N_t ≡ 1 on the element.

    Returns a 3x3 matrix for one (collocation, element) pair.
    """
    out = np.zeros((3, 3))

    match reg:
        case "reg":
            for (xi1, xi2), w in zip(quad_pts, quad_wts):
                y = map_to_physical_3d(elem, xi1, xi2)
                J_geo = jacobian_determinant_3d(elem, xi1, xi2)
                out += kernel(xc, y, normal, mu, nu) * (w * J_geo)

        case "near_sing":
            xi_star, eta_star = xi_eta_star
            xi_star = min(xi_star, 1.0 - 1e-12)
            u_star = float(np.clip(xi_star, 0.0, 1.0))
            v_star = float(np.clip(eta_star / max(1e-12, (1.0 - xi_star)), 0.0, 1.0))

            for t, wt in zip(t_1d, w_1d):
                u, du_dt = telles01(u_star, t)
                for s, ws in zip(t_1d, w_1d):
                    v, dv_ds = telles01(v_star, s)
                    xi, eta = u, (1.0 - u) * v
                    J_geo = jacobian_determinant_3d(elem, xi, eta)
                    J_map = (1.0 - u) * du_dt * dv_ds
                    y = map_to_physical_3d(elem, xi, eta)
                    out += kernel(xc, y, normal, mu, nu) * (wt * ws * J_map * J_geo)

        case "sing":
            assert sing_corner in (0, 1, 2), "sing_corner must be 0,1,or 2"

            for rho, wr in zip(t_1d, w_1d):
                for tau, wt in zip(t_1d, w_1d):
                    xi, eta, J_duffy = duffy_map_corner(sing_corner, rho, tau)
                    y = map_to_physical_3d(elem, xi, eta)
                    J_geo = jacobian_determinant_3d(elem, xi, eta)
                    w_total = wr * wt * J_duffy * J_geo
                    out += kernel(xc, y, normal, mu, nu) * w_total

    return out


def integrate(
    kernel, xc, elem, normal, loc_idx, reg, xi_eta_star=None, sing_corner=None
):
    out = np.zeros((3, 3))
    match reg:
        case "reg":
            # Dunavant rule
            for (xi1, xi2), w in zip(quad_pts, quad_wts):
                N_vals = shape_functions(xi1, xi2)
                N_local = N_vals[loc_idx]
                y = map_to_physical_3d(elem, xi1, xi2)
                J_geo = jacobian_determinant_3d(elem, xi1, xi2)
                out += kernel(xc, y, normal, mu, nu) * (N_local * w * J_geo)

        case "near_sing":
            xi_star, eta_star = xi_eta_star
            xi_star = min(xi_star, 1.0 - 1e-12)
            u_star = float(np.clip(xi_star, 0.0, 1.0))
            v_star = float(np.clip(eta_star / max(1e-12, (1.0 - xi_star)), 0.0, 1.0))

            for t, wt in zip(t_1d, w_1d):  # [0,1]
                u, du_dt = telles01(u_star, t)
                for s, ws in zip(t_1d, w_1d):  # [0,1]
                    v, dv_ds = telles01(v_star, s)
                    xi, eta = u, (1.0 - u) * v
                    J_geo = jacobian_determinant_3d(elem, xi, eta)  # P1: constant
                    J_map = (1.0 - u) * du_dt * dv_ds  # collapse * Telles_u * Telles_v
                    N_local = shape_functions(xi, eta)[loc_idx]
                    y = map_to_physical_3d(elem, xi, eta)
                    out += kernel(xc, y, normal, mu, nu) * (
                        N_local * wt * ws * J_map * J_geo
                    )

        case "sing":
            assert sing_corner in (0, 1, 2), "sing_corner must be 0,1,or 2"
            if kernel == kelvin_H and loc_idx == sing_corner:
                res = guiggiani_H_triangle(
                    xc=xc,
                    elem=elem,
                    normal=normal,
                    sing_corner=sing_corner,
                    loc_idx=loc_idx,
                    mu=mu,
                    nu=nu,
                    n_theta=10,
                    n_rho=10,
                )
                # If CPV handled, return directly
                if res is not None:
                    return res

            for rho, wr in zip(t_1d, w_1d):
                for tau, wt in zip(t_1d, w_1d):
                    xi, eta, J_duffy = duffy_map_corner(sing_corner, rho, tau)
                    N_vals = shape_functions(xi, eta)
                    y = map_to_physical_3d(elem, xi, eta)
                    J_geo = jacobian_determinant_3d(elem, xi, eta)
                    w_total = wr * wt * J_duffy * J_geo
                    out += kernel(xc, y, normal, mu, nu) * (N_vals[loc_idx] * w_total)

    return out


# # -----------------------------------Collocation BEM---------------------------------------------------------
# # -----------------------------------Assembling H and G------------------------------------------------------
tdim = 3
fdim = 2


def tri_normal(p1, p2, p3):
    a = np.cross(p2 - p1, p3 - p1)
    nrm = np.linalg.norm(a)
    if nrm == 0.0:
        raise ValueError("Degenerate triangle (zero area).")
    return a / nrm


mesh_center = boundary_points.mean(axis=0)
nElem = boundary_cells_conn.shape[0]

elem_nodes = np.empty((nElem, 3, 3), dtype=np.float64)
elem_normals = np.empty((nElem, 3), dtype=np.float64)
elem_h = np.empty(nElem, dtype=np.float64)

for e, tri in enumerate(boundary_cells_conn):
    p1, p2, p3 = (
        boundary_points[tri[0]],
        boundary_points[tri[1]],
        boundary_points[tri[2]],
    )
    elem_nodes[e, :, :] = np.vstack([p1, p2, p3])
    n = tri_normal(p1, p2, p3)
    c = (p1 + p2 + p3) / 3.0
    if np.dot(n, c - mesh_center) < 0:
        print("Problema")
        n = -n
    elem_normals[e, :] = n

    la = np.linalg.norm(p2 - p1)
    lb = np.linalg.norm(p3 - p2)
    lc = np.linalg.norm(p1 - p3)
    elem_h[e] = (la + lb + lc) / 3.0

NEAR_FACTOR = 0.5
near_thresh = NEAR_FACTOR * elem_h

n_collocs = len(boundary_points)


# ---------- Free-term c_i = omega_i/(4π) for each boundary node ----------
corners = []
edges = []
regs = []


def count_on_planes(p, tol=1e-9):
    hits = 0
    for k in range(3):
        if abs(p[k] - 0.0) <= tol or abs(p[k] - 1.0) <= tol:
            hits += 1
    return hits


c_vals = np.empty(n_collocs, dtype=float)
tol_plane = 1e-9
for i, x in enumerate(boundary_points):
    hits = count_on_planes(x, tol_plane)
    if hits == 1:  # face interior
        c_vals[i] = 0.5
        regs.append(i)
    elif hits == 2:  # edge
        c_vals[i] = 0.25
        edges.append(i)
    elif hits == 3:  # corner
        c_vals[i] = 0.125
        corners.append(i)
    else:
        c_vals[i] = 0.5


#################################################################################################

# G = np.zeros((tdim * n_collocs, tdim * nElem))
# H = np.zeros((tdim * n_collocs, tdim * n_collocs))

# for i, xc in tqdm(
#     enumerate(boundary_points),
#     total=n_collocs,
#     desc="Assembling global matrices (P0 traction / P1 disp)",
# ):

#     for e, elem_conn in enumerate(boundary_cells_conn):
#         elem = elem_nodes[e]
#         normal = elem_normals[e]

#         q, (lam1, lam2, lam3), dist = closest_point_on_triangle(
#             xc, elem[0], elem[1], elem[2]
#         )
#         xi_star, eta_star = lam2, lam3

#         on_element = i in elem_conn
#         near_sing = (not on_element) and (dist < near_thresh[e])

#         if on_element:
#             sing_corner = int(np.where(elem_conn == i)[0][0])
#         else:
#             sing_corner = None

#         if on_element:
#             reg_flag_G, xi_eta_G = "sing", None
#         elif near_sing:
#             reg_flag_G, xi_eta_G = "near_sing", (xi_star, eta_star)
#         else:
#             reg_flag_G, xi_eta_G = "reg", None

#         Ge = integrate_G_const(
#             kelvin_G,
#             xc,
#             elem,
#             normal,
#             reg_flag_G,
#             xi_eta_star=xi_eta_G,
#             sing_corner=sing_corner,
#         )

#         G[tdim * i : tdim * (i + 1), tdim * e : tdim * (e + 1)] += Ge

#         for loc_idx, j in enumerate(elem_conn):
#             if on_element:
#                 reg_flag_H, xi_eta_H = "sing", None
#             elif near_sing:
#                 reg_flag_H, xi_eta_H = "near_sing", (xi_star, eta_star)
#             else:
#                 reg_flag_H, xi_eta_H = "reg", None

#             Hij = integrate(
#                 kelvin_H,
#                 xc,
#                 elem,
#                 normal,
#                 loc_idx,
#                 reg_flag_H,
#                 xi_eta_H,
#                 sing_corner,
#             )

#             H[tdim * i : tdim * (i + 1), tdim * j : tdim * (j + 1)] += Hij

#     H[tdim * i : tdim * (i + 1), tdim * i : tdim * (i + 1)] += c_vals[i] * np.eye(tdim)


# print("Global Matrices G and H assembled")

# np.savez(
#     "GH_cube_P0P1.npz",
#     G=G,
#     H=H,
# )

# -------------------- Mixed BCs: Dirichlet on z=0, Neumann on a patch at (0.5,0.5,1) --------------------

data = np.load("GH_cube_P0P1.npz")

G, H = data["G"], data["H"]


#############################################################################################
# -------------------- Component-wise mixed BCs (P1 disp, P0 tractions) --------------------
#############################################################################################
# Dirichlet: ux = 0 on x=0 plane
#            uy = 0 on y=0 plane
#            uz = 0 on z=0 plane
# Neumann:   tz = -1e8 on z = 1  (all other tractions = 0)
#############################################################################################

tdim = 3
N_nodes = n_collocs
N_elems = nElem
pts = boundary_points
tol = 1e-9


# Helpers
def nodes2dofs_for_comp(nodes, comp):  # comp: 0=x, 1=y, 2=z
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    return tdim * nodes + int(comp)


def elems2dofs_for_comp(elems, comp):  # P0 tractions: DOFs per element
    elems = np.asarray(elems, dtype=np.int64).ravel()
    return tdim * elems + int(comp)


# --------------------------- Displacement DOFs (P1, nodal) -------------------------------
n_u = tdim * N_nodes
u_known_full = np.zeros(n_u, dtype=float)
is_u_known = np.zeros(n_u, dtype=bool)

# plane selectors for nodes
on_x0 = np.isclose(pts[:, 0], 0.0, atol=tol)
on_y0 = np.isclose(pts[:, 1], 0.0, atol=tol)
on_z0 = np.isclose(pts[:, 2], 0.0, atol=tol)

Iux_nodes = np.where(on_x0)[0]
Iuy_nodes = np.where(on_y0)[0]
Iuz_nodes = np.where(on_z0)[0]

Iu_x = nodes2dofs_for_comp(Iux_nodes, 0)  # ux on x=0
Iu_y = nodes2dofs_for_comp(Iuy_nodes, 1)  # uy on y=0
Iu_z = nodes2dofs_for_comp(Iuz_nodes, 2)  # uz on z=0

# all Dirichlet values are 0 here
is_u_known[Iu_x] = True
is_u_known[Iu_y] = True
is_u_known[Iu_z] = True

Iu_known = np.where(is_u_known)[0]
Iu_unknown = np.where(~is_u_known)[0]

# --------------------------- Traction DOFs (P0, per element) ----------------------------
n_t = tdim * N_elems
t_known_full = np.zeros(n_t, dtype=float)
is_t_unknown = np.zeros(n_t, dtype=bool)

elem_centers = elem_nodes.mean(axis=1)  # shape (N_elems, 3)

# plane selectors for elements (by centroid)
on_x0_elem = np.isclose(elem_centers[:, 0], 0.0, atol=tol)
on_y0_elem = np.isclose(elem_centers[:, 1], 0.0, atol=tol)
on_z0_elem = np.isclose(elem_centers[:, 2], 0.0, atol=tol)
on_z1_elem = np.isclose(elem_centers[:, 2], 1.0, atol=tol)  # top face

elem_x0 = np.where(on_x0_elem)[0]
elem_y0 = np.where(on_y0_elem)[0]
elem_z0 = np.where(on_z0_elem)[0]
elem_top = np.where(on_z1_elem)[0]

# Unknown tractions (analogous to P1/P1 logic):
#   tx unknown on x=0
#   ty unknown on y=0
#   tz unknown on z=0
It_unknown_x = elems2dofs_for_comp(elem_x0, 0)
It_unknown_y = elems2dofs_for_comp(elem_y0, 1)
It_unknown_z = elems2dofs_for_comp(elem_z0, 2)

It_unknown = np.unique(np.concatenate([It_unknown_x, It_unknown_y, It_unknown_z]))
is_t_unknown[It_unknown] = True

It_known = np.where(~is_t_unknown)[0]  # Neumann DOFs

# Neumann tractions:
# default = 0, EXCEPT tz = -1e8 on top (z=1)
t_known_full[:] = 0.0
tz_top_dofs = elems2dofs_for_comp(elem_top, 2)
# make sure they are actually in the known set (they should be, since top is not Dirichlet)
t_known_full[tz_top_dofs] = -1.0e8

# --------------------------- Build global linear system -------------------------------
# Equation: H u = G t
# Decompose: u = u_known_full + u_unknown, t = t_known_full + t_unknown
#
# => H u_unknown - G t_unknown = G t_known_full - H u_known_full  =: rhs_full
#
# Unknown vector x = [u_unknown; t_unknown]
# A = [ H[:, Iu_unknown]  ,  -G[:, It_unknown] ]

rhs_full = G @ t_known_full - H @ u_known_full

H_u = H[:, Iu_unknown]  # (3*N_nodes) × (#u_unknown)
G_t = G[:, It_unknown]  # (3*N_nodes) × (#t_unknown)
A = np.hstack([H_u, -G_t])  # (3*N_nodes) × (#u_unknown + #t_unknown)

# Solve in least squares sense (rectangular system allowed)
x, residuals, rank, svals = np.linalg.lstsq(A, rhs_full, rcond=None)

n_u_unknown = len(Iu_unknown)
u_unknown = x[:n_u_unknown]
t_unknown = x[n_u_unknown:]

# --------------------------- Reconstruct full DOF vectors -----------------------------
u_dofs = np.zeros(n_u, dtype=float)
t_dofs = np.zeros(n_t, dtype=float)

u_dofs[Iu_known] = u_known_full[Iu_known]
u_dofs[Iu_unknown] = u_unknown

t_dofs[It_known] = t_known_full[It_known]
t_dofs[It_unknown] = t_unknown

U = u_dofs.reshape((N_nodes, tdim))  # nodal displacements (P1)
T_elem = t_dofs.reshape((N_elems, tdim))  # element tractions (P0)

# --------------------------- Map P0 tractions to nodes for visualization ---------------
T_nodes = np.zeros((N_nodes, tdim), dtype=float)
count = np.zeros(N_nodes, dtype=int)

for e, conn in enumerate(boundary_cells_conn):
    for node in conn:
        T_nodes[node] += T_elem[e]
        count[node] += 1

nonzero = count > 0
T_nodes[nonzero] /= count[nonzero][:, None]

# --------------------------- BC mask per node -----------------------------------------
bc_mask = np.zeros(N_nodes, dtype=np.int32)

# Mark Dirichlet nodes
bc_mask[Iux_nodes] = 1  # ux=0
bc_mask[Iuy_nodes] = 1  # uy=0
bc_mask[Iuz_nodes] = 1  # uz=0

# Mark Neumann tz on top elements
nodes_top_neumann = np.unique(boundary_cells_conn[elem_top].ravel())
bc_mask[nodes_top_neumann] = 10

# --------------------------- Write VTK -------------------------------------------------
boundary_mesh.point_data["u"] = U
boundary_mesh.point_data["t_avg"] = T_nodes
boundary_mesh.cell_data["t_p0"] = [T_elem]
boundary_mesh.point_data["bc_mask"] = bc_mask

meshio.write("cube2bem_p0p1.vtu", boundary_mesh)
print(
    "Wrote cube2bem_p0p1.vtu with P0 tractions / P1 displacements and component-wise BCs"
)

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


mesh = meshio.read("fine2_sphere.msh")

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

# meshio.write("sphere_boundary1.vtk", boundary_mesh)


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


# -----------------------------------Collocation BEM---------------------------------------------------------
# -----------------------------------Assembling H and G------------------------------------------------------
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


# ================== FREE-TERM (c = 0.5) NUMERICAL CHECK ==================


def guiggiani_H_triangle_constN(
    xc, elem, normal, sing_corner, mu, nu, n_theta=16, n_rho=16
):
    E = reorder_triangle_for_corner(elem, sing_corner)
    a, b, J_geo, _ = triangle_edge_vectors(E)
    n = normal / np.linalg.norm(normal)

    I3 = np.eye(3)
    one_m_2nu = 1.0 - 2.0 * nu
    factor = -1.0 / (8.0 * np.pi * (1.0 - nu))  # matches your H convention

    thetas, wtheta = gl_interval(n_theta, 0.0, 0.5 * np.pi)
    C = np.zeros((3, 3))
    tiny = 1e-14

    for th, wth in zip(thetas, wtheta):
        c, s = np.cos(th), np.sin(th)
        denom = c + s
        if denom < tiny:
            continue

        A_val, Av = A_theta(a, b, th)
        if A_val < 1e-15:
            continue

        rhat = Av / A_val
        n_dot_Av = float(np.dot(n, Av))
        # leading angular tensor f_theta (same as in your Guiggiani code)
        term_leading = n_dot_Av * (
            one_m_2nu * I3 + 3.0 * np.outer(rhat, rhat)
        ) - one_m_2nu * (np.outer(Av, n) - np.outer(n, Av))
        f_theta = factor * (J_geo / (A_val**3)) * term_leading  # 3x3

        rho_max = 1.0 / (denom)

        # 1D analytic contribution (log term)
        C += f_theta * np.log(max(tiny, rho_max * A_val)) * wth

        # 2D regular remainder
        rhos, wrho = gl_interval(n_rho, 0.0, rho_max)
        accum_rho = np.zeros((3, 3))
        for rho, wr in zip(rhos, wrho):
            if rho < tiny:
                continue
            xi, eta = rho * c, rho * s
            y = map_to_physical_3d(E, xi, eta)
            Hmat = kelvin_H(xc, y, n, mu, nu)  # 3x3
            # N ≡ 1 here, Jacobian ~ J_geo * rho in polar collapse
            F = Hmat * (J_geo * rho)
            accum_rho += (F - (f_theta / rho)) * wr

        C += accum_rho * wth

    return C


def free_term_matrix_at_point(i):
    xc = boundary_points[i]
    C = np.zeros((3, 3))

    for e, elem_conn in enumerate(boundary_cells_conn):
        elem = elem_nodes[e]
        normal = elem_normals[e]

        if i in elem_conn:
            sing_corner = int(np.where(elem_conn == i)[0][0])  # 0,1,2
            C += guiggiani_H_triangle_constN(
                xc, elem, normal, sing_corner, mu, nu, n_theta=20, n_rho=20
            )
        else:
            # regular element: standard Dunavant quadrature
            for (xi1, xi2), w in zip(quad_pts, quad_wts):
                y = map_to_physical_3d(elem, xi1, xi2)
                J_geo = jacobian_determinant_3d(elem, xi1, xi2)
                C += kelvin_H(xc, y, normal, mu, nu) * (w * J_geo)

    return -C


test_indices = [0, len(boundary_points) // 3, 2 * len(boundary_points) // 3]
vals = []
print("\nChecking free-term c at selected collocation points:")
for i in test_indices:
    C = free_term_matrix_at_point(i)
    c_i = np.trace(C) / 3.0
    anis = np.linalg.norm(C - c_i * np.eye(3)) / np.linalg.norm(np.eye(3))
    vals.append(c_i)
    print(f"  i={i:6d}:  c_i ≈ {c_i:.6f}   isotropy error={anis:.2e}")

mean_val = np.mean(vals)
rel_error = np.abs((np.mean(vals) - 0.5) / 0.5)
print(
    f"\nMean c over {len(test_indices)} samples: {mean_val:.6f} (should be ~0.5) \n relative error = {rel_error*100:.2f} %"
)
#################################################################################################
NEAR_FACTOR = 1.5
near_thresh = NEAR_FACTOR * elem_h

n_collocs = len(boundary_points)

G = np.zeros((tdim * n_collocs, tdim * n_collocs))
H = np.zeros((tdim * n_collocs, tdim * n_collocs))

for i, xc in tqdm(
    enumerate(boundary_points),
    total=n_collocs,
    desc="Assembling global matrices",
):

    for e, elem_conn in enumerate(boundary_cells_conn):
        elem = elem_nodes[e]
        normal = elem_normals[e]

        q, (lam1, lam2, lam3), dist = closest_point_on_triangle(
            xc, elem[0], elem[1], elem[2]
        )
        xi_star, eta_star = lam2, lam3

        on_element = i in elem_conn
        near_sing = (not on_element) and (dist < near_thresh[e])

        if on_element:
            sing_corner = int(np.where(elem_conn == i)[0][0])  # 0,1, or 2
        else:
            sing_corner = None

        for loc_idx, j in enumerate(elem_conn):
            if on_element:
                reg_flag, xi_eta_star = "sing", None
            elif near_sing:
                reg_flag, xi_eta_star = "near_sing", (xi_star, eta_star)
            else:
                reg_flag, xi_eta_star = "reg", None

            Gij = integrate(
                kelvin_G, xc, elem, normal, loc_idx, reg_flag, xi_eta_star, sing_corner
            )
            Hij = integrate(
                kelvin_H, xc, elem, normal, loc_idx, reg_flag, xi_eta_star, sing_corner
            )

            G[tdim * i : tdim * (i + 1), tdim * j : tdim * (j + 1)] += Gij
            H[tdim * i : tdim * (i + 1), tdim * j : tdim * (j + 1)] += Hij

    H[tdim * i : tdim * (i + 1), tdim * i : tdim * (i + 1)] += 0.5 * np.eye(tdim)

print("Global Matrices G and H assembled")

np.savez(
    "GH_sphere.npz",
    G=G,
    H=H,
)

# -------------------- Apply uniform pressure over the sphere and solve --------------------

# data = np.load("GH_sphere.npz")

# G, H = data["G"], data["H"]

p = 1.0e9  # uniform external pressure (Pa)

N = n_collocs
dofs = 3 * N

node_norm = np.zeros((N, 3), dtype=float)
A_e = 0.5 * np.linalg.norm(
    np.cross(
        elem_nodes[:, 1, :] - elem_nodes[:, 0, :],
        elem_nodes[:, 2, :] - elem_nodes[:, 0, :],
    ),
    axis=1,
)
for e, tri in enumerate(boundary_cells_conn):
    for k in range(3):
        node_norm[tri[k]] += elem_normals[e] * A_e[e]

# normalize
nnorm = np.linalg.norm(node_norm, axis=1)
nnorm[nnorm == 0.0] = 1.0
node_norm /= nnorm[:, None]


t_nodes = -p * node_norm  # np.full_like(boundary_points, np.array([1, 0, 0]))  #(N,3)
t = t_nodes.reshape(-1)  # (3N,)

rhs = G @ t  # H u = rhs
u = np.linalg.inv(H) @ rhs
# # u = u.reshape(N, 3)

# i0 = int(np.argmax(boundary_points[:, 0]))  # max x
# i1 = int(np.argmax(boundary_points[:, 1]))  # max y
# if i1 == i0:
#     i1 = int(np.argmin(boundary_points[:, 1]))
# i2 = int(np.argmax(boundary_points[:, 2]))  # max z
# if i2 in (i0, i1):
#     i2 = int(np.argmin(boundary_points[:, 2]))

# # Constraints:
# #  - node i0: fix ux, uy, uz (3)
# #  - node i1: fix uy, uz     (2)
# #  - node i2: fix ux         (1)
# B = np.zeros((6, dofs), dtype=float)
# rows = [
#     (0, 3 * i0 + 0),  # i0 ux = 0
#     (1, 3 * i0 + 1),  # i0 uy = 0
#     (2, 3 * i0 + 2),  # i0 uz = 0
#     (3, 3 * i1 + 1),  # i1 uy = 0
#     (4, 3 * i1 + 2),  # i1 uz = 0
#     (5, 3 * i2 + 0),  # i2 ux = 0
# ]
# for r, c in rows:
#     B[r, c] = 1.0

# # KKT system
# K = np.block([[H, B.T], [B, np.zeros((B.shape[0], B.shape[0]))]])
# f = np.zeros(K.shape[0], dtype=float)
# f[:dofs] = rhs

# sol = np.linalg.solve(K, f)
# u = sol[:dofs]  # (3N,)
u_nodes = u.reshape(N, 3)  # (N,3)

r = np.linalg.norm(boundary_points, axis=1)
n_dir = np.divide(
    boundary_points,
    r[:, None],
    out=np.zeros_like(boundary_points),
    where=(r[:, None] > 0),
)
u_rad = np.einsum("ij,ij->i", u_nodes, n_dir)
R = float(r.mean())
u_th = -(1.0 - 2.0 * nu) / E * p * R

u_th_vec = u_th * n_dir
u_th_vec = u_th_vec.reshape(-1)
lhs = H @ u_th_vec
rel_err_th = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
print(f"Relative error | (Hu_th - Gt)/ Gt | = {rel_err_th} (~{100*rel_err_th:.2f}%)")

print(f"Analytic u_r(R) = {u_th:.6e} m  (R ≈ {R:.6e} m)")
print(f"Numerical mean(u_r) = {u_rad.mean():.6e} m")
err_L2 = np.sqrt(np.mean((u_rad - u_th) ** 2))
rel_err = err_L2 / max(1e-16, abs(u_th))
print(f"L2 error on u_r over boundary nodes = {err_L2:.6e}  (~{100*rel_err:.2f}%)")


rhs = rhs.reshape((N, 3))

# pressure case: t = -p*node_norm
n = node_norm
g_n = np.einsum("ij,ij->i", rhs, n)
g_tau = rhs - g_n[:, None] * n
print(
    "median |g_tau|/|g_n| =",
    np.median(np.linalg.norm(g_tau, axis=1) / np.maximum(1e-16, np.abs(g_n))),
)
print("std(g_n)/|mean(g_n)| =", np.std(g_n) / max(1e-16, abs(np.mean(g_n))))
print("sign(mean(g_n))      =", np.sign(np.mean(g_n)))  # should be negative (inward)

boundary_mesh.point_data["u"] = u_nodes
boundary_mesh.point_data["normal"] = node_norm
boundary_mesh.point_data["Gt"] = rhs
meshio.write("sphere_disp_coarse.vtk", boundary_mesh)
print("Output file written")

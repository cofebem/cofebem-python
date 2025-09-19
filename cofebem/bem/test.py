import meshio
import numpy as np
from tqdm import tqdm


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


mesh = meshio.read("fine_sphere.msh")

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


def tri_normal(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    return n / np.linalg.norm(n)


center = boundary_points.mean(axis=0)

nElem = boundary_cells_conn.shape[0]

elem_nodes = np.empty((nElem, 3, 3), dtype=np.float64)
elem_normals = np.empty((nElem, 3), dtype=np.float64)

for e, tri in enumerate(boundary_cells_conn):
    p1, p2, p3 = (
        boundary_points[tri[0]],
        boundary_points[tri[1]],
        boundary_points[tri[2]],
    )
    elem_nodes[e, :, :] = np.vstack([p1, p2, p3])
    n = tri_normal(p1, p2, p3)
    elem_normals[e, :] = n


# boundary_mesh.cell_data["normal"] = elem_normals
# meshio.write("sphere_elem_normals.vtk", boundary_mesh)
# print("DONE")


def kelvin_G(x, y, mu, nu):
    x = np.asarray(x)
    y = np.asarray(y)
    r = x - y
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: x and y coincide (r = 0).")

    I = np.eye(3)

    factor = 1.0 / (16 * np.pi * mu * (1 - nu) * r_norm)

    G = factor * ((3 - 4 * nu) * I + np.outer(r, r) / r_norm**2)

    return G


def rand_rotation(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure det(Q)=+1
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_axis_aligned(mu=2.0, nu=0.25):
    r = 1.0
    x = np.array([r, 0, 0.0])
    y = np.zeros(3)
    G = kelvin_G(x, y, mu, nu)
    fac = 1.0 / (16 * np.pi * mu * (1 - nu) * r)
    expected = fac * np.diag([4 - 4 * nu, 3 - 4 * nu, 3 - 4 * nu])
    err = np.linalg.norm(G - expected, ord=np.inf)
    return err


def test_symmetry(mu=2.0, nu=0.33):
    x = np.array([0.3, -0.2, 0.5])
    y = np.array([-0.4, 0.1, -0.3])
    Gxy = kelvin_G(x, y, mu, nu)
    Gyx = kelvin_G(y, x, mu, nu)
    err = np.linalg.norm(Gxy - Gyx.T, ord=np.inf)
    return err


def test_scaling(mu=1.0, nu=0.25):
    y = np.zeros(3)
    rhat = np.array([1.0, 1.0, 0.0])
    rhat /= np.linalg.norm(rhat)
    r1 = 0.37 * rhat
    r2 = 0.91 * rhat
    G1 = kelvin_G(r1, y, mu, nu)
    G2 = kelvin_G(r2, y, mu, nu)
    # Expect alpha*G(alpha r) == G(r) with same direction
    alpha = r2[0] / r1[0]  # same on all components since collinear
    err = np.linalg.norm(alpha * G2 - G1, ord=np.inf)
    return err


def test_trace(mu=3.0, nu=0.27):
    x = np.array([0.2, 0.7, -0.4])
    y = np.array([0.0, 0.0, 0.0])
    r = np.linalg.norm(x - y)
    G = kelvin_G(x, y, mu, nu)
    lhs = np.trace(G)
    rhs = (10 - 12 * nu) / (16 * np.pi * mu * (1 - nu) * r)
    return abs(lhs - rhs)


def test_rotation_equiv(mu=1.0, nu=0.33):
    x = np.array([0.4, -0.1, 0.2])
    y = np.array([0.05, 0.2, -0.3])
    Q = rand_rotation(seed=42)
    x2, y2 = Q @ x, Q @ y
    G1 = kelvin_G(x, y, mu, nu)
    G2 = kelvin_G(x2, y2, mu, nu)
    pred = Q @ G1 @ Q.T
    err = np.linalg.norm(G2 - pred, ord=np.inf)
    return err


def test_navier_residual(mu=2.0, nu=0.28, h=1e-3):
    # Check: mu ∆G + (λ+μ) ∇(∇·G) = 0 for x≠y
    lam = 2 * mu * nu / (1 - 2 * nu)
    y = np.array([0.0, 0.0, 0.0])
    x0 = np.array([0.31, -0.27, 0.44])  # away from source

    # Helper to sample G at shifted points
    def G(x):
        return kelvin_G(x, y, mu, nu)

    # Laplacian of G (componentwise)
    # ∆G_ij = sum_a d2/dx_a^2 G_ij
    lap = np.zeros((3, 3))
    for a in range(3):
        ea = np.eye(3)[a]
        Gp = G(x0 + h * ea)
        G0 = G(x0)
        Gm = G(x0 - h * ea)
        lap += (Gp - 2 * G0 + Gm) / (h * h)

    # divergence: (∇·G)_j = ∂_k G_{kj}
    div = np.zeros(3)
    for k in range(3):
        ek = np.eye(3)[k]
        Gp = G(x0 + h * ek)
        Gm = G(x0 - h * ek)
        dGdk = (Gp - Gm) / (2 * h)  # d/dx_k G_{ij}
        div += dGdk[k, :]  # sum over k: pick row k

    # gradient of divergence: [∇(∇·G)]_ij = ∂_i (∇·G)_j
    grad_div = np.zeros((3, 3))
    for i in range(3):
        ei = np.eye(3)[i]
        # recompute div at x0 ± h e_i
        div_p = np.zeros(3)
        div_m = np.zeros(3)
        for k in range(3):
            ek = np.eye(3)[k]
            Gp = G(x0 + h * ei + h * ek)
            G0p = G(x0 + h * ei)
            Gm = G(x0 + h * ei - h * ek)
            dGdk_p = (Gp - Gm) / (2 * h)
            div_p += dGdk_p[k, :]
        for k in range(3):
            ek = np.eye(3)[k]
            Gp = G(x0 - h * ei + h * ek)
            Gm = G(x0 - h * ei - h * ek)
            dGdk_m = (Gp - Gm) / (2 * h)
            div_m += dGdk_m[k, :]
        grad_div[i, :] = (div_p - div_m) / (2 * h)

    residual = mu * lap + (lam + mu) * grad_div
    return np.linalg.norm(residual, ord=np.inf)


tol = 1e-9
# print("axis_aligned err:", test_axis_aligned())
# print("symmetry err    :", test_symmetry())
# print("scaling err     :", test_scaling())
# print("trace err       :", test_trace())
# print("rotation err    :", test_rotation_equiv())
# print("Navier residual :", test_navier_residual())


# ---------------- Triangle quadrature (reference triangle: xi>=0, eta>=0, xi+eta<=1) ----------------
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

    else:
        # fallback to degree 5
        return dunavant_rule(5)

    return pts, w


# ---------------- Geometry helpers ----------------
def tri_cross_mag(P1, P2, P3):
    """|| (P2-P1) x (P3-P1) ||"""
    return np.linalg.norm(np.cross(P2 - P1, P3 - P1), axis=-1)


def map_to_triangle(P1, P2, P3, xi_eta):
    """Affine map from reference (xi,eta) to 3D triangle."""
    xi, eta = xi_eta[..., 0], xi_eta[..., 1]
    return (
        P1[None, :]
        + xi[:, None] * (P2 - P1)[None, :]
        + eta[:, None] * (P3 - P1)[None, :]
    )


def jacobian_determinant_3d(element, xi1, xi2):
    dN = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)  # shape (3,2)

    J = element.T @ dN
    return np.linalg.norm(np.cross(J[:, 0], J[:, 1]))


# ---------------- Integrate 1 over all triangles via quadrature ----------------
def integrate_one_over_mesh(points, triangles, quad_degree=5):
    pts_ref, wts = dunavant_rule(quad_degree)
    total = 0.0
    for tri in triangles:
        P1, P2, P3 = points[tri[0]], points[tri[1]], points[tri[2]]
        for (xi1, xi2), wt in zip(pts_ref, wts):
            # For an affine triangle the Jacobian "area factor" is constant:
            # J_area = tri_cross_mag(P1, P2, P3)
            elem = np.vstack((P1, P2, P3))
            J_area = jacobian_determinant_3d(elem, xi1, xi2)
            # Integrand is 1 -> value at quad points is 1, so contribution = |J| * sum(weights)
            total += J_area * wt
    return total  # equals sum of triangle areas


# ---------------- Exact per-triangle area (cross product) ---------2-------
def exact_area(points, triangles):
    P1 = points[triangles[:, 0]]
    P2 = points[triangles[:, 1]]
    P3 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(P2 - P1, P3 - P1), axis=1).sum()


# ---------------- Load mesh, pick surface triangles, run test ----------------
mesh = meshio.read("fine_sphere.msh")

if "triangle" not in mesh.cells_dict:
    raise RuntimeError(
        "No triangular surface cells found. Re-mesh the sphere surface with triangles."
    )

triangles = boundary_cells_conn
points = boundary_points
# Numerical integration of 1 ds (via quadrature)
area_quad = integrate_one_over_mesh(points, triangles, quad_degree=3)

# Exact by cross product (sanity check; should match area_quad up to ~machine precision)
area_exact = exact_area(points, triangles)

# Estimate sphere radius (robust enough if sphere is centered; otherwise fit center first)
center = points.mean(axis=0)
radii = np.linalg.norm(points - center, axis=1)
R = radii.mean()
area_true = 4.0 * np.pi * R * R

# Report
rel_err_quad_vs_exact = abs(area_quad - area_exact) / max(area_exact, 1.0)
rel_err_exact_vs_true = abs(area_exact - area_true) / area_true

print(f"Triangles: {len(triangles)}   Points: {len(points)}")
print(f"Area (quadrature): {area_quad:.12e}")
print(f"Area (exact xprod): {area_exact:.12e}")
print(f"Area (4πR^2)      : {area_true:.12e}  (R ≈ {R:.6f})")
print(f"rel err quad vs exact : {rel_err_quad_vs_exact:.3e}")
print(f"rel err exact vs 4πR^2: {rel_err_exact_vs_true:.3e}")

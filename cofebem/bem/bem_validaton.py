import meshio, numpy as np
from tqdm import tqdm

mesh = meshio.read("sphere.msh")
cell_type = (
    "triangle"
    if "triangle" in mesh.cells_dict
    else ("quad" if "quad" in mesh.cells_dict else None)
)
if cell_type != "triangle":
    raise RuntimeError("This accelerated path handles P1 triangles only for now.")
F = mesh.cells_dict[cell_type]  # (nElem, 3)
unique_ids, inv = np.unique(F.ravel(), return_inverse=True)
boundary_points = mesh.points[unique_ids]  # (N, 3)
boundary_facets_local = inv.reshape(F.shape)  # (nElem, 3)

mesh_center = boundary_points.mean(axis=0)


def tri_normal(p1, p2, p3):
    a = np.cross(p2 - p1, p3 - p1)
    nrm = np.linalg.norm(a)
    if nrm == 0.0:
        raise ValueError("Degenerate triangle.")
    return a / nrm


nElem = boundary_facets_local.shape[0]
elem_nodes = np.empty((nElem, 3, 3), dtype=np.float64)
elem_normals = np.empty((nElem, 3), dtype=np.float64)
for e, tri in enumerate(boundary_facets_local):
    p1, p2, p3 = (
        boundary_points[tri[0]],
        boundary_points[tri[1]],
        boundary_points[tri[2]],
    )
    elem_nodes[e, :, :] = np.vstack([p1, p2, p3])
    n = tri_normal(p1, p2, p3)
    c = (p1 + p2 + p3) / 3.0
    if np.dot(n, c - mesh_center) < 0:
        n = -n
    elem_normals[e, :] = n

# Per-node averaged normal to offset collocation points
N = boundary_points.shape[0]
node_normals = np.zeros((N, 3), dtype=np.float64)
counts = np.zeros(N, dtype=np.int32)
for e, tri in enumerate(boundary_facets_local):
    n = elem_normals[e]
    for v in tri:
        node_normals[v] += n
        counts[v] += 1
node_normals /= np.maximum(counts[:, None], 1)

edges = np.linalg.norm(
    boundary_points[boundary_facets_local[:, 1]]
    - boundary_points[boundary_facets_local[:, 0]],
    axis=1,
)
h_char = edges.mean()
eps = 1e-6 * h_char
colloc_points = boundary_points + eps * node_normals


def dunavant_rule(degree: int):
    if degree == 1:
        pts = np.array([[1 / 3, 1 / 3]], dtype=float)
        wts = np.array([0.5], dtype=float)
    elif degree == 2:
        pts = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], dtype=float)
        wts = np.array([1 / 6, 1 / 6, 1 / 6], dtype=float)
    elif degree == 5:
        a = 0.059715871789770
        b = 0.470142064105115
        w1 = 0.066197076394253
        c = 0.101286507323456
        d = 0.797426985353087
        w2 = 0.062969590272413
        pts = np.array(
            [[1 / 3, 1 / 3], [b, b], [a, b], [b, a], [c, c], [d, c], [c, d]],
            dtype=float,
        )
        wts = np.array([0.1125, w1, w1, w1, w2, w2, w2], dtype=float)
    else:
        raise ValueError("Dunavant degree not implemented.")
    return pts, wts


E = 1.0e9
nu = 0.3
mu = E / (2 * (1 + nu))


from numba import njit, prange, float64
import numba as nb


@njit(fastmath=True)
def safe_norm3(v):
    s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if s <= 0.0:
        return 0.0
    return s**0.5


@njit(fastmath=True)
def kelvin_G_numba(x, y, mu, nu):
    # Returns 3x3
    r = x - y
    rn = safe_norm3(r)
    if rn == 0.0:
        # caller offsets x; if still zero, return zero to avoid NaN
        out = np.zeros((3, 3), dtype=np.float64)
        return out
    factor = 1.0 / (16.0 * np.pi * mu * (1.0 - nu) * rn)
    rr_over_r2 = np.empty((3, 3), dtype=np.float64)
    inv_r2 = 1.0 / (rn * rn)
    for i in range(3):
        for j in range(3):
            rr_over_r2[i, j] = r[i] * r[j] * inv_r2
    out = np.empty((3, 3), dtype=np.float64)
    three_minus_4nu = 3.0 - 4.0 * nu
    for i in range(3):
        for j in range(3):
            out[i, j] = factor * (
                (three_minus_4nu if i == j else 0.0) + rr_over_r2[i, j]
            )
    return out


@njit(fastmath=True)
def kelvin_H_numba(x, y, n, mu, nu):
    # Returns 3x3 traction kernel with outward normal n at y-panel
    r = x - y
    rn = safe_norm3(r)
    out = np.zeros((3, 3), dtype=np.float64)
    if rn == 0.0:
        return out
    inv_r = 1.0 / rn
    inv_r2 = inv_r * inv_r
    r_dot_n = r[0] * n[0] + r[1] * n[1] + r[2] * n[2]
    # factor = -1 / (8*pi*(1-nu)*r^2)
    factor = -1.0 / (8.0 * np.pi * (1.0 - nu)) * inv_r2
    # term1 = (r·n) * ((1-2nu)I + 3 rr^T / r^2) / r
    # term2 = (1-2nu) * (r⊗n - n⊗r) / r
    one_minus_2nu = 1.0 - 2.0 * nu
    for i in range(3):
        for j in range(3):
            Iij = 1.0 if i == j else 0.0
            rr_over_r2 = 3.0 * r[i] * r[j] * inv_r2
            t1 = r_dot_n * (one_minus_2nu * Iij + rr_over_r2) * inv_r
            t2 = one_minus_2nu * (r[i] * n[j] - n[i] * r[j]) * inv_r
            out[i, j] = factor * (t1 - t2)
    return out


@njit(fastmath=True)
def tri_shape_vals(xi1, xi2):
    # [N1, N2, N3]
    return np.array([1.0 - xi1 - xi2, xi1, xi2], dtype=np.float64)


@njit(fastmath=True)
def tri_jac_area2(nodes):
    # nodes: (3,3)
    e1 = nodes[1] - nodes[0]
    e2 = nodes[2] - nodes[0]
    # area2 = ||e1 x e2||
    cx0 = e1[1] * e2[2] - e1[2] * e2[1]
    cx1 = e1[2] * e2[0] - e1[0] * e2[2]
    cx2 = e1[0] * e2[1] - e1[1] * e2[0]
    return (cx0 * cx0 + cx1 * cx1 + cx2 * cx2) ** 0.5  # = 2*Area


@njit(fastmath=True)
def map_to_phys(nodes, xi1, xi2):
    # nodes: (3,3); xi s.t. N = [1-xi1-xi2, xi1, xi2]
    N1 = 1.0 - xi1 - xi2
    N2 = xi1
    N3 = xi2
    return N1 * nodes[0] + N2 * nodes[1] + N3 * nodes[2]


@njit(parallel=True, fastmath=True)
def assemble_GH_numba(
    colloc_points,  # (N,3)
    elem_nodes,  # (nElem,3,3)
    elem_normals,  # (nElem,3)
    connectivity,  # (nElem,3) global node ids
    quad_pts,  # (Q,2)
    quad_wts,  # (Q,)
    mu,
    nu,
):
    N = colloc_points.shape[0]
    nElem = elem_nodes.shape[0]
    Q = quad_pts.shape[0]
    ND = 3 * N
    G_flat = np.zeros(ND * ND, dtype=np.float64)
    H_flat = np.zeros(ND * ND, dtype=np.float64)

    for i in prange(N):
        x_c = colloc_points[i]
        rs = 3 * i
        for e in range(nElem):
            nodes = elem_nodes[e]
            n = elem_normals[e]
            tri = connectivity[e]

            # integrate 3 local shape functions
            for loc in range(3):
                blockG00 = 0.0
                blockG01 = 0.0
                blockG02 = 0.0
                blockG10 = 0.0
                blockG11 = 0.0
                blockG12 = 0.0
                blockG20 = 0.0
                blockG21 = 0.0
                blockG22 = 0.0

                blockH00 = 0.0
                blockH01 = 0.0
                blockH02 = 0.0
                blockH10 = 0.0
                blockH11 = 0.0
                blockH12 = 0.0
                blockH20 = 0.0
                blockH21 = 0.0
                blockH22 = 0.0

                area2 = tri_jac_area2(nodes)
                for q in range(Q):
                    xi1 = quad_pts[q, 0]
                    xi2 = quad_pts[q, 1]
                    w = quad_wts[q]
                    Nvals = tri_shape_vals(xi1, xi2)
                    Nloc = Nvals[loc]
                    y = map_to_phys(nodes, xi1, xi2)
                    wJ = w * area2

                    Gblk = kelvin_G_numba(x_c, y, mu, nu)
                    Hblk = kelvin_H_numba(x_c, y, n, mu, nu)

                    # accumulate scaled by Nloc*wJ
                    s = Nloc * wJ
                    blockG00 += Gblk[0, 0] * s
                    blockG01 += Gblk[0, 1] * s
                    blockG02 += Gblk[0, 2] * s
                    blockG10 += Gblk[1, 0] * s
                    blockG11 += Gblk[1, 1] * s
                    blockG12 += Gblk[1, 2] * s
                    blockG20 += Gblk[2, 0] * s
                    blockG21 += Gblk[2, 1] * s
                    blockG22 += Gblk[2, 2] * s

                    blockH00 += Hblk[0, 0] * s
                    blockH01 += Hblk[0, 1] * s
                    blockH02 += Hblk[0, 2] * s
                    blockH10 += Hblk[1, 0] * s
                    blockH11 += Hblk[1, 1] * s
                    blockH12 += Hblk[1, 2] * s
                    blockH20 += Hblk[2, 0] * s
                    blockH21 += Hblk[2, 1] * s
                    blockH22 += Hblk[2, 2] * s

                j = tri[loc]
                cs = 3 * j

                # atomic adds into flattened matrices (row-major)
                # row r, col c  -> idx = r*ND + c
                base = rs * ND + cs
                # G
                nb.atomic.add(G_flat, base + 0, blockG00)
                nb.atomic.add(G_flat, base + 1, blockG01)
                nb.atomic.add(G_flat, base + 2, blockG02)
                nb.atomic.add(G_flat, base + ND + 0, blockG10)
                nb.atomic.add(G_flat, base + ND + 1, blockG11)
                nb.atomic.add(G_flat, base + ND + 2, blockG12)
                nb.atomic.add(G_flat, base + 2 * ND + 0, blockG20)
                nb.atomic.add(G_flat, base + 2 * ND + 1, blockG21)
                nb.atomic.add(G_flat, base + 2 * ND + 2, blockG22)
                # H
                nb.atomic.add(H_flat, base + 0, blockH00)
                nb.atomic.add(H_flat, base + 1, blockH01)
                nb.atomic.add(H_flat, base + 2, blockH02)
                nb.atomic.add(H_flat, base + ND + 0, blockH10)
                nb.atomic.add(H_flat, base + ND + 1, blockH11)
                nb.atomic.add(H_flat, base + ND + 2, blockH12)
                nb.atomic.add(H_flat, base + 2 * ND + 0, blockH20)
                nb.atomic.add(H_flat, base + 2 * ND + 1, blockH21)
                nb.atomic.add(H_flat, base + 2 * ND + 2, blockH22)

        # add 0.5*I on H diagonal block (smooth closed surface)
        base = rs * ND + rs
        nb.atomic.add(H_flat, base + 0, 0.5)
        nb.atomic.add(H_flat, base + ND + 1, 0.5)
        nb.atomic.add(H_flat, base + 2 * ND + 2, 0.5)

    return G_flat, H_flat


degree = 2
quad_pts, quad_wts = dunavant_rule(degree)

G_flat, H_flat = assemble_GH_numba(
    colloc_points=colloc_points.astype(np.float64),
    elem_nodes=elem_nodes.astype(np.float64),
    elem_normals=elem_normals.astype(np.float64),
    connectivity=boundary_facets_local.astype(np.int64),
    quad_pts=quad_pts.astype(np.float64),
    quad_wts=quad_wts.astype(np.float64),
    mu=mu,
    nu=nu,
)

ND = 3 * boundary_points.shape[0]
G = G_flat.reshape(ND, ND)
H = H_flat.reshape(ND, ND)
print("G,H assembled with Numba.")

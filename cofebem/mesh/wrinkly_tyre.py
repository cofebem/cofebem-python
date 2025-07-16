import numpy as np
import meshio


def rough_hollow_cylinder(
    nr,
    nt,
    nz,
    r_inner=2.0,
    r_outer=5.0,
    H=1.0,
    A=0.15,
    B=0.05,
    k_r=200,
    k_theta=50,
):

    r_vals = np.linspace(r_inner, r_outer, nr + 1)
    theta_vals = np.linspace(0, 2 * np.pi, nt, endpoint=False)  # unique
    z_vals = np.linspace(0, H, nz + 1)

    points = []
    for k, z in enumerate(z_vals):
        for j, theta in enumerate(theta_vals):
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for r in r_vals:
                points.append([r * cos_t, r * sin_t, z])
    points = np.asarray(points)

    def idx(i, j, k):
        return k * nt * (nr + 1) + j * (nr + 1) + i

    cells = []
    for k in range(nz):
        for j in range(nt):
            jp = (j + 1) % nt  # wrap θ
            for i in range(nr):
                n0 = idx(i, j, k)
                n1 = idx(i + 1, j, k)
                n2 = idx(i + 1, jp, k)
                n3 = idx(i, jp, k)
                n4 = idx(i, j, k + 1)
                n5 = idx(i + 1, j, k + 1)
                n6 = idx(i + 1, jp, k + 1)
                n7 = idx(i, jp, k + 1)
                cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    cells = np.asarray(cells, dtype=int)

    tol = 1e-12
    top_mask = np.abs(points[:, 2] - H) < tol
    x_top, y_top = points[top_mask, 0], points[top_mask, 1]
    r_top = np.hypot(x_top, y_top)
    theta_top = (np.arctan2(y_top, x_top) + 2 * np.pi) % (2 * np.pi)

    dz = A * np.sin(
        2 * np.pi * k_r * (r_top - r_inner) / (r_outer - r_inner)
    ) + B * np.sin(k_theta * theta_top)
    points[top_mask, 2] += dz  # shift upward or downward

    mesh = meshio.Mesh(points, [("hexahedron", cells)])
    meshio.write(f"rough_hollow_cylinder.xdmf", mesh)
    print(f"Mesh written to rough_hollow_cylinder.xdmf")

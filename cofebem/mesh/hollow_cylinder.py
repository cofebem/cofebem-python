import numpy as np
import meshio


def hollow_cylinder(nr, nt, nz, r_inner=4.0, r_outer=5.0, H=1.0):

    # nr: radial divisions; nt: angular divisions; nz: vertical divisions
    r_vals = np.linspace(r_inner, r_outer, nr + 1)

    theta_vals = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    z_vals = np.linspace(0, H, nz + 1)

    points = []
    for k, z in enumerate(z_vals):
        for j, theta in enumerate(theta_vals):
            for i, r in enumerate(r_vals):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append([x, y, z])
    points = np.array(points)

    def index(i, j, k):
        return i + j * (nr + 1) + k * nt * (nr + 1)

    cells = []
    for k in range(nz):
        for j in range(nt):
            jp = (j + 1) % nt
            for i in range(nr):
                n0 = index(i, j, k)
                n1 = index(i + 1, j, k)
                n2 = index(i + 1, jp, k)
                n3 = index(i, jp, k)
                n4 = index(i, j, k + 1)
                n5 = index(i + 1, j, k + 1)
                n6 = index(i + 1, jp, k + 1)
                n7 = index(i, jp, k + 1)
                cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    cells = np.array(cells, dtype=int)

    mesh = meshio.Mesh(points=points, cells=[("hexahedron", cells)])
    meshio.write("hollow_cylinder.xdmf", mesh)
    print("Mesh written to hollow_cylinder.xdmf")

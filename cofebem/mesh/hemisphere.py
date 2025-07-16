import numpy as np
import meshio
from scipy.spatial import Delaunay


def hemisphere(nr, nt, np_, R=1.0):

    r_vals = np.linspace(0.0, R, nr + 1, dtype=float)
    theta_vals = np.linspace(0.0, np.pi / 2, nt + 1, dtype=float)[1:]  # (0, π/2]
    phi_vals = np.linspace(0.0, 2 * np.pi, np_, endpoint=False, dtype=float)  # [0, 2π)

    points = []

    points.append([0.0, 0.0, 0.0])

    points.append([0.0, 0.0, R])

    for r in r_vals[1:]:
        for t in theta_vals:
            for p in phi_vals:
                x = r * np.sin(t) * np.cos(p)
                y = r * np.sin(t) * np.sin(p)
                z = r * np.cos(t)
                points.append([x, y, z])

    points = np.asarray(points, dtype=float)

    tetra = Delaunay(points).simplices  # (n_tets, 4) connectivity array

    mesh = meshio.Mesh(points, [("tetra", tetra)])
    meshio.write("hemisphere.xdmf", mesh)
    print(
        "Created 'hemisphere.xdmf' – %8d points, %8d tetrahedra"
        % (points.shape[0], tetra.shape[0])
    )

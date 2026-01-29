from lcp_solvers import lemke, ccg, psor, nnls
import numpy as np


def compute_nodal_normals_from_facets(mesh, boundary_facets):

    tdim = mesh.topology.dim
    fdim = tdim - 1
    gdim = mesh.geometry.dim

    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)

    X = mesh.geometry.x
    nV = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts

    normals = np.zeros((nV, gdim), dtype=float)

    for f in boundary_facets:
        verts = f2v.links(int(f))
        p0, p1, p2 = X[verts[0]], X[verts[1]], X[verts[2]]
        nf = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nf)
        if norm > 0:
            nf /= norm
        normals[verts] += nf

    norms = np.linalg.norm(normals, axis=1)
    mask = norms > 0
    normals[mask] /= norms[mask][:, None]

    return normals


class Contact:
    def __init__(
        self,
        body1,
        body2,
        g0,
        mu_f,
        lcp_solver="CCG",
    ):
        self.body1 = body1
        self.body2 = body2

        self_Gamma_c1 = body1.Gamma_c
        self_Gamma_c2 = body2.Gamma_c

        self.g0 = g0
        self.mu_f = mu_f

        self.lcp_solver = lcp_solver

    def g_N(self):
        pass

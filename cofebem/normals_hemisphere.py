import numpy as np
from mpi4py import MPI
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem import (
    Function,
    functionspace,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import TrialFunction, TestFunction, inner, FacetNormal, dx, ds, exterior_facet


# ---------------- Mesh ----------------
mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "./msh_files/hemisphere5.msh", MPI.COMM_WORLD, 0, gdim=3
)


V = functionspace(mesh, ("CG", 1, (mesh.topology.dim,)))

n = FacetNormal(mesh)

u, v = TrialFunction(V), TestFunction(V)

normal_fn = Function(V)
normal_fn.name = "normal"

eps = 1e-8
a = eps * inner(u, v) * dx + inner(u, v) * ds

L = inner(n, v) * ds

problem = LinearProblem(
    a=a,
    L=L,
    bcs=[],
    u=normal_fn,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

problem.solve()

normal_fn.x.scatter_forward()

# with XDMFFile(mesh.comm, "./results/normals/normals_hemisphere.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(normal_fn)

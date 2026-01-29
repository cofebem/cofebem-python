import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from ufl import Identity, sym, grad, tr, inner, Measure

from dolfinx.io import gmshio, XDMFFile
from dolfinx.mesh import (
    locate_entities_boundary,
    meshtags,
    exterior_facet_indices,
)
from dolfinx.fem import (
    functionspace,
    Constant,
    dirichletbc,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem

comm = MPI.COMM_WORLD
mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "coarse_sphere.msh", comm, rank=0, gdim=3
)

E = 1.0e9
nu = 0.3
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

tdim = mesh.topology.dim
fdim = tdim - 1


V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


def epsilon(w):
    return sym(grad(w))


def sigma(w):
    return 2 * mu * epsilon(w) + lmbda * tr(epsilon(w)) * Identity(3)


a = inner(sigma(u), epsilon(v)) * ufl.dx

n = ufl.FacetNormal(mesh)
p = Constant(mesh, PETSc.ScalarType(1.0e9))  # Pa
t = -p * n

mesh.topology.create_connectivity(fdim, 0)
ext_facets = exterior_facet_indices(mesh.topology)
facet_mt = meshtags(mesh, fdim, ext_facets, np.ones_like(ext_facets, dtype=np.int32))
ds = Measure("ds", domain=mesh, subdomain_data=facet_mt)

L = inner(t, v) * ds(1)  # apply traction everywhere on boundary


x = mesh.geometry.x
r = np.linalg.norm(x, axis=1)
R_guess = float(np.mean(r))  # rough radius
tol = 1e-8 + 5e-3 * R_guess


# def near_point(pt, atol):
#     def _pred(xx):
#         return (
#             np.isclose(xx[0], pt[0], atol=atol)
#             & np.isclose(xx[1], pt[1], atol=atol)
#             & np.isclose(xx[2], pt[2], atol=atol)
#         )

#     return _pred


# # Targets on the sphere
# px = np.array([R_guess, 0.0, 0.0])
# py = np.array([0.0, R_guess, 0.0])
# pz = np.array([0.0, 0.0, R_guess])

# # Component subspaces for partial constraints
# Vx = V.sub(0)
# Vy = V.sub(1)
# Vz = V.sub(2)

# # Full (x,y,z) at px
# dofs_px = locate_dofs_geometrical(V, near_point(px, tol))
# bc_px = dirichletbc(PETSc.ScalarType((0.0, 0.0, 0.0)), dofs_px, V)

# # (y,z)=0 at py
# dofs_py_y = locate_dofs_geometrical(Vy, near_point(py, tol))
# dofs_py_z = locate_dofs_geometrical(Vz, near_point(py, tol))
# bc_py_y = dirichletbc(PETSc.ScalarType(0.0), dofs_py_y, Vy)
# bc_py_z = dirichletbc(PETSc.ScalarType(0.0), dofs_py_z, Vz)

# # (x)=0 at pz
# dofs_pz_x = locate_dofs_geometrical(Vx, near_point(pz, tol))
# bc_pz_x = dirichletbc(PETSc.ScalarType(0.0), dofs_pz_x, Vx)

# bcs = [bc_px, bc_py_y, bc_py_z, bc_pz_x]
bcs = []
problem = LinearProblem(
    a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()
uh.name = "u"

# A = problem.assembler.assemble_scalar(ufl.Constant(mesh, PETSc.ScalarType(1.0)) * ds(1))
# comm.Allreduce(MPI.IN_PLACE, A, op=MPI.SUM)
A = problem.A
xvec = ufl.SpatialCoordinate(mesh)
x_dot_n = inner(xvec, n)
Xn = problem.assembler.assemble_scalar(x_dot_n * ds(1))
comm.Allreduce(MPI.IN_PLACE, Xn, op=MPI.SUM)
R_mean = Xn / A

Un = problem.assembler.assemble_scalar(inner(uh, n) * ds(1))
comm.Allreduce(MPI.IN_PLACE, Un, op=MPI.SUM)
u_r_mean = Un / A

# Theory
u_th = -(1.0 - 2.0 * nu) / E * float(p.value) * R_mean

if comm.rank == 0:
    print(f"Estimated radius R_mean = {R_mean:.6e} m")
    print(f"Analytic u_r(R)         = {u_th:.6e} m")
    print(f"Mean FEM <u·n>          = {u_r_mean:.6e} m")
    rel_err = 0.0 if abs(u_th) < 1e-30 else abs((u_r_mean - u_th) / u_th)
    print(f"Relative error (mean)   = {100*rel_err:.2f} %")

with XDMFFile(comm, "sphere_fem.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)

with XDMFFile(comm, "sphere_bcs.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_mt, mesh.geometry)

if comm.rank == 0:
    print("DONE")

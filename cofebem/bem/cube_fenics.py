import numpy as np
import math
import time
import matplotlib.pyplot as plt

from dolfinx.mesh import (
    CellType,
    GhostMode,
    locate_entities_boundary,
    locate_entities,
    meshtags,
    exterior_facet_indices,
)
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import (
    Identity,
    Measure,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)
from dolfinx.io import gmshio, XDMFFile, VTKFile

from mpi4py import MPI
from petsc4py import PETSc

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "msh_files/cube_tetra.msh", MPI.COMM_WORLD, 0, gdim=3
)

tdim = mesh.topology.dim
fdim = tdim - 1

E = 1.0e9
nu = 0.3

lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))


# -------------------------------------------------------------------------------------------------------
#  LE weak form
# -------------------------------------------------------------------------------------------------------

element_type = "Lagrange"
element_degree = 1

V = functionspace(mesh, (element_type, element_degree, (mesh.geometry.dim,)))

u, v = TrialFunction(V), TestFunction(V)


def epsilon(v):
    return sym(grad(v))


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * tr(epsilon(u)) * Identity(len(u))


def a(u, v):
    return inner(sigma(u), epsilon(v)) * dx


# -------------------------------------------------------------------------------------------------------
#  Component-wise Dirichlet BCs (ux|x=0 = 0, uy|y=0 = 0, uz|z=0 = 0) and uniform Neumann tz = -1e8 on z=1
# -------------------------------------------------------------------------------------------------------
TOL = 1e-8


def on_x0(x):
    return np.isclose(x[0], 0.0, atol=TOL)


def on_y0(x):
    return np.isclose(x[1], 0.0, atol=TOL)


def on_z0(x):
    return np.isclose(x[2], 0.0, atol=TOL)


Vx, map0 = V.sub(0).collapse()
Vy, map1 = V.sub(1).collapse()
Vz, map2 = V.sub(2).collapse()

fdim = mesh.topology.dim - 1
Gamma_x0 = locate_entities_boundary(mesh, fdim, on_x0)
Gamma_y0 = locate_entities_boundary(mesh, fdim, on_y0)
Gamma_z0 = locate_entities_boundary(mesh, fdim, on_z0)

ux_dofs = locate_dofs_topological((V.sub(0), Vx), fdim, Gamma_x0)
uy_dofs = locate_dofs_topological((V.sub(1), Vy), fdim, Gamma_y0)
uz_dofs = locate_dofs_topological((V.sub(2), Vz), fdim, Gamma_z0)

zero = Constant(mesh, 0.0)
u_Dx = Function(Vx)
u_Dy = Function(Vy)
u_Dz = Function(Vz)

bcx = dirichletbc(u_Dx, ux_dofs, V.sub(0))
bcy = dirichletbc(u_Dy, uy_dofs, V.sub(1))
bcz = dirichletbc(u_Dz, uz_dofs, V.sub(2))
bcs = [bcx, bcy, bcz]


def Gamma_u_locator(x):
    return np.isclose(x[2], 0, atol=TOL)


Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_locator)


def Gamma_t_locator(x):
    return np.isclose(x[2], 1, atol=TOL)


Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)

if not Gamma_t.size:
    raise ValueError("No boundary facets found for the given locator.")


dirchlet_marker = 1
neumann_marker = 2

Gamma_u_marker = np.full(Gamma_u.size, dirchlet_marker, dtype=np.int32)  # Dirichlet
Gamma_t_marker = np.full(Gamma_t.size, neumann_marker, dtype=np.int32)  # Neumann

It = locate_dofs_geometrical(V, Gamma_t_locator)

Gamma = np.concatenate([Gamma_u, Gamma_t])
Gamma_markers = np.concatenate([Gamma_u_marker, Gamma_t_marker])

perm = np.argsort(Gamma)

Gamma_mt = meshtags(mesh, fdim, Gamma[perm], Gamma_markers[perm])


ds = Measure("ds", domain=mesh, subdomain_data=Gamma_mt)

p = 1.0e8
t = Constant(mesh, np.array([0.0, 0.0, -p], dtype=PETSc.ScalarType))
f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


def L(v):
    return inner(f_v, v) * dx + inner(t, v) * ds(neumann_marker)


sol = Function(V)
sol.name = "u"


with VTKFile(mesh.comm, "CubePatch_fenics.pvd", "w") as vtk:
    vtk.write_mesh(mesh)
    vtk.write_function(sol, 0)


problem = LinearProblem(
    a=a(u, v),
    L=L(v),
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

uh = problem.solve()

sol.x.array[:] = uh.x.array[:]
sol.x.scatter_forward()

vtk.write_function(sol, 1)

if mesh.comm.rank == 0:
    print("Wrote CubePatch_fenics.pvd")


coords = mesh.geometry.x
N = coords.shape[0]

alpha = (nu * p) / E
beta = -p / E

u_th = np.zeros_like(coords)
u_th[:, 0] = alpha * coords[:, 0]
u_th[:, 1] = alpha * coords[:, 1]
u_th[:, 2] = beta * coords[:, 2]


u_th_flat = np.zeros(3 * N, dtype=sol.x.array.dtype)
u_th_flat[0::3] = u_th[:, 0]
u_th_flat[1::3] = u_th[:, 1]
u_th_flat[2::3] = u_th[:, 2]


u_num = sol.x.array

err_vec = u_num - u_th_flat

L2_abs_err = np.linalg.norm(err_vec)
L2_rel_err = L2_abs_err / np.linalg.norm(u_th_flat)

if mesh.comm.rank == 0:
    print("Absolute L2 error =", L2_abs_err)
    print("Relative L2 error =", L2_rel_err)

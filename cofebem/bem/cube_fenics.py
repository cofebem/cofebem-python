import numpy as np
import math
import time
import matplotlib.pyplot as plt

from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    locate_entities,
    meshtags,
    exterior_facet_indices,
)
from dolfinx.fem import (
    Constant,
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
from dolfinx.io import gmshio, XDMFFile

from mpi4py import MPI
from petsc4py import PETSc

import mpi4py.MPI as MPI


# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "cube_tetra.msh", MPI.COMM_WORLD, 0, gdim=3
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
#  Set boundary conditions
# -------------------------------------------------------------------------------------------------------
tol = 1.0e-5


def Gamma_u_locator(x):
    return np.isclose(x[2], 0, atol=tol)


Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_locator)
Iu = locate_dofs_topological(V, entity_dim=fdim, entities=Gamma_u)

u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)


bc = dirichletbc(
    u0,
    dofs=Iu,
    V=V,
)


def Gamma_t_locator(x):
    return np.isclose(x[2], 1, atol=tol) & (
        (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 <= 0.3**2
    )


Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)

if not Gamma_t.size:
    raise ValueError("No boundary facets found for the given locator.")


dirchlet_marker = 1
neumann_marker = 2

Gamma_u_marker = np.full(Gamma_u.size, dirchlet_marker, dtype=np.int32)  # Dirichlet
Gamma_t_marker = np.full(Gamma_t.size, neumann_marker, dtype=np.int32)  # Neumann

Gamma = np.concatenate([Gamma_u, Gamma_t])
Gamma_markers = np.concatenate([Gamma_u_marker, Gamma_t_marker])

perm = np.argsort(Gamma)

Gamma_mt = meshtags(mesh, fdim, Gamma[perm], Gamma_markers[perm])

ds = Measure("ds", domain=mesh, subdomain_data=Gamma_mt)


f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
t = Constant(mesh, np.array([0.0, 0.0, -1.0e9], dtype=PETSc.ScalarType))


def L(v):
    return inner(f_v, v) * dx + inner(t, v) * ds(neumann_marker)


# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------

problem = LinearProblem(
    a=a(u, v), L=L(v), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

problem.solve()
problem.u.name = "u"

with XDMFFile(MPI.COMM_WORLD, f"cube_bcs.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(Gamma_mt, mesh.geometry)
    # xdmf.write_function(problem.u)

print("DONE")

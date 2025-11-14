import numpy as np

from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    meshtags,
)
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    form,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from ufl import (
    Identity,
    Measure,
    FunctionSpace,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)
from dolfinx.io import VTKFile

from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

import mpi4py.MPI as MPI

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

nx = 20
ny = 20
nz = 10

l = 1

mesh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([l, l, l])],
    [nx, ny, nz],
    CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
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


f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


def L(v):
    return inner(f_v, v) * dx


# -------------------------------------------------------------------------------------------------------
#  Set boundary conditions
# -------------------------------------------------------------------------------------------------------

tol = 1.0e-5


def Gamma_u_selector(x):
    return np.isclose(x[2], 0, atol=tol)


Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_selector)
Iu = locate_dofs_topological(V, entity_dim=fdim, entities=Gamma_u)

u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)


bc = dirichletbc(
    u0,
    dofs=Iu,
    V=V,
)


# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------

problem = LinearProblem(
    a=a(u, v), L=L(v), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

problem.solve()


# -------------------------------------------------------------------------------------------------------
#  Gamma_c : Contact Region
# -------------------------------------------------------------------------------------------------------

tol = 1.0e-5


# def Gamma_c_selector(x):
#     return (np.isclose(x[2], l, atol=tol)) & ((x[0] - l) ** 2 + (x[1] - l) ** 2 <= 8.0)


def Gamma_c_selector(x):
    return np.isclose(x[2], l, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)
Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# -------------------------------------------------------------------------------------------------------
#  Sc construction: Contact compliance matrix
# -------------------------------------------------------------------------------------------------------

Sc_ = Sc(problem.A, problem.b, tdim, Ic)
Sc_dense = Sc_.by_sampling()

# Sc_.save(file="./results/cube/Sc.npy")

# -------------------------------------------------------------------------------------------------------
#  Contact Problem
# -------------------------------------------------------------------------------------------------------

# Sc_dense = np.load("./results/cube/Sc.npy")


num_facets = mesh.topology.index_map(fdim).size_local
facet_values = np.zeros(num_facets, dtype=np.int32)
facet_values[Gamma_c] = 1  # mark contact facets with 1

facet_tags = meshtags(mesh, fdim, np.arange(num_facets, dtype=np.int32), facet_values)

# Boundary measure restricted to Γ_c (marker 1)
ds_c = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=1)

# 2) Scalar P1 and P0 spaces on the volume mesh
Vs = functionspace(mesh, ("Lagrange", 1, (1,)))
# FunctionSpace(mesh, ("Lagrange", 1))  # scalar P1
Q = functionspace(mesh, ("Discontinuous Lagrange", 0, (1,)))
# FunctionSpace(mesh, ("Discontinuous Lagrange", 0))  # scalar P0 (per cell)

# Contact dofs in scalar P1 space (these should correspond to your contact nodes)
Is = locate_dofs_topological(Vs, fdim, Gamma_c)
Nc_scalar = len(Is)

# 3) Assemble mass matrices on Γ_c

# P0–P0 on Γ_c: M_tt
p = TrialFunction(Q)
q = TestFunction(Q)
a_Mtt = inner(p, q) * ds_c
M_tt = assemble_matrix(form(a_Mtt))
M_tt.assemble()

# P0–P1 on Γ_c: M_tu  (rows: P0 dofs, cols: P1 dofs)
us = TrialFunction(Vs)
q = TestFunction(Q)
a_Mtu = inner(us, q) * ds_c
M_tu = assemble_matrix(form(a_Mtu))
M_tu.assemble()

# 4) Extract only the P0 dofs that actually lie on Γ_c (cells touching Γ_c)

A_tt = M_tt.convert("dense").getDenseArray()
row_norms = np.linalg.norm(A_tt, axis=1)
contact_cells_p0 = np.where(row_norms > 1e-14)[
    0
]  # indices of P0 dofs with nonzero boundary measure
Ne = len(contact_cells_p0)

M_tt_c = A_tt[np.ix_(contact_cells_p0, contact_cells_p0)]

A_tu = M_tu.convert("dense").getDenseArray()
# Restrict columns to the P1 dofs on Γ_c (Is) and rows to the contact P0 dofs:
M_tu_c = A_tu[np.ix_(contact_cells_p0, Is)]  # shape: (Ne, Nc_scalar)

# 5) Projection matrix Π_{u->p} = M_tt^{-1} M_tu  (dense for now)
Pi_u_to_p = np.linalg.solve(M_tt_c, M_tu_c)  # (Ne x Nc_scalar)

# 6) Build element-wise Sc_elem from nodal Sc_nodes
#    IMPORTANT: Sc_nodes must be ordered consistently with the scalar P1 contact dofs Is.
#    If your Sc_nodes is already built on these nodes, you can do:

Sc_elem = Pi_u_to_p @ Sc_dense @ Pi_u_to_p.T  # (Ne x Ne)

print("Sc_nodes shape:", Sc_dense.shape, len(Ic))
print("Sc_elem shape :", Sc_elem.shape, len(Gamma_c))

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
from dolfinx.io import gmshio, XDMFFile

from mpi4py import MPI
from petsc4py import PETSc

# from cofebem.contact import Sc

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
# tol = 1.0e-5


# def Gamma_u_locator(x):
#     return np.isclose(x[2], 0, atol=tol)


# Gamma_u = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_u_locator)
# Iu = locate_dofs_topological(V, entity_dim=fdim, entities=Gamma_u)

# u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)


# bc = dirichletbc(
#     u0,
#     dofs=Iu,
#     V=V,
# )


# def Gamma_t_locator(x):
#     return np.isclose(x[2], 1, atol=tol) & (
#         (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 <= 0.3**2
#     )


# Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)

# if not Gamma_t.size:
#     raise ValueError("No boundary facets found for the given locator.")


# dirchlet_marker = 1
# neumann_marker = 2

# Gamma_u_marker = np.full(Gamma_u.size, dirchlet_marker, dtype=np.int32)  # Dirichlet
# Gamma_t_marker = np.full(Gamma_t.size, neumann_marker, dtype=np.int32)  # Neumann

# It = locate_dofs_geometrical(V, Gamma_t_locator)

# Gamma = np.concatenate([Gamma_u, Gamma_t])
# Gamma_markers = np.concatenate([Gamma_u_marker, Gamma_t_marker])

# perm = np.argsort(Gamma)

# Gamma_mt = meshtags(mesh, fdim, Gamma[perm], Gamma_markers[perm])


# ds = Measure("ds", domain=mesh, subdomain_data=Gamma_mt)


# f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
# t = Constant(mesh, np.array([0.0, 0.0, -1.0e8], dtype=PETSc.ScalarType))


# def L(v):
#     return inner(f_v, v) * dx + inner(t, v) * ds(neumann_marker)


# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------

# problem = LinearProblem(
#     a=a(u, v), L=L(v), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
# )

# problem.solve()
# problem.u.name = "u"

# mesh.topology.create_connectivity(fdim, 0)
# conn = mesh.topology.connectivity(fdim, 0)

# node_area = np.zeros(len(It))

# for id_node, node in enumerate(It):
#     for id_elem in Gamma_t:
#         elem_nodes = conn.links(id_elem)
#         if node in elem_nodes:
#             v1, v2, v3 = (
#                 mesh.geometry.x[elem_nodes[0]],
#                 mesh.geometry.x[elem_nodes[1]],
#                 mesh.geometry.x[elem_nodes[2]],
#             )

#             elem_area = (1 / 3) * (0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1)))
#             node_area[id_node] += elem_area


# solver_petsc = PETSc.KSP().create(mesh.comm)
# solver_petsc.setOperators(problem.A)
# solver_petsc.setType("preonly")
# solver_petsc.getPC().setType("lu")
# solver_petsc.setFromOptions()
# solver_petsc.setUp()

# b = problem.b.copy()
# u_ = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)

# b.set(0)
# for i, dof in enumerate(It):
#     b.setValue(dof * tdim + 2, -(1e9) * node_area[i])
#     b.assemble()

# solver_petsc.solve(b, u_)
# from dolfinx.fem import Function

# u_fenics = Function(V)

# u_fenics.x.array[:] = u_.array
# u_fenics.x.scatter_forward()

# u_fenics.name = "u"


# with XDMFFile(MPI.COMM_WORLD, f"cube_fenics.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(u_fenics)  # or omit t if you prefer


# with XDMFFile(MPI.COMM_WORLD, f"cube_bcs.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(Gamma_mt, mesh.geometry)
#     # xdmf.write_function(problem.u)

# with XDMFFile(MPI.COMM_WORLD, f"cube_fenics.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(problem.u)
# print("DONE")

# -------------------------------------------------------------------------------------------------------
#  Component-wise Dirichlet BCs (ux|x=0 = 0, uy|y=0 = 0, uz|z=0 = 0) and uniform Neumann tz = -1 everywhere
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

t = Constant(mesh, np.array([0.0, 0.0, -1.0e8], dtype=PETSc.ScalarType))
f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


def L(v):
    return inner(f_v, v) * dx + inner(t, v) * ds(neumann_marker)


problem = LinearProblem(
    a=a(u, v),
    L=L(v),
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

uh = problem.solve()
uh.name = "u"

with XDMFFile(MPI.COMM_WORLD, "cube_fenics_new.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)

if mesh.comm.rank == 0:
    print("Wrote cube_fenics_new.xdmf")


# def _as_int32_1d(dofs_like):
#     """Flatten sequences of arrays into one contiguous int32 vector."""
#     if isinstance(dofs_like, (list, tuple)):
#         return np.ascontiguousarray(
#             np.hstack([np.asarray(a, dtype=np.int32) for a in dofs_like]),
#             dtype=np.int32,
#         )
#     return np.ascontiguousarray(np.asarray(dofs_like, dtype=np.int32), dtype=np.int32)


# tol = 1.0e-8


# def on_x0(x):
#     return np.isclose(x[0], 0.0, atol=tol)


# def on_y0(x):
#     return np.isclose(x[1], 0.0, atol=tol)


# def on_z0(x):
#     return np.isclose(x[2], 0.0, atol=tol)


# fdim = mesh.topology.dim - 1
# Gamma_x0 = locate_entities_boundary(mesh, fdim, on_x0)
# Gamma_y0 = locate_entities_boundary(mesh, fdim, on_y0)
# Gamma_z0 = locate_entities_boundary(mesh, fdim, on_z0)

# ux_dofs = locate_dofs_topological((V.sub(0), V), fdim, Gamma_x0)
# uy_dofs = locate_dofs_topological((V.sub(1), V), fdim, Gamma_y0)
# uz_dofs = locate_dofs_topological((V.sub(2), V), fdim, Gamma_z0)

# ux_dofs = _as_int32_1d(locate_dofs_topological((V.sub(0), V), fdim, Gamma_x0))
# uy_dofs = _as_int32_1d(locate_dofs_topological((V.sub(1), V), fdim, Gamma_y0))
# uz_dofs = _as_int32_1d(locate_dofs_topological((V.sub(2), V), fdim, Gamma_z0))

# zero_c = Constant(mesh, PETSc.ScalarType(0.0))
# bcx = dirichletbc(zero_c, ux_dofs, V.sub(0))
# bcy = dirichletbc(zero_c, uy_dofs, V.sub(1))
# bcz = dirichletbc(zero_c, uz_dofs, V.sub(2))
# bcs = [bcx, bcy, bcz]

# ds = Measure("ds", domain=mesh)
# t = Constant(mesh, np.array([0.0, 0.0, -1.0e8], dtype=PETSc.ScalarType))
# f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))


# def L(v):
#     return inner(f_v, v) * dx + inner(t, v) * ds


# # -------------------------------------------------------------------------------------------------------
# #  Solve
# # -------------------------------------------------------------------------------------------------------
# problem = LinearProblem(
#     a=a(u, v),
#     L=L(v),
#     bcs=bcs,
#     petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
# )

# uh = problem.solve()
# uh.name = "u"

# with XDMFFile(MPI.COMM_WORLD, "cube_fenics_new.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(uh)

# # Quick sanity: report counts
# if mesh.comm.rank == 0:
#     print(
#         f"|ux fixed| = {len(ux_dofs)}, |uy fixed| = {len(uy_dofs)}, |uz fixed| = {len(uz_dofs)}"
#     )
#     print("Wrote cube_fenics_new.xdmf")


# def Gamma_c_selector(x):
#     return np.isclose(x[2], 1, atol=tol)


# Gamma_c = locate_entities_boundary(
#     mesh, dim=mesh.topology.dim - 1, marker=Gamma_c_selector
# )

# Ic = locate_dofs_topological(V, mesh.topology.dim - 1, Gamma_c)

# Sc_ = Sc.Sc(problem.A, problem.b.copy(), 3, Ic, full_Sc=True).by_sampling()

# np.savez("Sc_FEM.npz", Sc=Sc_)

# data_BEM = np.load("Sc_BEM.npz")

# data_FEM = np.load("Sc_FEM.npz")

# Sc_BEM = data_BEM["Sc"]
# Sc_FEM = data_FEM["Sc"]

# error_fro = np.linalg.norm(Sc_BEM - Sc_FEM)

# print(f"error relative Sc BEM/FEM = {error_fro}")
# print(f"norm BEM = {np.linalg.norm(Sc_BEM)}")
# print(f"norm FEM = {np.linalg.norm(Sc_FEM)}")


# from cofebem.contact.lcp_solvers.ccg import CCG

# displ = 0.3
# Rindenter = 10.0


# def _parabolic_indenter(x, y, x0, y0, R, z0):
#     if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > R:
#         return z0 + R
#     else:
#         return z0 + R - np.sqrt(R**2 - (x - x0) ** 2 - (y - y0) ** 2)


# parabolic_indenter = np.vectorize(_parabolic_indenter)


# max_iter = 1000
# tolerance = 1e-5
# error_type = "nw"
# pfactor = 1e8
# Nframes = 40

# ANIMATION = False
# if ANIMATION == True:
#     # # x_center = np.linspace(-7.0, 7.0, Nframes)
#     # theta_c = np.linspace(0.0, 2 * np.pi, Nframes)
#     # rc = 3.0
#     # for frame, theta_c_ in enumerate(theta_c):
#     #     contact_center = np.array([rc * np.cos(theta_c_), rc * np.sin(theta_c_)])
#     #     gap = (
#     #         flat_indenter(
#     #             Gamma_c_x[:, 0],
#     #             Gamma_c_x[:, 1],
#     #             contact_center[0],
#     #             contact_center[1],
#     #             Rindenter,
#     #             np.ones_like(Gamma_c_x[:, 2]) - displ,
#     #         )
#     #         - Gamma_c_x[:, 2]
#     #     )

#     contact_center = np.array([0.0, 0.0])
#     z_start = 1.2
#     z_end = 1.0 - displ
#     Nframes = 40

#     z_positions = np.linspace(z_start, z_end, Nframes)

#     for frame, z0 in enumerate(z_positions):
#         gap = (
#             flat_indenter(
#                 Gamma_c_x[:, 0],
#                 Gamma_c_x[:, 1],
#                 contact_center[0],
#                 contact_center[1],
#                 Rindenter,
#                 np.full_like(Gamma_c_x[:, 2], z0),
#             )
#             - Gamma_c_x[:, 2]
#         )
#         penetrating_nodes = np.where(gap < 0)[0]

#         if frame == 0:
#             p, _, _ = constrained_CG(Sc_classic, error_type, gap, max_iter, tolerance)
#         else:
#             p, _, _ = constrained_CG(Sc_classic, error_type, gap, max_iter, tolerance)

#         solver_petsc = PETSc.KSP().create(mesh.comm)
#         solver_petsc.setOperators(problem.A)
#         solver_petsc.setType("preonly")
#         solver_petsc.getPC().setType("lu")
#         solver_petsc.setFromOptions()
#         solver_petsc.setUp()

#         b = problem.b.copy()
#         u = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)

#         b.set(0)
#         for i, dof in enumerate(Ic):
#             b.setValue(dof * tdim + 2, p[i])
#             b.assemble()

#         solver_petsc.solve(b, u)
#         from dolfinx.fem import Function

#         u_fenics = Function(V)

#         u_fenics.x.array[:] = -u.array
#         u_fenics.x.scatter_forward()  # for parallel ghost updates

#         u_fenics.name = "u"

#         with XDMFFile(MPI.COMM_WORLD, f"wtyre_disp_{frame}.xdmf", "w") as xdmf:
#             xdmf.write_mesh(mesh)
#             xdmf.write_function(u_fenics, t=frame)

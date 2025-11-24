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
from dolfinx.io import VTKFile, XDMFFile, gmshio

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


def Gamma_c_selector(x):
    return np.isclose(x[2], l, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)

mesh.topology.create_connectivity(fdim, 0)
facet_to_vertices = mesh.topology.connectivity(fdim, 0)

contact_facets = Gamma_c
Ne = len(contact_facets)

contact_vertices = np.unique(
    np.concatenate([facet_to_vertices.links(f) for f in contact_facets])
).astype(np.int32)

Ic = contact_vertices
Nc = len(Ic)
print("Nc (contact nodes):", Nc)
Gamma_c_x = mesh.geometry.x[Ic]


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


# num_facets = mesh.topology.index_map(fdim).size_local
# facet_values = np.zeros(num_facets, dtype=np.int32)
# facet_values[Gamma_c] = 1

# facet_tags = meshtags(mesh, fdim, np.arange(num_facets, dtype=np.int32), facet_values)

# ds_c = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=1)

# Vs = functionspace(mesh, ("Lagrange", 1, (1,)))

# Q = functionspace(mesh, ("Discontinuous Lagrange", 0, (1,)))

# Is_list = []
# for v in Ic:
#     dofs_v = locate_dofs_topological(Vs, 0, np.array([v], dtype=np.int32))
#     assert len(dofs_v) == 1
#     Is_list.append(dofs_v[0])

# Is = np.array(Is_list, dtype=np.int32)
# Nc_scalar = len(Is)

# p = TrialFunction(Q)
# q = TestFunction(Q)
# a_Mtt = inner(p, q) * ds_c
# M_tt = assemble_matrix(form(a_Mtt))
# M_tt.assemble()

# us = TrialFunction(Vs)
# q = TestFunction(Q)
# a_Mtu = inner(us, q) * ds_c
# M_tu = assemble_matrix(form(a_Mtu))
# M_tu.assemble()


# A_tt = M_tt.convert("dense").getDenseArray()
# row_norms = np.linalg.norm(A_tt, axis=1)
# contact_cells_p0 = np.where(row_norms > 1e-14)[0]
# Ne = len(contact_cells_p0)

# M_tt_c = A_tt[np.ix_(contact_cells_p0, contact_cells_p0)]

# A_tu = M_tu.convert("dense").getDenseArray()

# M_tu_c = A_tu[np.ix_(contact_cells_p0, Is)]  # shape: (Ne, Nc_scalar)

# Pi_u_to_p = np.linalg.solve(M_tt_c, M_tu_c)  # (Ne x Nc_scalar)


# Sc_elem = Pi_u_to_p @ Sc_dense @ Pi_u_to_p.T  # (Ne x Ne)

# print("Sc_nodes shape:", Sc_dense.shape, len(Ic))
# print("Sc_elem shape :", Sc_elem.shape, len(Gamma_c))

#############################################################################

vertex_to_local = {v: i for i, v in enumerate(Ic)}

Pi_u_to_p = np.zeros((Ne, Nc), dtype=np.float64)

for e_idx, f in enumerate(contact_facets):
    vs = facet_to_vertices.links(f)
    local_ids = [vertex_to_local[v] for v in vs if v in vertex_to_local]
    nv = len(local_ids)
    assert nv > 0
    w = 1.0 / nv
    for lid in local_ids:
        Pi_u_to_p[e_idx, lid] = w

print("Pi_u_to_p shape:", Pi_u_to_p.shape)

Sc_elem = Pi_u_to_p @ Sc_dense @ Pi_u_to_p.T  # (Ne x Ne)
print("Sc_elem shape:", Sc_elem.shape)


# ================================================================
# Contact computation at P0 element centers using Sc_elem
# ================================================================

from cofebem.contact.lcp_solvers.ccg import CCG

centers = np.zeros((Ne, 3), dtype=np.float64)
for e_idx, f in enumerate(contact_facets):
    vs = facet_to_vertices.links(f)
    coords = mesh.geometry.x[vs]
    centers[e_idx, :] = coords.mean(axis=0)


displ = 0.2
Rindenter = 1.5


def _parabolic_indenter(x, y, x0, y0, R, z0):
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    z = np.empty_like(x)
    outside = r2 > R**2
    z[outside] = z0 + R
    z[~outside] = z0 + R - np.sqrt(R**2 - r2[~outside])
    return z


parabolic_indenter = np.vectorize(_parabolic_indenter)

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 10

x_start = 0.0
x_end = 1.0
x_pos = np.linspace(x_start, x_end, Nframes)

USE_3POINT_GAP = True
GAP_AGGREGATION = "average"

for frame, x0 in enumerate(x_pos):
    contact_center = np.array([x0, 0.5])

    if not USE_3POINT_GAP:
        z0 = np.full(Ne, 1.0) - displ
        z_ind = parabolic_indenter(
            centers[:, 0],
            centers[:, 1],
            contact_center[0],
            contact_center[1],
            Rindenter,
            z0,
        )
        gap_elem = z_ind - centers[:, 2]

    else:
        gap_elem = np.zeros(Ne, dtype=np.float64)
        z0_local = 1.0 - displ

        for e_idx, f in enumerate(contact_facets):
            vs = facet_to_vertices.links(f)  # vertices of facet f
            coords = mesh.geometry.x[vs]  # shape (3, 3) for triangles

            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            z_coords = coords[:, 2]

            z_ind_pts = parabolic_indenter(
                x_coords,
                y_coords,
                contact_center[0],
                contact_center[1],
                Rindenter,
                np.full(len(x_coords), z0_local),
            )
            gaps_pts = z_ind_pts - z_coords  # gap at each of the 3 points

            if GAP_AGGREGATION == "min":
                gap_elem[e_idx] = gaps_pts.min()
            else:  # "average" (default)
                gap_elem[e_idx] = gaps_pts.mean()

    p_elem, _, _ = CCG(Sc_elem, error_type, gap_elem, max_iter, tolerance).solve()
    # p_elem = lemkelcp(Sc_elem, -gap_elem)[0]  # alternative

    p_nodes = Pi_u_to_p.T @ p_elem

    solver_petsc = PETSc.KSP().create(mesh.comm)
    solver_petsc.setOperators(problem.A)
    solver_petsc.setType("preonly")
    solver_petsc.getPC().setType("lu")
    solver_petsc.setFromOptions()
    solver_petsc.setUp()

    b = problem.b.copy()
    b.set(0)

    for i_node, dof in enumerate(Ic):
        b.setValue(dof * tdim + 2, p_nodes[i_node])
    b.assemble()

    u = problem.A.createVecRight()
    solver_petsc.solve(b, u)

    from dolfinx.fem import Function

    u_fenics = Function(V)
    u_fenics.x.array[:] = -u.array
    u_fenics.x.scatter_forward()
    u_fenics.name = "u"

    with XDMFFile(mesh.comm, f"CubeP0traction_{frame}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_fenics, t=frame)

################################################################################################
################################################################################################

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
# Nframes = 10

# x_start = 0.0
# x_end = 1.0
# x_pos = np.linspace(x_start, x_end, Nframes)

# for frame, x0 in enumerate(x_pos):
#     contact_center = np.array([x0, 0.5])
#     gap_elem = (
#         parabolic_indenter(
#             centers[:, 0],
#             centers[:, 1],
#             contact_center[0],
#             contact_center[1],
#             Rindenter,
#             np.full(len(centers), 1) - displ,
#         )
#         - centers[:, 2]
#     )

#     p_elem, _, _ = CCG(Sc_elem, error_type, gap_elem, max_iter, tolerance).solve()
#     # p_elem = lemkelcp(Sc_elem, -gap_elem)[0]  # alternative

#     p_nodes = Pi_u_to_p.T @ p_elem

#     solver_petsc = PETSc.KSP().create(mesh.comm)
#     solver_petsc.setOperators(problem.A)
#     solver_petsc.setType("preonly")
#     solver_petsc.getPC().setType("lu")
#     solver_petsc.setFromOptions()
#     solver_petsc.setUp()

#     b = problem.b.copy()
#     b.set(0)

#     for i_node, dof in enumerate(Ic):
#         b.setValue(dof * tdim + 2, p_nodes[i_node])
#     b.assemble()

#     u = problem.A.createVecRight()
#     solver_petsc.solve(b, u)

#     from dolfinx.fem import Function

#     u_fenics = Function(V)
#     u_fenics.x.array[:] = -u.array
#     u_fenics.x.scatter_forward()
#     u_fenics.name = "u"

#     with XDMFFile(mesh.comm, f"CubeP0traction_{frame}.xdmf", "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         xdmf.write_function(u_fenics, t=frame)


# ================================================================
# Contact computation at P1 element centers using Sc_elem
# ================================================================


for frame, x0_ in enumerate(x_pos):
    contact_center = np.array([x0_, 0.5])
    gap_nodes = (
        parabolic_indenter(
            Gamma_c_x[:, 0],
            Gamma_c_x[:, 1],
            contact_center[0],
            contact_center[1],
            Rindenter,
            np.full(len(Gamma_c_x[:, 2]), 1) - displ,
        )
        - Gamma_c_x[:, 2]
    )

    p_nodes, _, _ = CCG(Sc_dense, error_type, gap_nodes, max_iter, tolerance).solve()

    solver_petsc = PETSc.KSP().create(mesh.comm)
    solver_petsc.setOperators(problem.A)
    solver_petsc.setType("preonly")
    solver_petsc.getPC().setType("lu")
    solver_petsc.setFromOptions()
    solver_petsc.setUp()

    b = problem.b.copy()
    b.set(0)

    for i_node, dof in enumerate(Ic):
        b.setValue(dof * tdim + 2, p_nodes[i_node])
    b.assemble()

    u = problem.A.createVecRight()
    solver_petsc.solve(b, u)

    from dolfinx.fem import Function

    u_fenics = Function(V)
    u_fenics.x.array[:] = -u.array
    u_fenics.x.scatter_forward()
    u_fenics.name = "u"

    with XDMFFile(mesh.comm, f"CubeP1traction_{frame}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_fenics, t=frame)

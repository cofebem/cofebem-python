import numpy as np
import math
import time


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
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from ufl import (
    Measure,
    Identity,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    zero,
    FacetNormal,
    dx,
    ds,
)

from dolfinx.io import XDMFFile

from mpi4py import MPI
from petsc4py import PETSc
from typing import Callable, Optional, Union
from tqdm import tqdm
import logging
from scipy.sparse.linalg import splu, spsolve
from scipy.linalg import solve


# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

mesh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [80, 80, 20],
    CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

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

pressure = 100000.0

normal_ = FacetNormal(mesh)

traction = -pressure * normal_

ds = Measure("ds", domain=mesh)


def L(v):
    return inner(f_v, v) * dx + inner(traction, v) * ds


# -------------------------------------------------------------------------------------------------------
#  Set boundary conditions
# -------------------------------------------------------------------------------------------------------
tol = 1.0e-5


def gamma_u_selector(x):
    return np.isclose(x[2], 0, atol=tol)


gamma_u = locate_entities_boundary(
    mesh, dim=mesh.topology.dim - 1, marker=gamma_u_selector
)

u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)

Iu = locate_dofs_topological(V, entity_dim=mesh.topology.dim - 1, entities=gamma_u)

bc = dirichletbc(
    u0,
    dofs=Iu,
    V=V,
)

dof_coords = V.tabulate_dof_coordinates().reshape((-1, 3))

# tol = 1e-8


def find_dofs_at_point(point):
    return np.where(np.linalg.norm(dof_coords - point, axis=1) < tol)[0]


dofs_origin = find_dofs_at_point(np.array([0.0, 0.0, 0.0]))
dofs_origin = np.array(dofs_origin, dtype=np.int32)

dofs_100 = find_dofs_at_point(np.array([1.0, 0.0, 0.0]))
dofs_100 = np.array(dofs_100, dtype=np.int32)

dofs_010 = find_dofs_at_point(np.array([0.0, 1.0, 0.0]))
dofs_010 = np.array(dofs_010, dtype=np.int32)

bc_origin = dirichletbc(np.array([0.0, 0.0, 0.0], dtype=np.double), dofs_origin, V)

bc_100_y = dirichletbc(np.array([0.0, 0.0, 0.0], dtype=np.double), dofs_100[1::3], V)
bc_100_z = dirichletbc(np.array([0.0, 0.0, 0.0], dtype=np.double), dofs_100[2::3], V)

bc_010_z = dirichletbc(np.array([0.0, 0.0, 0.0], dtype=np.double), dofs_010[2::3], V)

bcs = [bc_origin, bc_100_y, bc_100_z, bc_010_z]
# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------
problem = LinearProblem(
    a=a(u, v), L=L(v), bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

u = problem.solve()

problem.A.assemble()

# -------------------------------------------------------------------------------------------------------
#  Define contact region GammaC
# -------------------------------------------------------------------------------------------------------


def Gamma_c_selector(x):
    return np.isclose(x[2], 1, atol=tol)


Gamma_c = locate_entities_boundary(
    mesh, dim=mesh.topology.dim - 1, marker=Gamma_c_selector
)

Ic = locate_dofs_topological(V, mesh.topology.dim - 1, Gamma_c)

# -------------------------------------------------------------------------------------------------------
#  Sc by Direct Sampling
# -------------------------------------------------------------------------------------------------------


def Sc_direct_sampling(A, b, comm, gdim, Ic, full_Sc=False, single_direction=0):
    f_magnitude = 1e9

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setFromOptions()
    solver.setUp()

    b = b.copy()
    uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)

    n_c = len(Ic)

    if full_Sc:
        full_dofs = np.array(
            [vertex * gdim + comp for vertex in Ic for comp in range(gdim)],
            dtype=np.int32,
        )
        Sc = np.zeros((gdim * n_c, gdim * n_c), dtype=PETSc.ScalarType)
        for i, dof_applied in enumerate(
            tqdm(full_dofs, desc="Computing Contact Compliance Matrix", unit="it")
        ):
            b.set(0)
            b.setValue(
                dof_applied,
                f_magnitude,
            )
            b.assemble()

            solver.solve(b, uh)

            Sc[i, :] = uh.array[full_dofs] / f_magnitude

    else:
        selected_dofs = Ic * gdim + single_direction
        Sc = np.zeros((n_c, n_c), dtype=PETSc.ScalarType)
        for i, dof_applied in enumerate(
            tqdm(selected_dofs, desc="Computing Contact Compliance Matrix", unit="it")
        ):
            b.set(0)
            b.setValue(
                dof_applied,
                f_magnitude,
            )
            b.assemble()

            solver.solve(b, uh)

            Sc[i, :] = uh.array[selected_dofs] / f_magnitude

    return Sc


Sc_sampling = Sc_direct_sampling(
    problem.A, problem.b, mesh.comm, mesh.topology.dim, Ic, full_Sc=False
)

print(f"Sc sampling \n")
print(Sc_sampling.shape)

# -------------------------------------------------------------------------------------------------------
#  Sc by Collocation BEM
# -------------------------------------------------------------------------------------------------------

from cofebem.bem.fundamental_solutions import kelvin_G, kelvin_H
from cofebem.bem.integrate import integrate

inv = np.linalg.inv

tdim = mesh.topology.dim
fdim = tdim - 1

mesh.topology.create_connectivity(fdim, 0)
conn = mesh.topology.connectivity(fdim, 0)


Gamma = exterior_facet_indices(mesh.topology)

I_Gamma = locate_dofs_topological(V, fdim, Gamma)
n_collocs = len(I_Gamma)

mapping_dofs = {dof: i for i, dof in enumerate(I_Gamma)}

Gamma_x = mesh.geometry.x[I_Gamma]
mesh_center = np.mean(Gamma_x, axis=0)  # works only if the mesh has a convex geometry

G = np.zeros((tdim * n_collocs, tdim * n_collocs))
H = np.zeros((tdim * n_collocs, tdim * n_collocs))

n_gauss = 7

for i, x_c_id in tqdm(
    enumerate(I_Gamma),
    totalComplementarity=n_collocs,
    desc="Assembling global matrices",
):

    x_c = mesh.geometry.x[x_c_id]

    for element in Gamma:
        nodes_ids = list(conn.links(element))
        nodes_ids[2], nodes_ids[3] = nodes_ids[3], nodes_ids[2]
        singular = x_c_id in nodes_ids

        n1 = mesh.geometry.x[nodes_ids[0]]
        n2 = mesh.geometry.x[nodes_ids[1]]
        n3 = mesh.geometry.x[nodes_ids[2]]
        n4 = mesh.geometry.x[nodes_ids[3]]

        nodes = np.vstack((n1, n2, n3, n4))

        normal = np.cross(n2 - n1, n3 - n1)
        normal = normal / np.linalg.norm(normal)
        centroid = (n1 + n2 + n3 + n4) / 4.0
        if np.dot(normal, centroid - mesh_center) < 0:
            normal = -normal

        for local_index, global_node in enumerate(nodes_ids):
            Val_G = integrate(
                kelvin_G, x_c, normal, nodes, mu, nu, singular, n_gauss, local_index
            )
            Val_H = integrate(
                kelvin_H, x_c, normal, nodes, mu, nu, singular, n_gauss, local_index
            )

            if global_node in mapping_dofs:
                j = mapping_dofs[global_node]

                row_start = tdim * i
                row_end = row_start + tdim
                col_start = tdim * j
                col_end = col_start + tdim

                G[row_start:row_end, col_start:col_end] += Val_G
                H[row_start:row_end, col_start:col_end] += Val_H

    row_start = tdim * i
    row_end = row_start + tdim
    H[row_start:row_end, row_start:row_end] += 0.5 * np.eye(tdim)


print("Assembled Global Matrix G:")
print(G)
print("\nAssembled Global Matrix H:")
print(H)


I_uc = np.concatenate((Ic, Iu))
It = np.setdiff1d(I_Gamma, I_uc)

Iu_ = np.array([mapping_dofs[dof] for dof in Iu], dtype=np.int32)
Iu_ = tdim * Iu_ + 2
Ic_ = np.array([mapping_dofs[dof] for dof in Ic], dtype=np.int32)
Ic_ = tdim * Ic + 2
It_ = np.array([mapping_dofs[dof] for dof in It], dtype=np.int32)
It_ = tdim * It_ + 2

H_ut = H[np.ix_(Iu_, It_)]
H_uc = H[np.ix_(Iu_, Ic_)]
H_tt = H[np.ix_(It_, It_)]
H_tc = H[np.ix_(It_, Ic_)]
H_ct = H[np.ix_(Ic_, It_)]
H_cc = H[np.ix_(Ic_, Ic_)]

G_uu = G[np.ix_(Iu_, Iu_)]
G_tu = G[np.ix_(It_, Iu_)]
G_cu = G[np.ix_(Ic_, Iu_)]

A = np.block(
    [
        [H_ut, H_uc, -G_uu],
        [H_tt, H_tc, -G_tu],
        [H_ct, H_cc, -G_cu],
    ]
)

G_uc = G[np.ix_(Iu_, Ic_)]
G_tc = G[np.ix_(It_, Ic_)]
G_cc = G[np.ix_(Ic_, Ic_)]

Sc = inv(
    (H_cc - G_cu @ inv(G_uu) @ H_uc)
    - (H_ct - G_cu @ inv(G_uu) @ H_ut)
    @ inv(H_tt - G_tu @ inv(G_uu) @ H_ut)
    @ (H_tc - G_tu @ inv(G_uu) @ H_uc)
) @ (
    (G_cc - G_cu @ inv(G_uu) @ G_uc)
    - (H_ct - G_cu @ inv(G_uu) @ H_ut)
    @ inv(H_tt - G_tu @ inv(G_uu) @ H_ut)
    @ (G_tc - G_tu @ inv(G_uu) @ G_uc)
)

print(f"Sc = \n")
print(Sc)
print(f"Sc_sampling = \n")
print(Sc_sampling)
print(f" error = {np.linalg.norm(Sc_sampling- Sc)/np.linalg.norm(Sc_sampling)}")

###################################### TOY PROBLEM ###########################################


f_magnitude = 7e7

solver_ = PETSc.KSP().create(mesh.comm)
solver_.setOperators(problem.A)
solver_.setType("preonly")
solver_.getPC().setType("lu")
solver_.setFromOptions()
solver_.setUp()

b = problem.b.copy()
b.set(0)
# for i in range(4):
#     b.setValue(
#         tdim * i + 2,
#         f_magnitude,
#     )
for i in range(4):
    b.setValue(
        tdim * (i + 4) + 2,
        -f_magnitude,
    )
# b.setValue(3 * 4 + 2, -f_magnitude)

b.assemble()

u_FEM = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)


solver_.solve(b, u_FEM)

u_FEM_array = u_FEM.array


u_fenics_FEM = Function(V)


u_fenics_FEM.x.array[:] = u_FEM_array
u_fenics_FEM.x.scatter_forward()

u_fenics_FEM.name = "u"

with XDMFFile(MPI.COMM_WORLD, f"displacement_FEM.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_fenics_FEM)

##############################
t = np.zeros((tdim * n_collocs,))
for i in range(4):
    t[tdim * i + 2] = f_magnitude
for i in range(4):
    t[tdim * (i + 4) + 2] = -f_magnitude

u_BEM = inv(H) @ G @ t


u_fenics_BEM = Function(V)


u_fenics_BEM.x.array[:] = u_BEM
u_fenics_BEM.x.scatter_forward()

u_fenics_BEM.name = "u"

with XDMFFile(MPI.COMM_WORLD, f"displacement_BEM.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_fenics_BEM)


print(u_FEM_array)
print(u_BEM)
print(f"error <= {np.linalg.norm(u_FEM_array-u_BEM)/np.linalg.norm(u_FEM_array)}")

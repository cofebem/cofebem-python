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

from dolfinx.io import XDMFFile
import mpi4py.MPI as MPI

# from cofebem.mesh.hollow_cylinder import generate_hollow_cylinder
from cofebem.mesh.wrinkly_tyre import generate_wrinkly_tyre

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------

r_inner = 2
r_outer = 5

nr = 100
nt = int((r_inner + r_outer) * nr * np.pi / (r_outer - r_inner))
nz = 20

generate_wrinkly_tyre(
    nr, nt, nz, r_inner=r_inner, r_outer=r_outer, A=0.2, B=0.07, k_r=10, k_theta=4
)

with XDMFFile(MPI.COMM_WORLD, "wrinkly_tyre.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

tdim = mesh.topology.dim
fdim = tdim - 1

E = 1.0
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
#  Construct S_c for a symmetric domain
# -------------------------------------------------------------------------------------------------------
angle_tol = 1.0e-8
tol = 1.0e-8

# def Gamma_c_selector(x):
#     return np.isclose(x[2], 1, atol=tol)


def Gamma_c_selector(x):
    z = x[2]
    return (z >= 0.8 - tol) & (z <= 10.0 + tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)


# def reference_line_selector(x):
#     return np.isclose(x[1], 0.0, atol=tol) & (x[2] >= 0.99 - tol) & (x[2] <= 10.0 + tol)


# Gamma_c_ref = locate_entities_boundary(mesh, fdim, reference_line_selector)

# Ic_ref = locate_dofs_geometrical(V, reference_line_selector)


Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

# angles = (np.arctan2(Gamma_c_x[:, 1], Gamma_c_x[:, 0]) + 2 * np.pi) % (2 * np.pi)
# order_angle = np.argsort(angles)
# Ic_sorted = Ic[order_angle]


# Ic_sorted = Ic_sorted.reshape(nt, nr + 1)

# Gamma_c_x_segments = mesh.geometry.x[Ic_sorted].reshape(nt, nr + 1, tdim)
# radii = np.sqrt(Gamma_c_x_segments[:, :, 0] ** 2 + Gamma_c_x_segments[:, :, 1] ** 2)

# order_radius = np.argsort(radii, axis=1)
# Ic_sorted = np.take_along_axis(Ic_sorted, order_radius, axis=1).flatten()


# def compute_Sc_ref(A, b, Ic_ref, Ic_sorted, tdim, comm, f_magnitude=1.0):
#     solver = PETSc.KSP().create(comm)
#     solver.setOperators(A)
#     solver.setType("preonly")
#     solver.getPC().setType("lu")

#     uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)

#     dofs_ref = np.array([node * tdim + 2 for node in Ic_ref])
#     dofs_contact = np.array([node * tdim + 2 for node in Ic_sorted])

#     Sc_ref = np.zeros((len(dofs_ref), len(dofs_contact)))

#     for i, dof in tqdm(enumerate(dofs_ref), desc="Computing Sc_ref"):
#         b.zeroEntries()
#         b.setValue(dof, f_magnitude)
#         b.assemble()

#         uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)
#         solver.solve(b, uh)

#         Sc_ref[i, :] = uh.array[dofs_contact] / f_magnitude

#     return Sc_ref


# def construct_Sc_from_Sc_ref(Sc_ref, nt, Ic):
#     n_ref, nc = Sc_ref.shape
#     assert nc == n_ref * nt, "Mismatch between segments and contact nodes."

#     Sc = np.zeros((nc, nc))

#     for i in tqdm(range(nt), desc="Computing Sc by symmetry"):
#         row_start = i * n_ref
#         row_end = (i + 1) * n_ref

#         shift = i * n_ref

#         Sc[row_start:row_end, :] = np.roll(Sc_ref, shift=shift, axis=1)

#     mapping = {dof: i for i, dof in enumerate(Ic_sorted)}

#     perm = np.array([mapping[dof] for dof in Ic])

#     # Sc_classic_unsorted = Sc[np.ix_(perm, perm)]

#     return Sc  # Sc_classic_unsorted


# Sc_ref = compute_Sc_ref(problem.A, problem.b, Ic_ref, Ic_sorted, tdim, mesh.comm)
# Sc = construct_Sc_from_Sc_ref(Sc_ref, nt, Ic)

# -------------------------------------------------------------------------------------------------------
#  Construct S_c classic
# -------------------------------------------------------------------------------------------------------


def Sc_direct_sampling(A, b, comm, tdim, Ic, full_Sc=False, single_direction=2):
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
            [vertex * tdim + comp for vertex in Ic for comp in range(tdim)],
            dtype=np.int32,
        )
        Sc = np.zeros((tdim * n_c, tdim * n_c), dtype=PETSc.ScalarType)
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
        selected_dofs = Ic * tdim + single_direction
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


Sc_classic = Sc_direct_sampling(
    problem.A, problem.b, mesh.comm, tdim, Ic, full_Sc=False
)


# # -------------------------------------------------------------------------------------------------------
# #  Contact Problem
# # -------------------------------------------------------------------------------------------------------


def constrained_CG(
    Sc,
    error_type,
    gap,
    max_iter,
    tolerance,
    pressure_factor=1e12,
    initial_pressure=None,
):
    error_history = np.zeros((max_iter, 3))
    ub = -gap
    # Warmed start does not work well
    if initial_pressure is not None:
        p = np.maximum(-gap, 0) * pressure_factor
    else:
        p = np.zeros_like(ub)
        p = np.maximum(-gap, 0) * pressure_factor

    w = np.inner(Sc, p) - ub
    # w -= np.mean(w) #new
    t = w
    t_ = np.zeros_like(w)
    d = 0
    error = 1
    error_ = 1
    for iter in range(max_iter):
        if iter > 0:
            t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
            t[p <= 0] = 0
        q = np.inner(Sc, t)
        tau = np.inner(w, t) / np.inner(t, q)
        p = p - tau * t
        p = np.maximum(p, 0)
        zero_pressure = np.where(p == 0)[0]
        penetration = np.where(w < 0)[0]
        set_I = np.intersect1d(zero_pressure, penetration)
        if len(set_I) == 0:
            d = 1
        else:
            d = 0
            p[set_I] -= tau * w[set_I]
        t_ = t

        w = np.inner(Sc, p) - ub
        nw = np.linalg.norm(w, 2)

        error_ = error
        displ_error = np.linalg.norm(w[p > 0], 2) / nw
        ort = np.abs(np.dot(w, p) / nw)

        if error_type == "displacement":
            error = displ_error
        elif error_type == "mix":
            error = np.sqrt(displ_error * ort)
        elif error_type == "nw":
            error = nw
            if abs((error - error_) / error_) < tolerance:
                error_history[iter, 0] = displ_error
                error_history[iter, 1] = abs((error - error_) / error_)
                error_history[iter, 2] = ort
                return p, np.inner(Sc, p), error_history[: iter + 1]
        error_history[iter, 0] = displ_error
        error_history[iter, 1] = error
        error_history[iter, 2] = ort
        if error < tolerance:
            break
    return p, np.inner(Sc, p), error_history[: iter + 1]


displ = 0.3
Rindenter = 10.0


# def _parabolic_indenter(x, y, x0, y0, R, z0):
#     if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > R:
#         return z0 + R
#     else:
#         return z0 + R - np.sqrt(R**2 - (x - x0) ** 2 - (y - y0) ** 2)


# parabolic_indenter = np.vectorize(_parabolic_indenter)


def _flat_indenter(x, y, x0, y0, R, z0):
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < R:
        return z0
    else:
        return z0 + 10.0


flat_indenter = np.vectorize(_flat_indenter)

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 40

ANIMATION = False
if ANIMATION == True:
    # # x_center = np.linspace(-7.0, 7.0, Nframes)
    # theta_c = np.linspace(0.0, 2 * np.pi, Nframes)
    # rc = 3.0
    # for frame, theta_c_ in enumerate(theta_c):
    #     contact_center = np.array([rc * np.cos(theta_c_), rc * np.sin(theta_c_)])
    #     gap = (
    #         flat_indenter(
    #             Gamma_c_x[:, 0],
    #             Gamma_c_x[:, 1],
    #             contact_center[0],
    #             contact_center[1],
    #             Rindenter,
    #             np.ones_like(Gamma_c_x[:, 2]) - displ,
    #         )
    #         - Gamma_c_x[:, 2]
    #     )

    contact_center = np.array([0.0, 0.0])
    z_start = 1.2
    z_end = 1.0 - displ
    Nframes = 40

    z_positions = np.linspace(z_start, z_end, Nframes)

    for frame, z0 in enumerate(z_positions):
        gap = (
            flat_indenter(
                Gamma_c_x[:, 0],
                Gamma_c_x[:, 1],
                contact_center[0],
                contact_center[1],
                Rindenter,
                np.full_like(Gamma_c_x[:, 2], z0),
            )
            - Gamma_c_x[:, 2]
        )
        penetrating_nodes = np.where(gap < 0)[0]

        if frame == 0:
            p, _, _ = constrained_CG(Sc_classic, error_type, gap, max_iter, tolerance)
        else:
            p, _, _ = constrained_CG(Sc_classic, error_type, gap, max_iter, tolerance)

        solver_petsc = PETSc.KSP().create(mesh.comm)
        solver_petsc.setOperators(problem.A)
        solver_petsc.setType("preonly")
        solver_petsc.getPC().setType("lu")
        solver_petsc.setFromOptions()
        solver_petsc.setUp()

        b = problem.b.copy()
        u = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)

        b.set(0)
        for i, dof in enumerate(Ic):
            b.setValue(dof * tdim + 2, p[i])
            b.assemble()

        solver_petsc.solve(b, u)
        from dolfinx.fem import Function

        u_fenics = Function(V)

        u_fenics.x.array[:] = -u.array
        u_fenics.x.scatter_forward()  # for parallel ghost updates

        u_fenics.name = "u"

        with XDMFFile(MPI.COMM_WORLD, f"wtyre_disp_{frame}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(u_fenics, t=frame)

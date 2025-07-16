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


classic_times = []
on_the_fly_times = []
hs = []
e = np.linspace(1, 50, 20, dtype=np.int32)
# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------
# for e_ in e:
#     h = 1 / e_
#     hs.append(h)
mesh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [30, 30, 1],
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


def L(v):
    return inner(f_v, v) * dx


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

# -------------------------------------------------------------------------------------------------------
#  Setup the Problem
# -------------------------------------------------------------------------------------------------------
problem = LinearProblem(
    a=a(u, v),
    L=L(v),
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

u = problem.solve()

# -------------------------------------------------------------------------------------------------------
#  Define contact region GammaC
# -------------------------------------------------------------------------------------------------------


def Gamma_c_selector(x):
    return np.isclose(x[2], 1, atol=tol)


Gamma_c = locate_entities_boundary(
    mesh, dim=mesh.topology.dim - 1, marker=Gamma_c_selector
)

Ic = locate_dofs_topological(V, mesh.topology.dim - 1, Gamma_c)

Gamma_c_x = mesh.geometry.x[Ic]

# -------------------------------------------------------------------------------------------------------
#  Sc by Direct Sampling
# -------------------------------------------------------------------------------------------------------

problem.A.assemble()
problem.b.assemble()


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


Sc_classic = Sc_direct_sampling(
    problem.A, problem.b, mesh.comm, mesh.topology.dim, Ic, full_Sc=False
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


###### CCG on the fly############
def constrained_CG2(
    Sc,
    error_type,
    gap,
    Ic_new_loc,
    max_iter,
    tolerance,
    pressure_factor=1e12,
    initial_pressure=None,
):
    error_history = np.zeros((max_iter, 3))
    ub = -gap[Ic_new_loc]
    # Warmed start does not work well
    if initial_pressure is not None:
        p = np.maximum(-gap[Ic_new_loc], 0) * pressure_factor
    else:
        p = np.zeros_like(ub)
        p = np.maximum(-gap[Ic_new_loc], 0) * pressure_factor

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


displ = 0.15
Rindenter = 1

tdim = mesh.topology.dim


def _parabolic_indenter(x, y, x0, y0, R, z0):
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > R:
        return z0 + R
    else:
        return z0 + R - np.sqrt(R**2 - (x - x0) ** 2 - (y - y0) ** 2)


parabolic_indenter = np.vectorize(_parabolic_indenter)

max_iter = 1000
tolerance = 1e-5
error_type = "nw"
pfactor = 1e8
Nframes = 50

####################################################Sc_classic###########################################
classic_start = time.perf_counter()
ANIMATION = True
if ANIMATION == True:
    x_center = np.linspace(-1.0, 1.2, Nframes)
    for frame, xc in enumerate(x_center):
        contact_center = np.array([xc, 0.5])
        gap = (
            parabolic_indenter(
                Gamma_c_x[:, 0],
                Gamma_c_x[:, 1],
                contact_center[0],
                contact_center[1],
                Rindenter,
                np.ones_like(Gamma_c_x[:, 2]) - displ,
            )
            - Gamma_c_x[:, 2]
        )
        penetrating_nodes = np.where(gap < 0)[0]

        if frame == 0:
            p, _, _ = constrained_CG(Sc_classic, error_type, gap, max_iter, tolerance)
        else:
            p, _, _ = constrained_CG(Sc_classic, error_type, gap, max_iter, tolerance)

        # X_, Y_, Z_ = (
        #     Gamma_c_x[:, 0],
        #     Gamma_c_x[:, 1],
        #     Gamma_c_x[:, 2] - u,
        # )
        # X_ = X_.reshape(-1, 1)
        # Y_ = Y_.reshape(-1, 1)
        # Z_ = Z_.reshape(-1, 1)
        # disp_ = u.reshape(-1, 1)
        # p_ = p.reshape(-1, 1)
        # output = np.hstack((X_, Y_, Z_, disp_))

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

        # Create an empty function in the same space
        u_fenics = Function(V)

        # Copy the PETSc Vec into the Function
        u_fenics.x.array[:] = -u.array
        u_fenics.x.scatter_forward()  # for parallel ghost updates

        u_fenics.name = "u"
        # Now write
        with XDMFFile(MPI.COMM_WORLD, f"Ironing_{frame}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(u_fenics, t=frame)  # or omit t if you prefer

        # np.savetxt(
        #     fname=f"tyre_disp_{frame}.csv",
        #     X=u.array,
        #     header="u",
        # )

classic_end = time.perf_counter()
classic_duration = classic_end - classic_start

print(f"Sc classic time= {classic_duration}")
#     #####################################Sc on the fly################################################################
#     on_the_fly_start = time.perf_counter()
#     mapping_Ic = {dof: i for i, dof in enumerate(Ic)}

#     ANIMATION = True
#     if ANIMATION == True:
#         x_center = np.linspace(-7.0, 7.0, Nframes)
#         for frame, xc in enumerate(x_center):
#             contact_center = np.array([xc, 0.5])
#             gap = (
#                 parabolic_indenter(
#                     Gamma_c_x[:, 0],
#                     Gamma_c_x[:, 1],
#                     contact_center[0],
#                     contact_center[1],
#                     Rindenter,
#                     np.ones_like(Gamma_c_x[:, 2]) - displ,
#                 )
#                 - Gamma_c_x[:, 2]
#             )
#             penetrating_nodes = np.where(gap < 0)[0]
#             Ic_new = Ic[penetrating_nodes]
#             Sc_on_the_fly = Sc_direct_sampling(
#                 problem.A,
#                 problem.b,
#                 mesh.comm,
#                 mesh.topology.dim,
#                 Ic_new,
#                 full_Sc=False,
#             )

#             Ic_new_loc = np.array([mapping_Ic[dof] for dof in Ic_new], dtype=np.int32)
#             if frame == 0:
#                 p, _, _ = constrained_CG2(
#                     Sc_on_the_fly, error_type, gap, Ic_new_loc, max_iter, tolerance
#                 )
#             else:
#                 p, _, _ = constrained_CG2(
#                     Sc_on_the_fly, error_type, gap, Ic_new_loc, max_iter, tolerance
#                 )

#             # X_, Y_, Z_ = (
#             #     Gamma_c_x[:, 0],
#             #     Gamma_c_x[:, 1],
#             #     Gamma_c_x[:, 2] - u,
#             # )
#             # X_ = X_.reshape(-1, 1)
#             # Y_ = Y_.reshape(-1, 1)
#             # Z_ = Z_.reshape(-1, 1)
#             # disp_ = u.reshape(-1, 1)
#             # p_ = p.reshape(-1, 1)
#             # output = np.hstack((X_, Y_, Z_, disp_))
#             if len(Sc_on_the_fly) == 0:
#                 continue
#             solver_petsc = PETSc.KSP().create(mesh.comm)
#             solver_petsc.setOperators(problem.A)
#             solver_petsc.setType("preonly")
#             solver_petsc.getPC().setType("lu")
#             solver_petsc.setFromOptions()
#             solver_petsc.setUp()

#             b = problem.b.copy()
#             u = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)

#             b.set(0)
#             for i, dof in enumerate(Ic_new):
#                 b.setValue(dof * tdim + 2, p[i])
#                 b.assemble()

#             solver_petsc.solve(b, u)
#             from dolfinx.fem import Function

#             # Create an empty function in the same space
#             u_fenics = Function(V)

#             # Copy the PETSc Vec into the Function
#             u_fenics.x.array[:] = -u.array
#             u_fenics.x.scatter_forward()  # for parallel ghost updates

#             u_fenics.name = "u"
#             # Now write
#             with XDMFFile(MPI.COMM_WORLD, f"displacement_loc{frame}.xdmf", "w") as xdmf:
#                 xdmf.write_mesh(mesh)
#                 xdmf.write_function(u_fenics, t=frame)  # or omit t if you prefer

#             # np.savetxt(
#             #     fname=f"tyre_disp_{frame}.csv",
#             #     X=u.array,
#             #     header="u",
#             # )

#     on_the_fly_end = time.perf_counter()
#     on_the_fly_duration = on_the_fly_end - on_the_fly_start
#     print(f"Sc on the fly time = {on_the_fly_duration}")

#     classic_times.append(classic_duration)
#     on_the_fly_times.append(on_the_fly_duration)

# import matplotlib.pyplot as plt

# # x_lin = np.linspace(100, 1600, 10)
# # Shifted reference power curves to start at the same point as the first data point
# on_the_fly_times_times = np.asarray(on_the_fly_times)
# classic_times = np.asarray(classic_times)
# hs = np.asarray(hs)
# # shift_value = on_the_fly_times[0]  # classic_times[0]
# # power_1 = x_lin / x_lin[0] * shift_value
# # power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# # power_1 = x_lin / x_lin[0] * shift_value
# # power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(4, 3))

# # Plot the data
# ax.plot(hs, classic_times, "o-", label="Classic Sc", markersize=6, linewidth=2)
# ax.plot(
#     hs,
#     on_the_fly_times,
#     "s-",
#     label="Sc on the fly",
#     markersize=6,
#     linewidth=2,
# )

# # ax.plot(x_lin, power_1, "--", color="black")  # label="O(N)")
# # ax.plot(
# #     x_lin, power_2, "-.", color="black"
# # )  # label="O(N²)")  # Annotate power curves

# # ax.text(
# #     dofs[-1],
# #     power_1[-1],
# #     "O(N)",
# #     fontsize=8,
# #     color="black",
# #     verticalalignment="bottom",
# #     horizontalalignment="right",
# # )
# # ax.text(
# #     dofs[-1],
# #     power_2[-1],
# #     "O(N²)",
# #     fontsize=8,
# #     color="black",
# #     verticalalignment="bottom",
# #     horizontalalignment="right",
# # )

# # Logarithmic scale for better readability
# ax.set_xscale("log")
# ax.set_yscale("log")

# # Labels and title
# ax.set_xlabel("h (element size)", fontsize=10)
# ax.set_ylabel("CPU Time (s)", fontsize=10)

# ax.set_title("Comparison of Sc Classic and Sc on the fly", fontsize=16)

# # Grid and legend
# ax.grid(True, which="both", linestyle="--", linewidth=0.5)
# ax.legend(fontsize=8, loc="upper left")

# # Improve layout
# plt.tight_layout()

# fig.savefig("Sc_on_the_fly.png", format="png")

# # Show the plot
# plt.show()

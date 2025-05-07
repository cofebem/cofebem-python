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

from cofebem.contact.lcp_solvers.lemke import lemkelcp


ccg_times = []
lemke_times = []
errors = []


hs = []
e = np.linspace(10, 30, 20, dtype=np.int32)

i = 0

for e_ in e:
    h = 1 / e_
    hs.append(h)
    # -------------------------------------------------------------------------------------------------------
    #  Mesh and material parameters
    # -------------------------------------------------------------------------------------------------------
    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [e_, e_, 1],
        CellType.hexahedron,
        ghost_mode=GhostMode.shared_facet,
    )

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
        a=a(u, v),
        L=L(v),
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    problem.solve()

    # -------------------------------------------------------------------------------------------------------
    #  Construct S_c classic
    # -------------------------------------------------------------------------------------------------------

    angle_tol = 1.0e-8
    tol = 1.0e-8

    def Gamma_c_selector(x):
        return np.isclose(x[2], 1, atol=tol)

    Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
    Ic = locate_dofs_topological(V, fdim, Gamma_c)

    Gamma_c_x = mesh.geometry.x[Ic]

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
                tqdm(
                    selected_dofs, desc="Computing Contact Compliance Matrix", unit="it"
                )
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

    classic_start = time.perf_counter()

    Sc = Sc_direct_sampling(problem.A, problem.b, mesh.comm, tdim, Ic, full_Sc=False)

    classic_end = time.perf_counter()

    classic_duration = classic_end - classic_start

    print(f"classic duration = {classic_duration}")

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

    displ = 0.5
    Rindenter = 2.0

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
    Nframes = 20

    contact_center = np.array([0, 0])
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
    ccg_start = time.perf_counter()
    p_ccg, _, _ = constrained_CG(Sc, error_type, gap, max_iter, tolerance)
    ccg_end = time.perf_counter()
    ccg_duration = ccg_end - ccg_start
    ccg_times.append(ccg_duration)

    lemke_start = time.perf_counter()
    p_lemke, _, _ = lemkelcp(Sc, gap, max_iter)
    lemke_end = time.perf_counter()
    lemke_duration = lemke_end - lemke_start
    lemke_times.append(lemke_duration)

    error = np.linalg.norm(p_lemke - p_ccg) / np.linalg.norm(p_lemke)
    errors.append(error)


import matplotlib.pyplot as plt

ccg_times = np.asarray(ccg_times)
lemke_times = np.asarray(lemke_times)
errors = np.asarray(errors)
hs = np.asarray(hs)
# shift_value = on_the_fly_times[0]  # classic_times[0]
# power_1 = x_lin / x_lin[0] * shift_value
# power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# power_1 = x_lin / x_lin[0] * shift_value
# power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# Create the figure and axis
fig, ax = plt.subplots(figsize=(4, 3))

# Plot the data
ax.plot(hs, lemke_times, "o-", label="Lemke", markersize=6, linewidth=2)
ax.plot(
    hs,
    ccg_times,
    "s-",
    label="CCG",
    markersize=6,
    linewidth=2,
)

# ax.plot(x_lin, power_1, "--", color="black")  # label="O(N)")
# ax.plot(
#     x_lin, power_2, "-.", color="black"
# )  # label="O(N²)")  # Annotate power curves

# ax.text(
#     dofs[-1],
#     power_1[-1],
#     "O(N)",
#     fontsize=8,
#     color="black",
#     verticalalignment="bottom",
#     horizontalalignment="right",
# )
# ax.text(
#     dofs[-1],
#     power_2[-1],
#     "O(N²)",
#     fontsize=8,
#     color="black",
#     verticalalignment="bottom",
#     horizontalalignment="right",
# )

# Logarithmic scale for better readability
ax.set_xscale("log")
ax.set_yscale("log")

# Labels and title
ax.set_xlabel("h (element size)", fontsize=10)
ax.set_ylabel("CPU Time (s)", fontsize=10)

ax.set_title("Comparison of Lemke and CCG algorithms complexities", fontsize=16)

# Grid and legend
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(fontsize=8, loc="upper left")

# Improve layout
plt.tight_layout()

fig.savefig("lemke_vs_ccg.png", format="png")

fig1, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(
    hs,
    errors,
    "s-",
    label="Relative error",
    markersize=6,
    linewidth=2,
)


ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.set_xlabel("h (element size)", fontsize=10)
ax1.set_ylabel("CPU Time (s)", fontsize=10)

ax1.set_title("Comparison of Lemke and CCG algorithms (relative error)", fontsize=16)

ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
ax1.legend(fontsize=8, loc="upper left")


fig1.savefig("lemke_vs_ccg_error.png", format="png")
plt.tight_layout()

# Show the plot
plt.show()

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

from cofebem.contact.Sc import Sc
from cofebem.contact.rigid_indenters import parabolic
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp


ccg_times = []
lemke_times = []
errors = []


hs = []
nl = np.linspace(5, 30, 25, dtype=np.int32)


for nl in nl:

    # -------------------------------------------------------------------------------------------------------
    #  Mesh and material parameters
    # -------------------------------------------------------------------------------------------------------
    nx = ny = nl
    nz = 5

    n_elems = nl**2
    h = 1 / n_elems
    hs.append(h)

    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
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

    tol = 1.0e-7

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
    #  Gamma_c : Contact Region
    # -------------------------------------------------------------------------------------------------------

    def Gamma_c_selector(x):
        return np.isclose(x[2], 1, atol=tol)

    Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
    Ic = locate_dofs_topological(V, fdim, Gamma_c)
    Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

    # -------------------------------------------------------------------------------------------------------
    #  Sc construction: Contact compliance matrix
    # -------------------------------------------------------------------------------------------------------

    Sc_ = Sc(problem.A, problem.b, tdim, Ic).by_sampling()

    # # -------------------------------------------------------------------------------------------------------
    # #  Contact Problem
    # # -------------------------------------------------------------------------------------------------------

    displ = 0.3
    Rindenter = 0.3

    max_iter = 1000
    tolerance = 1e-8
    error_type = "nw"
    pfactor = 1e8

    contact_center = np.array([0.5, 0.5])

    gap = (
        parabolic(
            Gamma_c_x[:, 0],
            Gamma_c_x[:, 1],
            contact_center[0],
            contact_center[1],
            Rindenter,
            np.ones_like(Gamma_c_x[:, 2]) - displ,
        )
        - Gamma_c_x[:, 2]
    )

    ccg_start = time.perf_counter()
    p_ccg, _, _ = CCG(Sc_, error_type, gap, max_iter, tolerance).solve()
    ccg_end = time.perf_counter()
    ccg_duration = ccg_end - ccg_start
    ccg_times.append(ccg_duration)

    lemke_start = time.perf_counter()
    p_lemke, _, _ = lemkelcp(Sc_, gap, max_iter)
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
fig, ax = plt.subplots(figsize=(6, 4))

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

fig.savefig("lemke_vs_ccg.pdf", format="pdf")

fig1, ax1 = plt.subplots(figsize=(6, 4))

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

ax1.set_xlabel("h (element size)", fontsize=14)
ax1.set_ylabel("CPU Time (s)", fontsize=14)

ax1.set_title("Comparison of Lemke and CCG algorithms (relative error)", fontsize=16)

ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
ax1.legend(fontsize=8, loc="upper left")


fig1.savefig("lemke_vs_ccg_error.pdf", format="pdf")
plt.tight_layout()

# Show the plot
plt.show()

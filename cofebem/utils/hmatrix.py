import numpy as np
from numba import njit
from time import perf_counter

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

from cofebem.mesh.hollow_cylinder import hollow_cylinder
from cofebem.utils.clustering.cluster_tree import ClusterTree
from cofebem.utils.aca import aca
from cofebem.hmatrices.hmatrix import HMatrix
from cofebem.hmatrices.low_rank_approx import truncated_svd

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------
nr = 8
nt = 70
nz = 4

r_inner = 1
r_outer = 5

hollow_cylinder(nr, nt, nz, r_inner, r_outer)

with XDMFFile(MPI.COMM_WORLD, "hex_hollow_cylinder.xdmf", "r") as xdmf:
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


def Gamma_c_selector(x):
    return np.isclose(x[2], 1, atol=tol)


Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
Ic = locate_dofs_topological(V, fdim, Gamma_c)


Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)


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


Sc = Sc_direct_sampling(problem.A, problem.b, mesh.comm, tdim, Ic, full_Sc=False)


# @njit
def MatVec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute b = A · x for
        A : shape (m, n)
        x : shape (n,) or (n, 1)
    Returns
    -------
    b : shape (m,)
    """
    m, n = A.shape

    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError("x must be (n,) or (n,1).")
        x_flat = x[:, 0]
    elif x.ndim == 1:
        x_flat = x
    else:
        raise ValueError("x must be (n,) or (n,1).")

    if x_flat.shape[0] != n:
        raise ValueError("Dimension mismatch: A is (m, n) but x length is not n.")

    b = np.empty(m, dtype=A.dtype)

    for i in range(m):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x_flat[j]
        b[i] = acc
        # acc = 0.0
    return b


# -------------------------------------------------------------------------------------------------------
# Matrix-Vector Product: H-matrix vs Classical Product
# -------------------------------------------------------------------------------------------------------

########################## Construct H-matrix S_c ###########################################3

n = Sc.shape[1]
x = np.random.rand(n)

# first run for jit
b_scratch = MatVec(Sc, x)


scratch_start = perf_counter()
scratch_prod = MatVec(Sc, x)
scratch_end = perf_counter()
scratch_duration = scratch_end - scratch_start

# -------------------------------------------------------------------------------------------------------
#  Complexity Comparison
# -------------------------------------------------------------------------------------------------------

h_matrix_times = []
classic_prod_times = []

levels = np.linspace(5, 30, 5)


for level in levels:
    level = int(level)

    max_leaf_size = n / level

    Sc_hmat = HMatrix(
        A=Sc,
        coords=Gamma_c_x,
        compress_func=truncated_svd,
        max_leaf_size=max_leaf_size,
        eta=1.5,
        tol=1e-6,
        max_rank=200,
    )

    Sc_hmat.print_summary()

    scratch_start = perf_counter()
    scratch_prod = MatVec(Sc, x)
    scratch_end = perf_counter()
    scratch_duration = scratch_end - scratch_start

    hmatrix_start = perf_counter()
    hmatrix_prod = Sc_hmat(x)
    hmatrix_end = perf_counter()
    hmatrix_duration = hmatrix_end - hmatrix_start

    h_matrix_times.append(hmatrix_duration)
    classic_prod_times.append(scratch_duration)


x_lin = np.linspace(5, 30, 5)

classic_prod_times = np.asarray(classic_prod_times)
h_matrix_times = np.asarray(h_matrix_times)
shift_value = classic_prod_times[0]  # classic_times[0]
power_1 = x_lin / x_lin[0] * shift_value
power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

# power_1 = x_lin / x_lin[0] * shift_value
# power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

fig, ax = plt.subplots(figsize=(4, 3))

# Plot the data
ax.plot(
    levels, classic_prod_times, "o-", label="Classic MatVec", markersize=6, linewidth=2
)
ax.plot(
    levels,
    h_matrix_times,
    "s-",
    label="HMatVec",
    markersize=6,
    linewidth=2,
)

ax.plot(x_lin, power_1, "--", color="black")  # label="O(N)")
ax.plot(x_lin, power_2, "-.", color="black")  # label="O(N²)")  # Annotate power curves

ax.text(
    levels[-1],
    power_1[-1],
    "O(N)",
    fontsize=8,
    color="black",
    verticalalignment="bottom",
    horizontalalignment="right",
)
ax.text(
    levels[-1],
    power_2[-1],
    "O(N²)",
    fontsize=8,
    color="black",
    verticalalignment="bottom",
    horizontalalignment="right",
)

# Logarithmic scale for better readability
ax.set_xscale("log")
ax.set_yscale("log")

# Labels and title
ax.set_xlabel("Levels (numbers od subdomains)", fontsize=8)
ax.set_ylabel("CPU Time (s)", fontsize=8)

ax.set_title("Comparison of Classic and HMatrix MatVec Product", fontsize=16)

# Grid and legend
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(fontsize=8, loc="upper left")

# Improve layout
plt.tight_layout()

# fig.savefig("Sc_by_symmetry.png", format="png")

# Show the plot
plt.show()

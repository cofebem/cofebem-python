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
from cofebem.utils.clustering.cluster_tree import ClusterTree, H

from cofebem.hmatrices.hmatrix import HMatrix
from cofebem.hmatrices.low_rank_approx import truncated_svd

# -------------------------------------------------------------------------------------------------------
#  Mesh and material parameters
# -------------------------------------------------------------------------------------------------------
nr = 5
nt = 70
nz = 4

r_inner = 1
r_outer = 5

hollow_cylinder(nr, nt, nz, r_inner, r_outer)

with XDMFFile(MPI.COMM_WORLD, "hollow_cylinder.xdmf", "r") as xdmf:
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


def reference_line_selector(x):
    return (
        np.isclose(x[1], 0.0, atol=tol) & np.isclose(x[2], 1.0, atol=tol) & (x[0] > 0)
    )


Gamma_c_ref = locate_entities_boundary(mesh, fdim, reference_line_selector)

Ic_ref = locate_dofs_geometrical(V, reference_line_selector)


Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

angles = (np.arctan2(Gamma_c_x[:, 1], Gamma_c_x[:, 0]) + 2 * np.pi) % (2 * np.pi)
order_angle = np.argsort(angles)
Ic_sorted = Ic[order_angle]


Ic_sorted = Ic_sorted.reshape(nt, nr + 1)

Gamma_c_x_segments = mesh.geometry.x[Ic_sorted].reshape(nt, nr + 1, tdim)
radii = np.sqrt(Gamma_c_x_segments[:, :, 0] ** 2 + Gamma_c_x_segments[:, :, 1] ** 2)

order_radius = np.argsort(radii, axis=1)
Ic_sorted = np.take_along_axis(Ic_sorted, order_radius, axis=1).flatten()


def compute_Sc_ref(A, b, Ic_ref, Ic_sorted, tdim, comm, f_magnitude=1.0):
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")

    uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)

    dofs_ref = np.array([node * tdim + 2 for node in Ic_ref])
    dofs_contact = np.array([node * tdim + 2 for node in Ic_sorted])

    Sc_ref = np.zeros((len(dofs_ref), len(dofs_contact)))

    for i, dof in tqdm(enumerate(dofs_ref), desc="Computing Sc_ref"):
        b.zeroEntries()
        b.setValue(dof, f_magnitude)
        b.assemble()

        uh = PETSc.Vec().createMPI(b.getSize(), comm=comm)
        solver.solve(b, uh)

        Sc_ref[i, :] = uh.array[dofs_contact] / f_magnitude

    return Sc_ref


def construct_Sc_from_Sc_ref(Sc_ref, nt, Ic):
    n_ref, nc = Sc_ref.shape
    assert nc == n_ref * nt, "Mismatch between segments and contact nodes."

    Sc = np.zeros((nc, nc))

    for i in tqdm(range(nt), desc="Computing Sc by symmetry"):
        row_start = i * n_ref
        row_end = (i + 1) * n_ref

        shift = i * n_ref

        Sc[row_start:row_end, :] = np.roll(Sc_ref, shift=shift, axis=1)

    mapping = {dof: i for i, dof in enumerate(Ic_sorted)}

    perm = np.array([mapping[dof] for dof in Ic])

    Sc_classic_unsorted = Sc[np.ix_(perm, perm)]

    return Sc_classic_unsorted  # Sc #


Sc_ref = compute_Sc_ref(problem.A, problem.b, Ic_ref, Ic_sorted, tdim, mesh.comm)
Sc = construct_Sc_from_Sc_ref(Sc_ref, nt, Ic)


########################## Construct H-matrix S_c ###########################################3

Sc_hmat = HMatrix(
    A=Sc,
    coords=mesh.geometry.x[Ic_sorted].reshape(-1, tdim),  # Gamma_c_x,  #
    compress_func=truncated_svd,
    max_leaf_size=10,
    eta=2.0,
    tol=1e-6,
    max_rank=200,
)

Sc_hmat.print_summary()

x = 1000 * np.random.rand(Sc.shape[0])

y_dense = Sc @ x
y_hmat = Sc_hmat(x)

error = np.linalg.norm(y_hmat - y_dense)  # / np.linalg.norm(y_dense)
print(f"[TEST] Relative error: {error:.2e}")

Sc_hmat.visualize()
Sc_hmat_dense = Sc_hmat.to_dense()


error_mat = np.linalg.norm(Sc - Sc_hmat_dense) / np.linalg.norm(Sc)
print(f"[TEST] Relative error: {error_mat:.2e}")


def is_symmetric_fast(A: np.ndarray, tol: float = 1e-8) -> bool:
    i_upper = np.triu_indices_from(A, k=1)
    return np.allclose(A[i_upper], A.T[i_upper], atol=tol, rtol=0)


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

    t = w
    t_ = np.zeros_like(w)
    d = 0
    error = 1
    error_ = 1
    for iter in tqdm(range(max_iter), desc="Solving the the contact"):
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


def constrained_CG_hmat(
    Sc_hmat,
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

    w = Sc_hmat(p) - ub

    t = w
    t_ = np.zeros_like(w)
    d = 0
    error = 1
    error_ = 1
    for iter in tqdm(range(max_iter), desc="Solving the the contact"):
        if iter > 0:
            t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
            t[p <= 0] = 0
        q = Sc_hmat(t)
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

        w = Sc_hmat(p)
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
                return p, Sc_hmat(p), error_history[: iter + 1]
        error_history[iter, 0] = displ_error
        error_history[iter, 1] = error
        error_history[iter, 2] = ort
        if np.allclose(p, 0) or np.any(np.isnan(p)):
            print("[WARNING] Pressure is zero or invalid")
        if error < tolerance:
            break
    return p, Sc_hmat(p), error_history[: iter + 1]


displ = 0.7
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


x_center = np.linspace(-7.0, 7.0, Nframes)
theta_c = np.linspace(0.0, 2 * np.pi, Nframes)
rc = 3.0
for frame, theta_c_ in enumerate(theta_c):
    contact_center = np.array([rc * np.cos(theta_c_), rc * np.sin(theta_c_)])
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
        # p, _, _ = constrained_CG(Sc, error_type, gap, max_iter, tolerance)
        p_hmat, _, _ = constrained_CG_hmat(
            Sc_hmat, error_type, gap, max_iter, tolerance
        )
    else:
        # p, _, _ = constrained_CG(Sc, error_type, gap, max_iter, tolerance)
        p_hmat, _, _ = constrained_CG_hmat(
            Sc_hmat, error_type, gap, max_iter, tolerance
        )

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
        b.setValue(dof * tdim + 2, p_hmat[i])
        b.assemble()

    solver_petsc.solve(b, u)
    from dolfinx.fem import Function

    u_fenics = Function(V)

    u_fenics.x.array[:] = -u.array
    u_fenics.x.scatter_forward()

    u_fenics.name = "u"

    with XDMFFile(MPI.COMM_WORLD, f"tyre_disp_hmat_{frame}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_fenics, t=frame)

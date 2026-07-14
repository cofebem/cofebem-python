import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import (
    CellType,
    create_box,
    locate_entities_boundary,
)
from dolfinx.fem import (
    form,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import assemble_matrix

from ufl import (
    Identity,
    TrialFunction,
    TestFunction,
    dx,
    grad,
    inner,
    sym,
    tr,
)

# ============================================================
# Lanczos approximation of b^T K^{-1} b
# ============================================================


def build_tridiagonal(alphas, betas):
    """
    Construct the Lanczos tridiagonal matrix

             [a1  b1              ]
             [b1  a2  b2          ]
        T =  [    b2  a3  ...     ]
             [          ...  b_m-1]
             [             b_m-1 am]
    """
    n = len(alphas)

    T = np.diag(np.asarray(alphas, dtype=np.float64))

    if n > 1:
        off_diagonal = np.asarray(betas[: n - 1], dtype=np.float64)
        indices = np.arange(n - 1)

        T[indices, indices + 1] = off_diagonal
        T[indices + 1, indices] = off_diagonal

    return T


def lanczos_inverse_quadratic_form(
    K,
    b,
    max_iterations=60,
    breakdown_tolerance=1.0e-14,
):
    """
    Approximate

        q = b^T K^{-1} b

    using the Lanczos formula

        q_m = ||b||^2 e_1^T T_m^{-1} e_1.

    Parameters
    ----------
    K
        PETSc SPD matrix.
    b
        PETSc right-hand-side vector.
    max_iterations
        Maximum number of Lanczos iterations.
    breakdown_tolerance
        Stop when beta_k becomes very small.

    Returns
    -------
    estimates
        estimates[k] is the approximation after k + 1 iterations.
    alphas
        Diagonal entries of T_m.
    betas
        Off-diagonal entries of T_m.
    """

    gamma = b.norm()

    if gamma == 0.0:
        raise ValueError("The starting vector b must be nonzero.")

    # v_1 = b / ||b||
    v = b.copy()
    v.scale(1.0 / gamma)

    # v_0 = 0
    v_previous = b.duplicate()
    v_previous.set(0.0)

    beta_previous = 0.0

    alphas = []
    betas = []
    estimates = []

    for k in range(max_iterations):

        # w = K v_k
        w = K.createVecLeft()
        K.mult(v, w)

        # w = K v_k - beta_{k-1} v_{k-1}
        if k > 0:
            w.axpy(-beta_previous, v_previous)

        # alpha_k = v_k^T w
        alpha = float(np.real(v.dot(w)))

        # w = w - alpha_k v_k
        w.axpy(-alpha, v)

        beta = w.norm()

        alphas.append(alpha)

        # Construct the current T_m and compute
        #
        # q_m = ||b||^2 e_1^T T_m^{-1} e_1
        T = build_tridiagonal(alphas, betas)

        e1 = np.zeros(len(alphas), dtype=np.float64)
        e1[0] = 1.0

        y = np.linalg.solve(T, e1)
        q_m = gamma**2 * y[0]

        estimates.append(float(q_m))

        # Exact or numerical Lanczos breakdown
        if beta < breakdown_tolerance:
            print(f"Lanczos breakdown after {k + 1} iterations.")
            break

        # No need to construct v_{m+1} after the final iteration
        if k == max_iterations - 1:
            break

        # beta_k is an off-diagonal entry of T
        betas.append(float(beta))

        # Prepare the next iteration
        v_previous = v.copy()

        v = w.copy()
        v.scale(1.0 / beta)

        beta_previous = beta

    return estimates, alphas, betas


# ============================================================
# Exact reference value using an LU solve
# ============================================================


def exact_inverse_quadratic_form(K, b):
    """
    Compute exactly, up to the accuracy of the direct solver,

        q = b^T K^{-1} b.
    """

    x = K.createVecRight()
    x.set(0.0)

    solver = PETSc.KSP().create(K.getComm())
    solver.setOperators(K)

    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    solver.setFromOptions()
    solver.setUp()

    solver.solve(b, x)

    reason = solver.getConvergedReason()

    if reason < 0:
        raise RuntimeError(f"The reference LU solve failed with PETSc reason {reason}.")

    value = float(np.real(b.dot(x)))

    return value, x


# ============================================================
# Mesh
# ============================================================

comm = MPI.COMM_WORLD

# This simple example uses local DOF indices directly as PETSc global
# indices, so it is deliberately restricted to serial execution.
if comm.size != 1:
    raise RuntimeError("Run this simple verification script in serial, without mpirun.")

nx = 10
ny = 10
nz = 5

length = 1.0

mesh = create_box(
    comm,
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([length, length, length]),
    ],
    [nx, ny, nz],
    CellType.hexahedron,
)

tdim = mesh.topology.dim
fdim = tdim - 1


# ============================================================
# Material
# ============================================================

E = 1.0e9
nu = 0.3

lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))


# ============================================================
# Elasticity bilinear form
# ============================================================

V = functionspace(mesh, ("Lagrange", 1, (tdim,)))

u = TrialFunction(V)
v = TestFunction(V)


def epsilon(w):
    return sym(grad(w))


def sigma(w):
    return lmbda * tr(epsilon(w)) * Identity(tdim) + 2.0 * mu * epsilon(w)


a = inner(sigma(u), epsilon(v)) * dx


# ============================================================
# Fixed bottom surface
# ============================================================


def bottom_locator(x):
    return np.isclose(x[2], 0.0)


bottom_facets = locate_entities_boundary(
    mesh,
    fdim,
    bottom_locator,
)

bottom_dofs = locate_dofs_topological(
    V,
    fdim,
    bottom_facets,
)

zero_displacement = np.zeros(tdim, dtype=PETSc.ScalarType)

bc = dirichletbc(
    zero_displacement,
    bottom_dofs,
    V,
)


# ============================================================
# Assemble K
# ============================================================

K = assemble_matrix(
    form(a),
    bcs=[bc],
)

K.assemble()

number_of_rows, number_of_columns = K.getSize()

print("Elasticity matrix assembled")
print(f"Matrix size: {number_of_rows} x {number_of_columns}")


# ============================================================
# Select one contact normal DOF
# ============================================================


def top_locator(x):
    return np.isclose(x[2], length)


top_facets = locate_entities_boundary(
    mesh,
    fdim,
    top_locator,
)

# The normal to the upper surface is n = (0, 0, 1).
# Therefore, use the z-displacement subspace V.sub(2).
top_normal_dofs = locate_dofs_topological(
    V.sub(2),
    fdim,
    top_facets,
)

if len(top_normal_dofs) == 0:
    raise RuntimeError("No normal contact DOFs were found.")

# For this simple test, choose one top-surface normal DOF.
contact_dof = int(top_normal_dofs[0])

print(f"Selected contact normal DOF: {contact_dof}")


# ============================================================
# Construct b_1 = B^T e_1
# ============================================================
#
# Here:
#
#     B = e_contact_dof^T
#
# and therefore:
#
#     b_1 = B^T e_1 = e_contact_dof.
#
# Thus:
#
#     (S_c)_11 = b_1^T K^{-1} b_1.
#

b1 = K.createVecRight()
b1.set(0.0)

b1.setValue(
    contact_dof,
    PETSc.ScalarType(1.0),
    addv=PETSc.InsertMode.INSERT_VALUES,
)

b1.assemblyBegin()
b1.assemblyEnd()


# ============================================================
# Lanczos approximation
# ============================================================

max_lanczos_iterations = 60

estimates, alphas, betas = lanczos_inverse_quadratic_form(
    K,
    b1,
    max_iterations=max_lanczos_iterations,
)


# ============================================================
# Reference LU solve
# ============================================================

Sc11_exact, displacement = exact_inverse_quadratic_form(K, b1)

# Since b1 is a canonical vector, this is also:
Sc11_from_solution_entry = float(np.real(displacement.getValue(contact_dof)))


# ============================================================
# Results
# ============================================================

print()
print("=" * 72)
print("Reference result")
print("=" * 72)

print(f"b1^T K^(-1) b1       = {Sc11_exact:.16e}")
print(f"solution[contact_dof] = {Sc11_from_solution_entry:.16e}")

print()
print("=" * 72)
print("Lanczos convergence")
print("=" * 72)

iterations_to_print = sorted(
    {
        1,
        2,
        5,
        10,
        20,
        40,
        len(estimates),
    }
)

print(
    f"{'iterations':>12}"
    f"{'Lanczos estimate':>24}"
    f"{'absolute error':>24}"
    f"{'relative error':>24}"
)

for iteration in iterations_to_print:
    if iteration > len(estimates):
        continue

    estimate = estimates[iteration - 1]

    absolute_error = abs(estimate - Sc11_exact)

    if abs(Sc11_exact) > 0.0:
        relative_error = absolute_error / abs(Sc11_exact)
    else:
        relative_error = np.nan

    print(
        f"{iteration:12d}"
        f"{estimate:24.16e}"
        f"{absolute_error:24.16e}"
        f"{relative_error:24.16e}"
    )


Sc11_lanczos = estimates[-1]

absolute_error = abs(Sc11_lanczos - Sc11_exact)
relative_error = absolute_error / abs(Sc11_exact)

print()
print("=" * 72)
print("Final comparison")
print("=" * 72)

print(f"Exact (LU)        : {Sc11_exact:.16e}")
print(f"Lanczos           : {Sc11_lanczos:.16e}")
print(f"Absolute error    : {absolute_error:.16e}")
print(f"Relative error    : {relative_error:.16e}")
print(f"Lanczos iterations: {len(estimates)}")

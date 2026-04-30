import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import locate_entities_boundary, create_box, CellType
from dolfinx.fem import Constant, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix_mat,
    assemble_vector,
    apply_lifting,
)
from ufl import Identity, TrialFunction, TestFunction, rhs, sym, grad, inner, tr, dx
import time

# ---------------- Mesh ----------------
Lbox = 1.0
ncells = 5
mesh = create_box(
    MPI.COMM_WORLD,
    [[0.0, 0.0, 0.0], [Lbox, Lbox, Lbox]],
    [ncells * 10, ncells * 10, ncells],
    CellType.hexahedron,
)
tdim = mesh.topology.dim
fdim = tdim - 1

# ---------------- Material ----------------
E = 1.0e9
nu = 0.3
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# ---------------- Variational forms ----------------
V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
u = TrialFunction(V)
v = TestFunction(V)


def epsilon(w):
    return sym(grad(w))


def sigma(w):
    return lmbda * tr(epsilon(w)) * Identity(tdim) + 2 * mu * epsilon(w)


f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
a = inner(sigma(u), epsilon(v)) * dx
Lform = inner(f_v, v) * dx


# ---------------- Dirichlet BC ----------------
def Gamma_u_locator(x):
    return np.isclose(x[2], 0.0)


Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)
u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
bc = dirichletbc(u0, Gamma_u_dofs, V)
bcs = [bc]


# ------------------ Contact BC ----------------
def Gamma_c_locator(x):
    return np.isclose(x[2], Lbox)


Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
Gamma_c_dofs = locate_dofs_topological(V, fdim, Gamma_c)

# ---------------- Problem setup ----------------
problem = LinearProblem(
    a=a,
    L=Lform,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

# ---------------- Assemble system ----------------
problem._A.zeroEntries()
assemble_matrix_mat(problem._A, problem._a, bcs=problem.bcs)
problem._A.assemble()

with problem._b.localForm() as b_loc:
    b_loc.set(0)
assemble_vector(problem._b, problem._L)

apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
for bc in problem.bcs:
    bc.set(problem._b.array_w)

A = problem._A
b = problem._b


# ------------------Setup Solver -----------------
def setup_solver(
    mesh,
    A,
    ksp_type="preonly",
    pc_type="lu",
    rtol=None,
    atol=None,
    max_it=None,
    options_prefix=None,
):
    solver = PETSc.KSP().create(mesh.comm)
    if options_prefix is not None:
        solver.setOptionsPrefix(options_prefix)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    if rtol is not None or atol is not None or max_it is not None:
        solver.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    solver.setFromOptions()
    solver.setUp()
    return solver


def setup_direct_solver(mesh, A):
    return setup_solver(
        mesh,
        A,
        ksp_type="preonly",
        pc_type="lu",
        options_prefix="direct_",
    )


def setup_iterative_solver(mesh, A):
    return setup_solver(
        mesh,
        A,
        ksp_type="cg",
        pc_type="gamg",
        rtol=1e-8,
        atol=1e-12,
        max_it=2000,
        options_prefix="iter_",
    )


def print_solver_info(name, solver):
    pc = solver.getPC()
    print(f"{name} KSP type:", solver.getType())
    print(f"{name} PC type:", pc.getType())
    if pc.getType() == "lu":
        print(f"{name} factor solver type:", pc.getFactorSolverType())


def it_solve_time(mesh, solver, b, Gamma_c_dofs, n, label="Repeated"):
    duration = 0
    N = b.getSize()
    nc = len(Gamma_c_dofs)

    rhs = PETSc.Vec().createMPI(N, comm=mesh.comm)
    sol = PETSc.Vec().createMPI(N, comm=mesh.comm)
    Sc = np.zeros((nc, n), dtype=PETSc.ScalarType)

    for i in range(n):
        rhs.set(0)
        rhs.setValue(3 * Gamma_c_dofs[i] + 2, 1.0e9)
        rhs.assemble()
        start_time = time.time()
        solver.solve(rhs, sol)
        end_time = time.time()
        duration += end_time - start_time
        Sc[:, i] = sol.array[3 * Gamma_c_dofs + 2] / 1.0e9

    print(f"{label} {n} solves took {duration:.4f} seconds")
    return duration, Sc


def batch_solve_time(mesh, solver, b, Gamma_c_dofs, n, label="Batch"):
    N = b.getSize()
    nc = len(Gamma_c_dofs)
    B = PETSc.Mat().createDense([N, n], comm=mesh.comm)
    B.setUp()

    duration = 0

    vals = np.zeros((N, n), dtype=PETSc.ScalarType)
    for i in range(n):
        vals[3 * Gamma_c_dofs[i] + 2, i] = 1.0e9

    rows = np.arange(N, dtype=np.int32)
    cols = np.arange(n, dtype=np.int32)
    B.setValues(rows, cols, vals)
    B.assemble()

    X = PETSc.Mat().createDense([N, n], comm=mesh.comm)
    X.setUp()
    X.assemble()

    start_time = time.time()
    solver.matSolve(B, X)
    end_time = time.time()
    duration = end_time - start_time

    Sc = X.getDenseArray()[3 * Gamma_c_dofs + 2, :] / 1.0e9
    print(f"{label} {n} solves took {duration:.4f} seconds")
    return duration, Sc


def assert_close(label, reference, candidate, rtol=1e-6, atol=1e-12):
    if np.allclose(reference, candidate, rtol=rtol, atol=atol):
        print(f"✔ {label} matches the direct repeated solve")
        return

    err = np.linalg.norm(reference - candidate)
    ref = np.linalg.norm(reference)
    rel = err / ref if ref > 0 else err
    raise AssertionError(f"✗ {label} mismatch: abs={err:.3e}, rel={rel:.3e}")


def benchmark_solve_times(mesh, A, b, Gamma_c_dofs, n):
    direct_solver = setup_direct_solver(mesh, A)
    iter_solver = setup_iterative_solver(mesh, A)

    print_solver_info("Direct", direct_solver)
    print_solver_info("Iterative", iter_solver)

    it_duration, it_Sc = it_solve_time(
        mesh,
        direct_solver,
        b,
        Gamma_c_dofs,
        n,
        label="Repeated direct",
    )
    batch_duration, batch_Sc = batch_solve_time(
        mesh,
        direct_solver,
        b,
        Gamma_c_dofs,
        n,
        label="Batch direct",
    )
    batch_iter_duration, batch_iter_Sc = batch_solve_time(
        mesh,
        iter_solver,
        b,
        Gamma_c_dofs,
        n,
        label="Batch iterative",
    )

    assert_close("Batch direct", it_Sc, batch_Sc)
    assert_close("Batch iterative", it_Sc, batch_iter_Sc)

    return it_duration, batch_duration, batch_iter_duration


if __name__ == "__main__":
    solver = setup_direct_solver(mesh, A)
    rhs0 = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)
    sol0 = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)
    solver.solve(rhs0, sol0)
    print("done")
    # print(sol0)
    it_times = []
    batch_times = []
    batch_iter_times = []
    ns = np.linspace(1, 20, 5, dtype=int)
    for n in ns:
        it_time, batch_time, batch_iter_time = benchmark_solve_times(
            mesh, A, b, Gamma_c_dofs, n
        )
        it_times.append(it_time)
        batch_times.append(batch_time)
        batch_iter_times.append(batch_iter_time)
        print("===========================================================")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))

    plt.plot(
        ns,
        it_times,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Repeated direct solves",
    )

    plt.plot(
        ns,
        batch_times,
        marker="s",
        linewidth=2.2,
        markersize=6,
        label="Batch direct solves",
    )

    plt.plot(
        ns,
        batch_iter_times,
        marker="^",
        linewidth=2.2,
        markersize=6,
        label="Batch iterative solves",
    )

    plt.xlabel("Number of solves", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(True, which="both", linestyle="--", alpha=0.8)
    plt.legend(fontsize=11, frameon=True)

    plt.title("Solve time scaling", fontsize=13, pad=10)

    plt.tight_layout()
    plt.show()

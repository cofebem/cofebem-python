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
def setup_solver(mesh, A):
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    solver.setFromOptions()
    solver.setUp()
    return solver


def it_solve_time(mesh, solver, b, Gamma_c_dofs, n):
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

    print(f"Iterative {n} solves took {duration:.4f} seconds")
    return duration, Sc


def batch_solve_time(mesh, solver, b, Gamma_c_dofs, n):
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
    print(f"Batch {n} solves took {duration:.4f} seconds")
    return duration, Sc


def benchmark_solve_times(mesh, A, b, Gamma_c_dofs, n):
    solver = setup_solver(mesh, A)
    print("KSP type:", solver.getType())

    pc = solver.getPC()
    print("PC type:", pc.getType())

    print("Factor solver type:", pc.getFactorSolverType())

    it_duration, it_Sc = it_solve_time(mesh, solver, b, Gamma_c_dofs, n)
    batch_duration, batch_Sc = batch_solve_time(mesh, solver, b, Gamma_c_dofs, n)

    for i in range(n):
        it_col = it_Sc[:, i]
        batch_col = batch_Sc[:, i]
        assert np.allclose(it_col, batch_col), f"✗ Mismatch in solution for column {i}"

    print("✔ All solutions match between iterative and batch solves")
    return it_duration, batch_duration


if __name__ == "__main__":
    solver = setup_solver(mesh, A)
    rhs0 = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)
    sol0 = PETSc.Vec().createMPI(b.getSize(), comm=mesh.comm)
    solver.solve(rhs0, sol0)
    print("done")
    print(sol0)
    # it_times = []
    # batch_times = []
    # ns = np.linspace(1, len(Gamma_c_dofs), 10, dtype=int)
    # for n in ns:
    #     it_time, batch_time = benchmark_solve_times(mesh, A, b, Gamma_c_dofs, n)
    #     it_times.append(it_time)
    #     batch_times.append(batch_time)
    #     print("===========================================================")

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(7, 4.5))

    # plt.plot(
    #     ns, it_times, marker="o", linewidth=2.2, markersize=6, label="Iterative solves"
    # )

    # plt.plot(
    #     ns, batch_times, marker="s", linewidth=2.2, markersize=6, label="Batch solves"
    # )

    # plt.xlabel("Number of solves", fontsize=12)
    # plt.ylabel("Time (seconds)", fontsize=12)

    # plt.xticks(fontsize=11)
    # plt.yticks(fontsize=11)

    # plt.grid(True, which="both", linestyle="--", alpha=0.8)
    # plt.legend(fontsize=11, frameon=True)

    # plt.title("Solve time scaling", fontsize=13, pad=10)

    # plt.tight_layout()
    # plt.show()

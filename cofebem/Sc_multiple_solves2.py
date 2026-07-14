import time
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import locate_entities_boundary, create_box, CellType
from dolfinx.fem import Constant, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix,
    assemble_vector,
    apply_lifting,
)
from ufl import Identity, TrialFunction, TestFunction, sym, grad, inner, tr, dx


# ============================================================
# Important: this script is intentionally written for sequential
# benchmarking. Your dense Mat creation and sol.array indexing are
# not valid as-is for several MPI ranks.
# ============================================================
COMM = MPI.COMM_WORLD
if COMM.size != 1:
    raise RuntimeError(
        "This benchmark is written for sequential runs only. "
        "Run it with one MPI rank, e.g. python script.py or mpirun -n 1 python script.py."
    )


# ---------------- Mesh ----------------
Lbox = 1.0
ncells = 5
mesh = create_box(
    COMM,
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

# In a vector-valued P1 space, the PETSc vector has block entries.
# For displacement component uz, use block_size * dof + 2.
BLOCK_SIZE = V.dofmap.index_map_bs
NORMAL_COMPONENT = 2
FORCE_SCALE = 1.0e9


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


# ---------------- Contact boundary ----------------
def Gamma_c_locator(x):
    return np.isclose(x[2], Lbox)


Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
Gamma_c_dofs = locate_dofs_topological(V, fdim, Gamma_c)


# ---------------- Problem setup ----------------
problem = LinearProblem(
    a=a,
    L=Lform,
    bcs=bcs,
    petsc_options_prefix="lp",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

# ---------------- Assemble system ----------------
problem._A.zeroEntries()
assemble_matrix(problem._A, problem._a, bcs=problem.bcs)
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


# ============================================================
# Timing helpers
# ============================================================
def now(comm=COMM):
    """Synchronized wall-clock timer."""
    comm.Barrier()
    return MPI.Wtime()


def timed_call(func, comm=COMM):
    """Return elapsed time and function result."""
    t0 = now(comm)
    out = func()
    t1 = now(comm)
    return t1 - t0, out


@dataclass
class TimingResult:
    name: str
    setup_time: float
    rhs_time: float
    solve_time: float
    total_time: float
    Sc: np.ndarray
    avg_iterations: float | None = None


# ============================================================
# Solver setup
# ============================================================
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
    solver.setInitialGuessNonzero(False)

    if rtol is not None or atol is not None or max_it is not None:
        solver.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    # Allows command-line override, e.g.
    #   -direct_pc_factor_mat_solver_type mumps
    #   -iter_ksp_type cg -iter_pc_type gamg
    solver.setFromOptions()

    # For LU this should include factorization setup; for GAMG this should
    # include hierarchy/preconditioner setup. We time this separately.
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
    PETSc.Sys.Print(f"{name} KSP type: {solver.getType()}")
    PETSc.Sys.Print(f"{name} PC type:  {pc.getType()}")
    if pc.getType() == "lu":
        PETSc.Sys.Print(f"{name} factor solver type: {pc.getFactorSolverType()}")


# ============================================================
# RHS creation and extraction helpers
# ============================================================
def contact_component_dofs(Gamma_c_dofs, n=None):
    """Return PETSc-vector indices of the normal component on Gamma_c."""
    dofs = np.asarray(Gamma_c_dofs, dtype=np.int32)
    if n is not None:
        if n > len(dofs):
            raise ValueError(f"n={n} is larger than len(Gamma_c_dofs)={len(dofs)}")
        dofs = dofs[:n]
    return BLOCK_SIZE * dofs + NORMAL_COMPONENT


def check_dofs_are_valid(N, *dof_arrays):
    for arr in dof_arrays:
        if len(arr) == 0:
            raise ValueError("Empty dof array.")
        if np.min(arr) < 0 or np.max(arr) >= N:
            raise ValueError(
                "Some computed PETSc-vector dof indices are outside [0, N). "
                "Check whether Gamma_c_dofs are block dofs or scalar dofs."
            )


def build_rhs_vectors(mesh, N, Gamma_c_dofs, n):
    """Create n PETSc Vec RHS objects, one point load per RHS."""
    force_dofs = contact_component_dofs(Gamma_c_dofs, n=n)
    check_dofs_are_valid(N, force_dofs)

    rhs_list = []
    for dof in force_dofs:
        rhs = PETSc.Vec().createMPI(N, comm=mesh.comm)
        rhs.set(0.0)
        rhs.setValue(int(dof), FORCE_SCALE)
        rhs.assemble()
        rhs_list.append(rhs)
    return rhs_list


def build_rhs_dense_matrix(mesh, N, Gamma_c_dofs, n):
    """Create dense RHS matrix B = [b1, ..., bn]. Sequential only."""
    force_dofs = contact_component_dofs(Gamma_c_dofs, n=n)
    check_dofs_are_valid(N, force_dofs)

    vals = np.zeros((N, n), dtype=PETSc.ScalarType)
    for j, dof in enumerate(force_dofs):
        vals[int(dof), j] = FORCE_SCALE

    B = PETSc.Mat().createDense([N, n], comm=mesh.comm)
    B.setUp()
    B.setValues(np.arange(N, dtype=np.int32), np.arange(n, dtype=np.int32), vals)
    B.assemble()
    return B


def extract_contact_Sc_from_vec_solutions(solutions, Gamma_c_dofs):
    """Extract Sc[:, j] from a list of solution vectors."""
    contact_dofs = contact_component_dofs(Gamma_c_dofs, n=None)
    N = solutions[0].getSize()
    check_dofs_are_valid(N, contact_dofs)

    nc = len(contact_dofs)
    n = len(solutions)
    Sc = np.empty((nc, n), dtype=PETSc.ScalarType)
    for j, sol in enumerate(solutions):
        Sc[:, j] = sol.array[contact_dofs] / FORCE_SCALE
    return Sc


def extract_contact_Sc_from_dense_solution(X, Gamma_c_dofs):
    """Extract Sc from dense matrix X whose columns are displacement solutions."""
    contact_dofs = contact_component_dofs(Gamma_c_dofs, n=None)
    N = X.getSize()[0]
    check_dofs_are_valid(N, contact_dofs)
    return X.getDenseArray()[contact_dofs, :] / FORCE_SCALE


# ============================================================
# Solve kernels
# ============================================================
def repeated_vec_solve(mesh, solver, rhs_list):
    """Solve each RHS as a separate PETSc Vec solve."""
    N = rhs_list[0].getSize()
    sol = PETSc.Vec().createMPI(N, comm=mesh.comm)
    solutions = []
    iteration_counts = []

    t0 = now(mesh.comm)
    for rhs in rhs_list:
        sol.set(0.0)
        solver.solve(rhs, sol)

        reason = solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"KSP failed with converged reason {reason}")

        iteration_counts.append(solver.getIterationNumber())
        solutions.append(sol.copy())
    t1 = now(mesh.comm)

    avg_it = float(np.mean(iteration_counts)) if iteration_counts else None
    return t1 - t0, solutions, avg_it


def dense_mat_solve(mesh, solver, B):
    """Solve A X = B using PETSc KSP.matSolve."""
    N, n = B.getSize()
    X = PETSc.Mat().createDense([N, n], comm=mesh.comm)
    X.setUp()
    X.assemble()

    t0 = now(mesh.comm)
    solver.matSolve(B, X)
    t1 = now(mesh.comm)
    return t1 - t0, X


# ============================================================
# Benchmark for one n
# ============================================================
def compare_to_reference(label, reference, candidate, rtol=1e-5, atol=1e-12):
    err = np.linalg.norm(reference - candidate)
    ref = np.linalg.norm(reference)
    rel = err / ref if ref > 0 else err
    ok = np.allclose(reference, candidate, rtol=rtol, atol=atol)
    status = "OK" if ok else "WARNING"
    PETSc.Sys.Print(f"  {status:7s} {label:<24s} abs={err:.3e}, rel={rel:.3e}")
    return ok, err, rel


def benchmark_one_n(mesh, A, b, Gamma_c_dofs, n, include_iterative_matsolve=True):
    PETSc.Sys.Print(f"\n================ n = {n} ================")
    N = b.getSize()

    # Setup each solver once. The same setup time is assigned to all methods
    # using that solver type. This avoids timing different factorizations for
    # repeated direct and batch direct.
    direct_setup_time, direct_solver = timed_call(lambda: setup_direct_solver(mesh, A), mesh.comm)
    iter_setup_time, iter_solver = timed_call(lambda: setup_iterative_solver(mesh, A), mesh.comm)

    print_solver_info("Direct", direct_solver)
    print_solver_info("Iterative", iter_solver)
    PETSc.Sys.Print(f"Direct setup time:    {direct_setup_time:.6f} s")
    PETSc.Sys.Print(f"Iterative setup time: {iter_setup_time:.6f} s")

    results: list[TimingResult] = []

    # 1) Repeated direct: Vec RHS + KSP.solve in a loop
    rhs_time, rhs_list = timed_call(lambda: build_rhs_vectors(mesh, N, Gamma_c_dofs, n), mesh.comm)
    solve_time, sols, avg_it = repeated_vec_solve(mesh, direct_solver, rhs_list)
    Sc_repeated_direct = extract_contact_Sc_from_vec_solutions(sols, Gamma_c_dofs)
    results.append(
        TimingResult(
            name="Repeated direct",
            setup_time=direct_setup_time,
            rhs_time=rhs_time,
            solve_time=solve_time,
            total_time=direct_setup_time + rhs_time + solve_time,
            Sc=Sc_repeated_direct,
            avg_iterations=avg_it,
        )
    )

    # 2) Batch direct: dense RHS matrix + KSP.matSolve
    rhs_time, B = timed_call(lambda: build_rhs_dense_matrix(mesh, N, Gamma_c_dofs, n), mesh.comm)
    solve_time, X = dense_mat_solve(mesh, direct_solver, B)
    Sc_batch_direct = extract_contact_Sc_from_dense_solution(X, Gamma_c_dofs)
    results.append(
        TimingResult(
            name="Batch direct",
            setup_time=direct_setup_time,
            rhs_time=rhs_time,
            solve_time=solve_time,
            total_time=direct_setup_time + rhs_time + solve_time,
            Sc=Sc_batch_direct,
            avg_iterations=None,
        )
    )

    # 3) Repeated iterative: Vec RHS + CG/GAMG solve in a loop
    # This is the fair iterative counterpart to repeated direct.
    rhs_time, rhs_list = timed_call(lambda: build_rhs_vectors(mesh, N, Gamma_c_dofs, n), mesh.comm)
    solve_time, sols, avg_it = repeated_vec_solve(mesh, iter_solver, rhs_list)
    Sc_repeated_iter = extract_contact_Sc_from_vec_solutions(sols, Gamma_c_dofs)
    results.append(
        TimingResult(
            name="Repeated iterative",
            setup_time=iter_setup_time,
            rhs_time=rhs_time,
            solve_time=solve_time,
            total_time=iter_setup_time + rhs_time + solve_time,
            Sc=Sc_repeated_iter,
            avg_iterations=avg_it,
        )
    )

    # 4) Optional: KSP.matSolve with the iterative solver.
    # This is usually not a true block Krylov method; PETSc may internally
    # solve RHS columns one by one depending on the KSP/PC implementation.
    if include_iterative_matsolve:
        rhs_time, B_iter = timed_call(lambda: build_rhs_dense_matrix(mesh, N, Gamma_c_dofs, n), mesh.comm)
        try:
            solve_time, X_iter = dense_mat_solve(mesh, iter_solver, B_iter)
            Sc_iter_mat = extract_contact_Sc_from_dense_solution(X_iter, Gamma_c_dofs)
            results.append(
                TimingResult(
                    name="Iterative matSolve",
                    setup_time=iter_setup_time,
                    rhs_time=rhs_time,
                    solve_time=solve_time,
                    total_time=iter_setup_time + rhs_time + solve_time,
                    Sc=Sc_iter_mat,
                    avg_iterations=None,
                )
            )
        except PETSc.Error as err:
            PETSc.Sys.Print(f"Iterative matSolve failed/skipped: {err}")

    PETSc.Sys.Print("\nAccuracy vs repeated direct reference:")
    for result in results[1:]:
        compare_to_reference(result.name, Sc_repeated_direct, result.Sc)

    PETSc.Sys.Print("\nTiming table:")
    PETSc.Sys.Print(
        f"{'Method':<22s} {'setup':>10s} {'rhs':>10s} "
        f"{'solve':>10s} {'total':>10s} {'avg it':>10s}"
    )
    for result in results:
        avg_it_str = "-" if result.avg_iterations is None else f"{result.avg_iterations:.1f}"
        PETSc.Sys.Print(
            f"{result.name:<22s} "
            f"{result.setup_time:10.4f} {result.rhs_time:10.4f} "
            f"{result.solve_time:10.4f} {result.total_time:10.4f} "
            f"{avg_it_str:>10s}"
        )

    return results


# ============================================================
# Main benchmark loop
# ============================================================
def append_result(history, n, results):
    for result in results:
        history.setdefault(result.name, {"n": [], "solve": [], "total": []})
        history[result.name]["n"].append(n)
        history[result.name]["solve"].append(result.solve_time)
        history[result.name]["total"].append(result.total_time)


if __name__ == "__main__":
    PETSc.Sys.Print(f"Global system size N = {b.getSize()}")
    PETSc.Sys.Print(f"Number of Gamma_c dofs = {len(Gamma_c_dofs)}")
    PETSc.Sys.Print(f"Vector block size = {BLOCK_SIZE}")

    # Use unique n values; np.linspace(..., dtype=int) can duplicate values
    # for some ranges.
    ns = np.unique(np.linspace(1, 20, 5, dtype=int))

    history = {}
    for n in ns:
        results = benchmark_one_n(
            mesh,
            A,
            b,
            Gamma_c_dofs,
            int(n),
            include_iterative_matsolve=True,
        )
        append_result(history, int(n), results)

    # ---------------- Plots ----------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for name, data in history.items():
        plt.plot(data["n"], data["solve"], marker="o", linewidth=2.0, markersize=5, label=name)
    plt.xlabel("Number of RHS / columns", fontsize=12)
    plt.ylabel("Solve time only (s)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.8)
    plt.legend(fontsize=9, frameon=True)
    plt.title("Solve phase only", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4.5))
    for name, data in history.items():
        plt.plot(data["n"], data["total"], marker="o", linewidth=2.0, markersize=5, label=name)
    plt.xlabel("Number of RHS / columns", fontsize=12)
    plt.ylabel("Setup + RHS construction + solve time (s)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.8)
    plt.legend(fontsize=9, frameon=True)
    plt.title("End-to-end time", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show()

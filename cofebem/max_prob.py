import argparse
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)

from dolfinx import la
import dolfinx.cpp.la

from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix,
    assemble_vector,
    apply_lifting,
)

from dolfinx.mesh import (
    CellType,
    create_box,
    locate_entities_boundary,
    meshtags,
    create_submesh,
)
from dolfinx.io import XDMFFile
from ufl import (
    Identity,
    FacetNormal,
    Measure,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
)

from cofebem.mesh.tyre_hex import tyre_hex_mesh

# ---------------- material / geometry parameters ----------------
# E = 2.5e8
# NU = 0.48
E = 1.0e9
NU = 0.3

A0 = 0.20
B0 = 0.10
THICKNESS = 0.03
OX = 0.0
OZ = 0.5
THETA_CUT = np.pi / 6

XTOL = 1e-3
FTOL = 5e-3


def normalize(v):
    return v / np.linalg.norm(v)


def rotation_matrix_from_a_to_b(a, b):
    a = normalize(a)
    b = normalize(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-14:
        if c > 0:
            return np.eye(3)
        # 180° rotation
        e = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, e)) > 0.9:
            e = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, e))
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + 2 * (K @ K)

    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    return R


def build_problem(
    Nr: int, Nt: int, Np: int
) -> tuple[LinearProblem, int, int, int, np.ndarray]:
    """Build the mesh, assemble the linear-elasticity system and return the
    ready-to-solve PETSc operator/vector along with the mesh size."""

    lmbda = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))

    Ns = tyre_hex_mesh(
        A0,
        B0,
        THICKNESS,
        OX,
        OZ,
        nr=Nr,
        ntt=Nt,
        npp=Np,
        filename=f"tyre_hex_{Nr}_{Nt}_{Np}.xdmf",
    )
    with XDMFFile(MPI.COMM_WORLD, f"tyre_hex_{Nr}_{Nt}_{Np}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    tdim = mesh.topology.dim
    fdim = tdim - 1

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
    L = inner(f_v, v) * dx

    # ---------------- Dirichlet BC ----------------
    def Gamma_u_locator(x):
        X = x[0]
        Y = x[1]
        Z = x[2]

        r = np.sqrt(Y * Y + Z * Z)

        a_ref = A0 + 0.5 * THICKNESS
        b_ref = B0 + 0.5 * THICKNESS

        theta = np.arctan2((r - OZ) / b_ref, (X - OX) / a_ref)

        theta1 = -np.pi + THETA_CUT
        theta2 = -THETA_CUT

        tol = 0.15  # radians (~8.5 degrees)

        def ang_dist(a, b):
            return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

        return (ang_dist(theta, theta1) < tol) | (ang_dist(theta, theta2) < tol)

    Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
    Gamma_u_set = set(Gamma_u.tolist())
    Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)

    u0 = np.array([0, 0, 0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gamma_u_dofs, V)
    bcs = [bc]

    # ---------------- Neumann BC----------------
    def Gamma_t_locator(x):
        X = x[0]
        Y = x[1]
        Z = x[2]
        r = np.sqrt(Y * Y + Z * Z)

        F = ((X - OX) / A0) ** 2 + ((r - OZ) / B0) ** 2 - 1.0
        on_inner = np.abs(F) < FTOL

        return on_inner

    Gamma_t = locate_entities_boundary(mesh, fdim, Gamma_t_locator)
    Gamma_t = np.array([f for f in Gamma_t if f not in Gamma_u_set], dtype=np.int32)
    Gamma_t_id = 1
    Gamma_t_tags = np.full(Gamma_t.shape, Gamma_t_id, dtype=np.int32)

    n = FacetNormal(mesh)

    p0 = Constant(mesh, PETSc.ScalarType(1.5e5))

    t0 = -p0 * n

    # ---------------- Contact BC ----------------
    def Gamma_c_locator(x):
        X = x[0]
        Y = x[1]
        Z = x[2]
        r = np.sqrt(Y * Y + Z * Z)

        aout = A0 + THICKNESS
        bout = B0 + THICKNESS
        F = ((X - OX) / aout) ** 2 + ((r - OZ) / bout) ** 2 - 1.0
        on_outer = np.abs(F) < FTOL

        return on_outer

    Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
    Gamma_c = np.array([f for f in Gamma_c if f not in Gamma_u_set], dtype=np.int32)
    Gamma_c_id = 2
    Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)

    # Contact-zone scalar DOFs (Ic): used to probe the system with
    # unit-force right-hand sides when building/benchmarking Sc.
    Gamma_c_dof_blocks = locate_dofs_topological(V, fdim, Gamma_c)
    bs = V.dofmap.index_map_bs
    Ic = (
        (bs * Gamma_c_dof_blocks[:, None] + np.arange(bs)[None, :])
        .flatten()
        .astype(np.int32)
    )

    tc = Function(V)
    tc.name = "$p_{c}$"
    # ---------------------- Setup Neumann and contact contributions to L ----------------
    facet_indices = np.hstack([Gamma_t, Gamma_c]).astype(np.int32)
    facet_values = np.hstack(
        [
            Gamma_t_tags,
            Gamma_c_tags,
        ]
    ).astype(np.int32)

    order = np.argsort(facet_indices)
    facet_indices = facet_indices[order]
    facet_values = facet_values[order]
    uniq, first = np.unique(facet_indices, return_index=True)
    facet_indices = facet_indices[first]
    facet_values = facet_values[first]

    mt = meshtags(mesh, fdim, facet_indices, facet_values)

    ds = Measure("ds", domain=mesh, subdomain_data=mt)

    L += inner(t0, v) * ds(Gamma_t_id) + inner(tc, v) * ds(Gamma_c_id)

    num_nodes = mesh.geometry.x.shape[0]
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="lu_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    problem._A.zeroEntries()
    assemble_matrix(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()

    with problem._b.localForm() as b0:
        b0.set(0.0)

    assemble_vector(problem._b, problem._L)
    apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
    problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    for bc in problem.bcs:
        bc.set(problem._b.array_w)

    return problem, num_nodes, num_dofs, Ns, Ic


def build_nullspace(V: FunctionSpace) -> PETSc.NullSpace:
    """Build the PETSc near-nullspace of rigid body modes for 3D elasticity."""

    dtype = PETSc.ScalarType

    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for _ in range(6)]
    b = [x.array for x in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    _basis = [x._cpp_object for x in basis]
    dolfinx.cpp.la.orthonormalize(_basis)
    assert dolfinx.cpp.la.is_orthonormal(_basis, 1.0e-8)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=bs, comm=V.mesh.comm)
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


def run_lu(problem: LinearProblem, n_solves: int) -> dict:
    x = problem.A.createVecRight()

    lu_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    lu_solver.setOperators(problem.A)
    lu_solver.setType("preonly")
    lu_solver.getPC().setType("lu")
    lu_solver.setFromOptions()

    # Setup / factorization time
    t0 = time.time()
    lu_solver.setUp()
    factor_time = time.time() - t0

    # Solve time
    lu_times = []
    lu_reasons = []

    for _ in range(n_solves):
        x.set(0.0)

        t0 = time.time()
        lu_solver.solve(problem.b, x)
        solve_time = time.time() - t0

        lu_times.append(solve_time)
        lu_reasons.append(lu_solver.getConvergedReason())

    lu_solver.destroy()

    return {
        "factor_time": factor_time,
        "solve_time": float(np.mean(lu_times)),
        "reason": lu_reasons[-1],
        "converged": lu_reasons[-1] > 0,
    }


def run_cg(problem: LinearProblem, n_solves: int, pc_type: str) -> dict:
    """Solve with CG preconditioned by `pc_type` (e.g. "hypre" or "gamg")."""

    x = problem.A.createVecRight()

    cg_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    cg_solver.setOperators(problem.A)

    cg_solver.setType("cg")

    pc = cg_solver.getPC()
    pc.setType(pc_type)

    cg_solver.setTolerances(
        rtol=1e-6,
        atol=1e-8,
        max_it=1000,
    )

    cg_solver.setFromOptions()

    # Setup / preconditioner setup time
    t0 = time.time()
    cg_solver.setUp()
    pc_setup_time = time.time() - t0

    # Solve time
    cg_times = []
    cg_iterations = []
    cg_reasons = []
    cg_residuals = []

    for _ in range(n_solves):
        x.set(0.0)

        t0 = time.time()
        cg_solver.solve(problem.b, x)
        solve_time = time.time() - t0

        cg_times.append(solve_time)
        cg_iterations.append(cg_solver.getIterationNumber())
        cg_reasons.append(cg_solver.getConvergedReason())
        cg_residuals.append(cg_solver.getResidualNorm())

    cg_solver.destroy()

    solve_time = float(np.mean(cg_times))
    iterations = float(np.mean(cg_iterations))

    return {
        "pc_setup_time": pc_setup_time,
        "solve_time": solve_time,
        "solve_time_per_it": solve_time / iterations if iterations > 0 else solve_time,
        "iterations": iterations,
        "reason": cg_reasons[-1],
        "residual": cg_residuals[-1],
        "converged": cg_reasons[-1] > 0,
    }


def run_cg_gamg_nullspace(problem: LinearProblem, n_solves: int) -> dict:
    """Solve with CG + GAMG using a near-nullspace of the 3D elasticity rigid
    body modes, an SPD matrix hint, and Chebyshev/Jacobi multigrid
    smoothing (as opposed to plain `run_cg(problem, n_solves, "gamg")`)."""

    V = problem.u.function_space
    bs = V.dofmap.index_map_bs
    problem.A.setBlockSize(bs)

    nullspace = build_nullspace(V)
    problem.A.setNearNullSpace(nullspace)
    problem.A.setOption(PETSc.Mat.Option.SPD, True)

    x = problem.A.createVecRight()

    cg_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    cg_solver.setOperators(problem.A)
    cg_solver.setOptionsPrefix("gamg_ns_")

    opts = PETSc.Options()
    opts["gamg_ns_ksp_type"] = "cg"
    opts["gamg_ns_ksp_rtol"] = 1.0e-6
    opts["gamg_ns_ksp_atol"] = 1.0e-8
    opts["gamg_ns_ksp_max_it"] = 1000
    opts["gamg_ns_pc_type"] = "gamg"
    # Chebyshev smoothing for multigrid, with an improved eigenvalue estimate
    # opts["gamg_ns_mg_levels_ksp_type"] = "chebyshev"
    # opts["gamg_ns_mg_levels_pc_type"] = "jacobi"
    # opts["gamg_ns_mg_levels_ksp_chebyshev_esteig_steps"] = 10

    cg_solver.setFromOptions()

    # Setup / preconditioner setup time
    t0 = time.time()
    cg_solver.setUp()
    pc_setup_time = time.time() - t0

    # Solve time
    cg_times = []
    cg_iterations = []
    cg_reasons = []
    cg_residuals = []

    for _ in range(n_solves):
        x.set(0.0)

        t0 = time.time()
        cg_solver.solve(problem.b, x)
        solve_time = time.time() - t0

        cg_times.append(solve_time)
        cg_iterations.append(cg_solver.getIterationNumber())
        cg_reasons.append(cg_solver.getConvergedReason())
        cg_residuals.append(cg_solver.getResidualNorm())

    cg_solver.destroy()

    solve_time = float(np.mean(cg_times))
    iterations = float(np.mean(cg_iterations))

    return {
        "pc_setup_time": pc_setup_time,
        "solve_time": solve_time,
        "solve_time_per_it": solve_time / iterations if iterations > 0 else solve_time,
        "iterations": iterations,
        "reason": cg_reasons[-1],
        "residual": cg_residuals[-1],
        "converged": cg_reasons[-1] > 0,
    }


def select_probe_dofs(Ic: np.ndarray, n_rhs: int) -> np.ndarray:
    """Pick `n_rhs` DOFs spread evenly across the contact-zone DOF set `Ic`."""

    if len(Ic) <= n_rhs:
        return Ic

    idx = np.linspace(0, len(Ic) - 1, n_rhs).round().astype(int)
    return Ic[idx]


def run_cg_probe(
    problem: LinearProblem, pc_type: str, dofs: np.ndarray, force: float = 1e5
) -> dict:
    """Solve CG+`pc_type` once per DOF in `dofs`, each time with a fresh
    right-hand side that is zero everywhere except a unit force `force` at
    that DOF. The factorisation/PC setup is done once and reused across all
    right-hand sides."""

    cg_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    cg_solver.setOperators(problem.A)
    cg_solver.setType("cg")
    cg_solver.getPC().setType(pc_type)
    cg_solver.setTolerances(rtol=1e-6, atol=1e-8, max_it=1000)
    cg_solver.setFromOptions()

    t0 = time.time()
    cg_solver.setUp()
    pc_setup_time = time.time() - t0

    x = problem.A.createVecRight()
    b = problem.A.createVecRight()

    rows = []
    for dof in dofs:
        b.array_w[:] = 0.0
        b.array_w[int(dof)] = force

        x.set(0.0)

        t0 = time.time()
        cg_solver.solve(b, x)
        solve_time = time.time() - t0

        reason = cg_solver.getConvergedReason()
        rows.append(
            {
                "dof": int(dof),
                "solve_time": solve_time,
                "iterations": cg_solver.getIterationNumber(),
                "reason": reason,
                "residual": cg_solver.getResidualNorm(),
                "converged": reason > 0,
            }
        )

    cg_solver.destroy()

    return {"pc_setup_time": pc_setup_time, "rows": rows}


def summarize_probe_rows(rows: list[dict]) -> dict:
    """Reduce a list of per-DOF probe results to: convergence rate, and the
    average solve time / iterations / residual over the converged solves
    only."""

    n_rhs = len(rows)
    converged_rows = [r for r in rows if r["converged"]]
    n_converged = len(converged_rows)

    return {
        "n_rhs": n_rhs,
        "n_converged": n_converged,
        "conv_rate": n_converged / n_rhs if n_rhs > 0 else 0.0,
        "avg_solve_time": (
            float(np.mean([r["solve_time"] for r in converged_rows]))
            if converged_rows
            else float("nan")
        ),
        "avg_iterations": (
            float(np.mean([r["iterations"] for r in converged_rows]))
            if converged_rows
            else float("nan")
        ),
        "avg_residual": (
            float(np.mean([r["residual"] for r in converged_rows]))
            if converged_rows
            else float("nan")
        ),
    }


def run_multi_rhs_sweep(
    Nr: int, nt_values: list[int], n_rhs: int, force: float
) -> list[dict]:
    """Fix Nr and sweep Nt (with Np = 2.5 * Nt), and on each mesh compare
    CG+Hypre vs CG+GAMG over `n_rhs` unit-force right-hand sides at
    different contact-zone DOFs, reusing one PC setup each."""

    results = []

    for Nt in nt_values:
        Np = int(round(2.5 * Nt))

        print(f"\n--- Sweep case: Nr={Nr}, Nt={Nt}, Np={Np} ---")

        problem, num_nodes, num_dofs, Ns, Ic = build_problem(Nr, Nt, Np)
        probe_dofs = select_probe_dofs(Ic, n_rhs)
        print(
            f"nodes={num_nodes}, dofs={num_dofs}, |Ic|={len(Ic)}, "
            f"probing {len(probe_dofs)} right-hand sides"
        )

        hypre = run_cg_probe(problem, "hypre", probe_dofs, force=force)
        gamg = run_cg_probe(problem, "gamg", probe_dofs, force=force)

        hypre_stats = summarize_probe_rows(hypre["rows"])
        gamg_stats = summarize_probe_rows(gamg["rows"])

        print(
            f"Hypre: converged {hypre_stats['n_converged']}/{hypre_stats['n_rhs']}, "
            f"avg solve = {hypre_stats['avg_solve_time']:.4e} s, "
            f"avg iters = {hypre_stats['avg_iterations']:.1f}, "
            f"avg residual = {hypre_stats['avg_residual']:.3e}"
        )
        print(
            f"GAMG:  converged {gamg_stats['n_converged']}/{gamg_stats['n_rhs']}, "
            f"avg solve = {gamg_stats['avg_solve_time']:.4e} s, "
            f"avg iters = {gamg_stats['avg_iterations']:.1f}, "
            f"avg residual = {gamg_stats['avg_residual']:.3e}"
        )

        results.append(
            {
                "Nr": Nr,
                "Nt": Nt,
                "Np": Np,
                "Ns": Ns,
                "num_nodes": num_nodes,
                "num_dofs": num_dofs,
                "hypre_pc_setup_time": hypre["pc_setup_time"],
                "hypre_conv_rate": hypre_stats["conv_rate"],
                "hypre_avg_solve_time": hypre_stats["avg_solve_time"],
                "hypre_avg_iterations": hypre_stats["avg_iterations"],
                "hypre_avg_residual": hypre_stats["avg_residual"],
                "gamg_pc_setup_time": gamg["pc_setup_time"],
                "gamg_conv_rate": gamg_stats["conv_rate"],
                "gamg_avg_solve_time": gamg_stats["avg_solve_time"],
                "gamg_avg_iterations": gamg_stats["avg_iterations"],
                "gamg_avg_residual": gamg_stats["avg_residual"],
            }
        )

    return results


def _fmt_avg_iterations(x: float) -> str:
    """Round to the nearest iteration count, without crashing on NaN
    (no converged solves)."""

    return "nan" if np.isnan(x) else str(round(x))


def print_multi_rhs_sweep_table(results: list[dict]) -> None:
    headers = [
        "Nt",
        "Np",
        "dofs",
        "Hypre conv",
        "Hypre avg s[s]",
        "Hypre avg it",
        "Hypre avg resid",
        "GAMG conv",
        "GAMG avg s[s]",
        "GAMG avg it",
        "GAMG avg resid",
    ]

    col_w = 16
    print("\n" + "=" * (col_w * len(headers)))
    print("".join(h.rjust(col_w) for h in headers))
    print("-" * (col_w * len(headers)))

    for row in results:
        cells = [
            str(row["Nt"]),
            str(row["Np"]),
            str(row["num_dofs"]),
            f"{row['hypre_conv_rate'] * 100:.0f}%",
            f"{row['hypre_avg_solve_time']:.3e}",
            _fmt_avg_iterations(row["hypre_avg_iterations"]),
            f"{row['hypre_avg_residual']:.3e}",
            f"{row['gamg_conv_rate'] * 100:.0f}%",
            f"{row['gamg_avg_solve_time']:.3e}",
            _fmt_avg_iterations(row["gamg_avg_iterations"]),
            f"{row['gamg_avg_residual']:.3e}",
        ]
        print("".join(c.rjust(col_w) for c in cells))

    print("=" * (col_w * len(headers)))


def run_sweep(Nr: int, nt_values: list[int], n_solves: int) -> list[dict]:
    """Fix Nr and sweep Nt (with Np = 2.5 * Nt), running LU, CG+Hypre and
    CG+GAMG on each mesh size and tracking setup/factorisation time, solve
    time, iteration counts and convergence info for all three solvers."""

    results = []

    for Nt in nt_values:
        Np = int(round(2.5 * Nt))

        print(f"\n--- Sweep case: Nr={Nr}, Nt={Nt}, Np={Np} ---")

        problem, num_nodes, num_dofs, Ns, _Ic = build_problem(Nr, Nt, Np)
        print(f"nodes={num_nodes}, dofs={num_dofs}, Ns={Ns}")

        lu = run_lu(problem, n_solves)
        print(
            f"LU: factor time = {lu['factor_time']:.4e} s, "
            f"solve time = {lu['solve_time']:.4e} s"
        )

        cg = run_cg(problem, n_solves, "hypre")
        print(
            f"CG+Hypre: PC setup time = {cg['pc_setup_time']:.4e} s, "
            f"solve time = {cg['solve_time']:.4e} s, "
            f"iterations = {cg['iterations']:.1f}, "
            f"reason = {cg['reason']}, "
            f"residual = {cg['residual']:.3e}"
        )

        gamg = run_cg(problem, n_solves, "gamg")
        print(
            f"CG+GAMG: PC setup time = {gamg['pc_setup_time']:.4e} s, "
            f"solve time = {gamg['solve_time']:.4e} s, "
            f"iterations = {gamg['iterations']:.1f}, "
            f"reason = {gamg['reason']}, "
            f"residual = {gamg['residual']:.3e}"
        )

        results.append(
            {
                "Nr": Nr,
                "Nt": Nt,
                "Np": Np,
                "Ns": Ns,
                "num_nodes": num_nodes,
                "num_dofs": num_dofs,
                "lu_factor_time": lu["factor_time"],
                "lu_solve_time": lu["solve_time"],
                "lu_solve_time_per_it": lu["solve_time"],
                "lu_converged": lu["converged"],
                "cg_pc_setup_time": cg["pc_setup_time"],
                "cg_solve_time": cg["solve_time"],
                "cg_solve_time_per_it": cg["solve_time_per_it"],
                "cg_iterations": cg["iterations"],
                "cg_reason": cg["reason"],
                "cg_residual": cg["residual"],
                "cg_converged": cg["converged"],
                "gamg_pc_setup_time": gamg["pc_setup_time"],
                "gamg_solve_time": gamg["solve_time"],
                "gamg_solve_time_per_it": gamg["solve_time_per_it"],
                "gamg_iterations": gamg["iterations"],
                "gamg_reason": gamg["reason"],
                "gamg_residual": gamg["residual"],
                "gamg_converged": gamg["converged"],
            }
        )

    return results


def print_gamg_hypre_table(results: list[dict]) -> None:
    """Compare CG+Hypre and CG+GAMG: PC setup time, solve time per
    iteration, total solve time, iteration count and final residual."""

    headers = [
        "Nt",
        "Np",
        "dofs",
        "Hypre setup[s]",
        "Hypre s/it[s]",
        "Hypre total[s]",
        "Hypre iters",
        "Hypre resid",
        "GAMG setup[s]",
        "GAMG s/it[s]",
        "GAMG total[s]",
        "GAMG iters",
        "GAMG resid",
    ]

    col_w = 15
    print("\n" + "=" * (col_w * len(headers)))
    print("".join(h.rjust(col_w) for h in headers))
    print("-" * (col_w * len(headers)))

    for row in results:
        cg_total = row["cg_pc_setup_time"] + row["cg_solve_time"]
        gamg_total = row["gamg_pc_setup_time"] + row["gamg_solve_time"]

        cells = [
            str(row["Nt"]),
            str(row["Np"]),
            str(row["num_dofs"]),
            f"{row['cg_pc_setup_time']:.3e}",
            f"{row['cg_solve_time_per_it']:.3e}",
            f"{cg_total:.3e}",
            f"{row['cg_iterations']:.1f}",
            f"{row['cg_residual']:.3e}",
            f"{row['gamg_pc_setup_time']:.3e}",
            f"{row['gamg_solve_time_per_it']:.3e}",
            f"{gamg_total:.3e}",
            f"{row['gamg_iterations']:.1f}",
            f"{row['gamg_residual']:.3e}",
        ]
        print("".join(c.rjust(col_w) for c in cells))

    print("=" * (col_w * len(headers)))


def run_gamg_nullspace_sweep(
    Nr: int, nt_values: list[int], n_solves: int
) -> list[dict]:
    """Fix Nr and sweep Nt (with Np = 2.5 * Nt), comparing plain CG+GAMG
    against CG+GAMG with a near-nullspace of rigid body modes (SPD hint +
    Chebyshev/Jacobi smoothing) on each mesh size."""

    results = []

    for Nt in nt_values:
        Np = int(round(2.5 * Nt))

        print(f"\n--- Sweep case: Nr={Nr}, Nt={Nt}, Np={Np} ---")

        problem, num_nodes, num_dofs, Ns, _Ic = build_problem(Nr, Nt, Np)
        print(f"nodes={num_nodes}, dofs={num_dofs}, Ns={Ns}")

        gamg = run_cg(problem, n_solves, "gamg")
        print(
            f"CG+GAMG: PC setup time = {gamg['pc_setup_time']:.4e} s, "
            f"solve time = {gamg['solve_time']:.4e} s, "
            f"iterations = {gamg['iterations']:.1f}, "
            f"reason = {gamg['reason']}, "
            f"residual = {gamg['residual']:.3e}"
        )

        gamg_ns = run_cg_gamg_nullspace(problem, n_solves)
        print(
            f"CG+GAMG+NS: PC setup time = {gamg_ns['pc_setup_time']:.4e} s, "
            f"solve time = {gamg_ns['solve_time']:.4e} s, "
            f"iterations = {gamg_ns['iterations']:.1f}, "
            f"reason = {gamg_ns['reason']}, "
            f"residual = {gamg_ns['residual']:.3e}"
        )

        results.append(
            {
                "Nr": Nr,
                "Nt": Nt,
                "Np": Np,
                "Ns": Ns,
                "num_nodes": num_nodes,
                "num_dofs": num_dofs,
                "gamg_pc_setup_time": gamg["pc_setup_time"],
                "gamg_solve_time": gamg["solve_time"],
                "gamg_solve_time_per_it": gamg["solve_time_per_it"],
                "gamg_iterations": gamg["iterations"],
                "gamg_residual": gamg["residual"],
                "gamg_converged": gamg["converged"],
                "gamg_ns_pc_setup_time": gamg_ns["pc_setup_time"],
                "gamg_ns_solve_time": gamg_ns["solve_time"],
                "gamg_ns_solve_time_per_it": gamg_ns["solve_time_per_it"],
                "gamg_ns_iterations": gamg_ns["iterations"],
                "gamg_ns_residual": gamg_ns["residual"],
                "gamg_ns_converged": gamg_ns["converged"],
            }
        )

    return results


def print_gamg_nullspace_table(results: list[dict]) -> None:
    """Compare plain CG+GAMG and CG+GAMG+nullspace: PC setup time, solve
    time per iteration, total solve time, iteration count and residual."""

    headers = [
        "Nt",
        "Np",
        "dofs",
        "GAMG setup[s]",
        "GAMG s/it[s]",
        "GAMG total[s]",
        "GAMG iters",
        "GAMG resid",
        "GAMG+NS setup[s]",
        "GAMG+NS s/it[s]",
        "GAMG+NS total[s]",
        "GAMG+NS iters",
        "GAMG+NS resid",
    ]

    col_w = 17
    print("\n" + "=" * (col_w * len(headers)))
    print("".join(h.rjust(col_w) for h in headers))
    print("-" * (col_w * len(headers)))

    for row in results:
        gamg_total = row["gamg_pc_setup_time"] + row["gamg_solve_time"]
        gamg_ns_total = row["gamg_ns_pc_setup_time"] + row["gamg_ns_solve_time"]

        cells = [
            str(row["Nt"]),
            str(row["Np"]),
            str(row["num_dofs"]),
            f"{row['gamg_pc_setup_time']:.3e}",
            f"{row['gamg_solve_time_per_it']:.3e}",
            f"{gamg_total:.3e}",
            f"{row['gamg_iterations']:.1f}",
            f"{row['gamg_residual']:.3e}",
            f"{row['gamg_ns_pc_setup_time']:.3e}",
            f"{row['gamg_ns_solve_time_per_it']:.3e}",
            f"{gamg_ns_total:.3e}",
            f"{row['gamg_ns_iterations']:.1f}",
            f"{row['gamg_ns_residual']:.3e}",
        ]
        print("".join(c.rjust(col_w) for c in cells))

    print("=" * (col_w * len(headers)))


def _overlay_unconverged(ax, dofs, values, converged, label_used: bool) -> bool:
    """Overlay a distinct marker on points where the solve did not converge.
    Returns whether the "unconverged" legend label has now been used."""

    bad_dofs = [d for d, c in zip(dofs, converged) if not c]
    bad_vals = [v for v, c in zip(values, converged) if not c]

    if not bad_dofs:
        return label_used

    ax.plot(
        bad_dofs,
        bad_vals,
        marker="x",
        color="red",
        markersize=14,
        markeredgewidth=3,
        linestyle="None",
        label=None if label_used else "unconverged",
        zorder=5,
    )
    return True


def plot_lu_factor_time(
    results: list[dict], filename: str = "lu_factor_time.png"
) -> None:
    import matplotlib.pyplot as plt

    dofs = [row["num_dofs"] for row in results]
    factor_times = [row["lu_factor_time"] for row in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dofs, factor_times, "o-", color="tab:blue", label="LU factorisation time")
    ax.set_xlabel("Number of DOFs")
    ax.set_ylabel("Time [s]")
    ax.set_title("LU factorisation time vs problem size")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved LU factorisation time plot to {filename}")


def plot_pc_setup_times(
    results: list[dict], filename: str = "pc_setup_times.png"
) -> None:
    import matplotlib.pyplot as plt

    dofs = [row["num_dofs"] for row in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        dofs,
        [row["cg_pc_setup_time"] for row in results],
        "s-",
        color="tab:orange",
        label="Hypre PC setup time",
    )
    ax.plot(
        dofs,
        [row["gamg_pc_setup_time"] for row in results],
        "^-",
        color="tab:green",
        label="GAMG PC setup time",
    )
    ax.set_xlabel("Number of DOFs")
    ax.set_ylabel("Time [s]")
    ax.set_title("Preconditioner setup time vs problem size")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved PC setup time plot to {filename}")


def plot_per_iteration_solve_time(
    results: list[dict], filename: str = "per_iteration_solve_time.png"
) -> None:
    import matplotlib.pyplot as plt

    dofs = [row["num_dofs"] for row in results]

    series = [
        ("lu_solve_time_per_it", "lu_converged", "LU (direct solve)", "o", "tab:blue"),
        ("cg_solve_time_per_it", "cg_converged", "CG + Hypre", "s", "tab:orange"),
        ("gamg_solve_time_per_it", "gamg_converged", "CG + GAMG", "^", "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    label_used = False

    for value_key, conv_key, label, marker, color in series:
        values = [row[value_key] for row in results]
        converged = [row[conv_key] for row in results]

        ax.plot(dofs, values, marker=marker, linestyle="-", color=color, label=label)
        label_used = _overlay_unconverged(ax, dofs, values, converged, label_used)

    ax.set_xlabel("Number of DOFs")
    ax.set_ylabel("Time per iteration [s]")
    ax.set_title("Per-iteration solve time vs problem size")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved per-iteration solve time plot to {filename}")


def plot_total_time(results: list[dict], filename: str = "total_time.png") -> None:
    import matplotlib.pyplot as plt

    dofs = [row["num_dofs"] for row in results]

    series = [
        ("lu_factor_time", "lu_solve_time", "lu_converged", "LU", "o", "tab:blue"),
        (
            "cg_pc_setup_time",
            "cg_solve_time",
            "cg_converged",
            "CG + Hypre",
            "s",
            "tab:orange",
        ),
        (
            "gamg_pc_setup_time",
            "gamg_solve_time",
            "gamg_converged",
            "CG + GAMG",
            "^",
            "tab:green",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    label_used = False

    for setup_key, solve_key, conv_key, label, marker, color in series:
        values = [row[setup_key] + row["Ns"] * row[solve_key] for row in results]
        converged = [row[conv_key] for row in results]

        ax.plot(dofs, values, marker=marker, linestyle="-", color=color, label=label)
        label_used = _overlay_unconverged(ax, dofs, values, converged, label_used)

    ax.set_xlabel("Number of DOFs")
    ax.set_ylabel("Total time (setup + Ns * solve) [s]")
    ax.set_title("Total time (setup + Ns probing solves) vs problem size")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Saved total time plot to {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a 3D linear-elasticity problem")

    parser.add_argument(
        "--solver",
        type=str,
        choices=["lu", "cg", "all"],
        default="all",
        help="Solver to use: lu, cg, or all",
    )

    parser.add_argument(
        "--pc",
        type=str,
        choices=["hypre", "gamg", "both"],
        default="both",
        help="Preconditioner(s) to use with the CG solver: hypre, gamg, or both",
    )

    parser.add_argument(
        "--Nr",
        type=int,
        default=3,
        help="Number of cells along r",
    )

    parser.add_argument(
        "--Nt",
        type=int,
        default=40,
        help="Number of cells along theta",
    )

    parser.add_argument(
        "--Np",
        type=int,
        default=80,
        help="Number of cells along phi",
    )

    parser.add_argument(
        "--nsol",
        type=int,
        default=3,
        help="Number of repeated solves for timing",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run a mesh-size sweep instead of a single case: Nr is kept fixed "
            "and Nt is increased from --sweep-nt-min to --sweep-nt-max (step "
            "--sweep-nt-step), with Np = 2.5 * Nt each time."
        ),
    )

    parser.add_argument("--sweep-nt-min", type=int, default=20)
    parser.add_argument("--sweep-nt-max", type=int, default=140)
    parser.add_argument("--sweep-nt-step", type=int, default=20)

    parser.add_argument(
        "--multi-rhs",
        action="store_true",
        help=(
            "Compare CG+Hypre vs CG+GAMG over the same mesh sweep as "
            "--sweep (Nr fixed, Nt from --sweep-nt-min to --sweep-nt-max). "
            "On each mesh, reuse one PC setup each across --n-rhs "
            "unit-force right-hand sides at different contact-zone DOFs, "
            "and report the convergence rate and the average solve time / "
            "iterations / residual over the converged solves."
        ),
    )
    parser.add_argument(
        "--n-rhs",
        type=int,
        default=10,
        help="Number of different right-hand sides for --multi-rhs",
    )
    parser.add_argument(
        "--rhs-force",
        type=float,
        default=1e5,
        help="Magnitude of the unit force applied at each probed DOF for --multi-rhs",
    )

    parser.add_argument(
        "--gamg-nullspace",
        action="store_true",
        help=(
            "Compare plain CG+GAMG vs CG+GAMG with a near-nullspace of the "
            "3D elasticity rigid body modes (SPD matrix hint + "
            "Chebyshev/Jacobi multigrid smoothing), over the same mesh "
            "sweep as --sweep."
        ),
    )

    args = parser.parse_args()

    solver = args.solver
    pc = args.pc
    n_solves = args.nsol

    if args.sweep:
        nt_values = list(
            range(args.sweep_nt_min, args.sweep_nt_max + 1, args.sweep_nt_step)
        )
        print(f"Sweeping Nt = {nt_values} with Nr={args.Nr} fixed, Np = 2.5 * Nt")
        print(
            "Sweep always runs LU, CG+Hypre and CG+GAMG to build the comparison plots/table"
        )
        print(f"Number of solves per case: {n_solves}")

        results = run_sweep(args.Nr, nt_values, n_solves)
        print_gamg_hypre_table(results)
        plot_lu_factor_time(results)
        plot_pc_setup_times(results)
        plot_per_iteration_solve_time(results)
        plot_total_time(results)
        return

    if args.multi_rhs:
        nt_values = list(
            range(args.sweep_nt_min, args.sweep_nt_max + 1, args.sweep_nt_step)
        )
        print(f"Sweeping Nt = {nt_values} with Nr={args.Nr} fixed, Np = 2.5 * Nt")
        print(
            f"Comparing CG+Hypre vs CG+GAMG over {args.n_rhs} unit-force "
            f"({args.rhs_force:.1e}) right-hand sides at contact DOFs, per mesh"
        )

        results = run_multi_rhs_sweep(args.Nr, nt_values, args.n_rhs, args.rhs_force)
        print_multi_rhs_sweep_table(results)
        return

    if args.gamg_nullspace:
        nt_values = list(
            range(args.sweep_nt_min, args.sweep_nt_max + 1, args.sweep_nt_step)
        )
        print(f"Sweeping Nt = {nt_values} with Nr={args.Nr} fixed, Np = 2.5 * Nt")
        print("Comparing plain CG+GAMG vs CG+GAMG with rigid-body near-nullspace")
        print(f"Number of solves per case: {n_solves}")

        results = run_gamg_nullspace_sweep(args.Nr, nt_values, n_solves)
        print_gamg_nullspace_table(results)
        return

    Nr, Nt, Np = args.Nr, args.Nt, args.Np

    print(f"Grid size: Nr={Nr}, Nt={Nt}, Np={Np}")
    print(f"Solver: {solver}, PC: {pc}")
    print(f"Number of solves: {n_solves}")

    problem, num_nodes, num_dofs, Ns, _Ic = build_problem(Nr, Nt, Np)

    print(f"Total number of nodes: {num_nodes}")
    print(f"Ns (trimmed nodes along theta): {Ns}")
    print(f"Total number of DOFs: {num_dofs}")

    # ============================================================
    # LU solver
    # ============================================================

    if solver in ["lu", "all"]:
        print("======================LU====================")
        print("Solving with LU...")

        lu = run_lu(problem, n_solves)

        print(f"LU Factorisation CPU time = {lu['factor_time']}")
        print(f"LU average solve CPU time = {lu['solve_time']}")
        print(f"LU total average time = {lu['factor_time'] + lu['solve_time']}")
        print()

    # ============================================================
    # CG solver(s)
    # ============================================================

    if solver in ["cg", "all"]:
        if pc in ["hypre", "both"]:
            print("======================CG + Hypre===============")
            print("Solving with CG + hypre...")

            cg = run_cg(problem, n_solves, "hypre")

            print(f"CG + hypre PC setup CPU time = {cg['pc_setup_time']}")
            print(f"CG + hypre average solve CPU time = {cg['solve_time']}")
            print(
                f"CG + hypre total average time = {cg['pc_setup_time'] + cg['solve_time']}"
            )
            print(f"CG + hypre average iterations = {cg['iterations']}")
            print(f"CG + hypre last convergence reason = {cg['reason']}")
            print(f"CG + hypre last residual norm = {cg['residual']}")
            print()

        if pc in ["gamg", "both"]:
            print("======================CG + GAMG================")
            print("Solving with CG + gamg...")

            gamg = run_cg(problem, n_solves, "gamg")

            print(f"CG + gamg PC setup CPU time = {gamg['pc_setup_time']}")
            print(f"CG + gamg average solve CPU time = {gamg['solve_time']}")
            print(
                f"CG + gamg total average time = {gamg['pc_setup_time'] + gamg['solve_time']}"
            )
            print(f"CG + gamg average iterations = {gamg['iterations']}")
            print(f"CG + gamg last convergence reason = {gamg['reason']}")
            print(f"CG + gamg last residual norm = {gamg['residual']}")
            print()

    print("All requested solves completed.")


if __name__ == "__main__":
    main()

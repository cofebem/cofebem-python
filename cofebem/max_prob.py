import numpy as np
import matplotlib.pyplot as plt
import resource
from pathlib import Path
import sys
import time
from time import perf_counter
from typing import Any

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import create_box, locate_entities_boundary, meshtags, CellType
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    form,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    LinearProblem,
    assemble_matrix,
    assemble_vector,
    apply_lifting,
)
from ufl import (
    Identity,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    dx,
    Measure,
)


def problem_setup(
    mesh,
    element,
    material,
    locate_u,
    locate_t,
    locate_c,
    f=None,
    t0=None,
    u0=None,
    petsc_options=None,
):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    gdim = mesh.geometry.dim

    # Material
    E = material["E"]
    nu = material["nu"]
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    # Function space and forms
    element_type, element_order = element
    V = functionspace(mesh, (element_type, element_order, (gdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(gdim) + 2.0 * mu * epsilon(w)

    if f is None:
        f = np.zeros(gdim)
    if t0 is None:
        t0 = np.zeros(gdim)
    if u0 is None:
        u0 = np.zeros(gdim)

    f = Constant(mesh, np.asarray(f, dtype=PETSc.ScalarType))
    t0 = Constant(mesh, np.asarray(t0, dtype=PETSc.ScalarType))
    u0 = Constant(mesh, np.asarray(u0, dtype=PETSc.ScalarType))

    tc = Function(V)
    tc.name = "tc"

    Gamma_u = locate_entities_boundary(mesh, fdim, locate_u)
    Gamma_t = locate_entities_boundary(mesh, fdim, locate_t)
    Gamma_c = locate_entities_boundary(mesh, fdim, locate_c)

    Gamma_u_id = 0
    Gamma_t_id = 1
    Gamma_c_id = 2

    facet_indices = np.hstack([Gamma_u, Gamma_t, Gamma_c]).astype(np.int32)
    facet_values = np.hstack(
        [
            np.full(Gamma_u.size, Gamma_u_id, dtype=np.int32),
            np.full(Gamma_t.size, Gamma_t_id, dtype=np.int32),
            np.full(Gamma_c.size, Gamma_c_id, dtype=np.int32),
        ]
    )

    has_overlap = np.unique(facet_indices).size != facet_indices.size
    if mesh.comm.allreduce(int(has_overlap)):
        raise ValueError("Gamma_u, Gamma_t and Gamma_c must not share facets.")

    order = np.argsort(facet_indices)
    mt = meshtags(mesh, fdim, facet_indices[order], facet_values[order])
    ds = Measure("ds", domain=mesh, subdomain_data=mt)

    # Dirichlet condition
    Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)
    bc = dirichletbc(u0, Gamma_u_dofs, V)
    bcs = [bc]

    # Variational problem
    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(f, v) * dx + inner(t0, v) * ds(Gamma_t_id) + inner(tc, v) * ds(Gamma_c_id)

    if petsc_options is None:
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="sc_",
        petsc_options=dict(petsc_options),
    )
    problem.u.name = "u"

    return problem, mt, f, t0, tc


def assemble(problem):
    problem._A.zeroEntries()
    assemble_matrix(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()

    with problem._b.localForm() as local_b:
        local_b.set(0.0)

    assemble_vector(problem._b, problem._L)
    apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
    problem._b.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )

    for bc in problem.bcs:
        bc.set(problem._b.array_w)


def cholesky(
    A: PETSc.Mat,
    b: list[PETSc.Vec],
    compression: bool = False,
    blr_tol: float = 1e-4,
) -> dict[str, Any]:

    if A.comm.getSize() != 1:
        raise ValueError("This benchmark function requires one MPI process.")

    A.setOption(PETSc.Mat.Option.SPD, True)

    ksp = PETSc.KSP().create(comm=A.comm)

    ksp.setOperators(A)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("cholesky")
    pc.setFactorSolverType("mumps")

    t0 = perf_counter()
    pc.setFactorSetUpSolverType()
    factor = pc.getFactorMatrix()
    factor.setMumpsIcntl(35, 2 if compression else 0)
    if compression:
        factor.setMumpsCntl(7, float(blr_tol))

    ksp.setUp()
    factorization_s = perf_counter() - t0

    return ksp, factorization_s


def cg(
    A: PETSc.Mat,
    b: list[PETSc.Vec],
    precond: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_iter: int = 10_000,
) -> dict[str, Any]:

    if A.comm.getSize() != 1:
        raise ValueError("This function requires one MPI process.")

    ksp = PETSc.KSP().create(comm=A.comm)

    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_iter)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    pc = ksp.getPC()
    if precond:
        pc.setType("hypre")
        pc.setHYPREType("boomeramg")
    else:
        pc.setType("none")

    t0 = perf_counter()
    ksp.setUp()
    setup_s = perf_counter() - t0

    return ksp, setup_s


def test1():
    N = [20, 40, 60, 80]
    cases = [
        ("Cholesky", cholesky, {}),
        ("Cholesky BLR", cholesky, {"compression": True, "blr_tol": 1e-4}),
        ("CG", cg, {"max_iter": 1000}),
        ("CG + AMG", cg, {"precond": True, "max_iter": 1000}),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for n, ax in zip(N, axes.flat):
        mesh = create_box(
            MPI.COMM_WORLD,
            [np.zeros(3), np.ones(3)],
            [n, n, n],
            cell_type=CellType.tetrahedron,
        )
        problem, mt, _, _, _ = problem_setup(
            mesh,
            element=("Lagrange", 1),
            material={"E": 1e9, "nu": 0.3},
            locate_u=lambda x: np.isclose(x[2], 0.0),
            locate_t=lambda x: np.isclose(x[0], 0.0),
            locate_c=lambda x: np.isclose(x[2], 1.0),
        )
        assemble(problem)

        A = problem.A
        V = problem.u.function_space
        bs = V.dofmap.index_map_bs
        contact_nodes = locate_dofs_topological(V, mesh.topology.dim - 1, mt.find(2))
        nc = contact_nodes.size
        selected = np.random.default_rng(0).choice(contact_nodes, size=3, replace=False)

        rhs_list = []
        for node in selected:
            rhs = A.createVecLeft()
            rhs.set(0.0)
            rhs.setValue(bs * int(node) + 2, -1.0)
            rhs.assemble()
            rhs_list.append(rhs)

        x = A.createVecRight()
        setups, estimates, labels, converged = [], [], [], []

        for name, solver, kwargs in cases:
            print(f"Mesh {n} x {n} x {n}: {name}", flush=True)
            ksp, setup_s = solver(A, rhs_list, **kwargs)
            solves_s, reasons = [], []

            for rhs in rhs_list:
                x.set(0.0)
                t0 = perf_counter()
                ksp.solve(rhs, x)
                solves_s.append(perf_counter() - t0)
                reasons.append(int(ksp.getConvergedReason()))

            estimated_s = nc * float(np.mean(solves_s))
            ok = all(reason > 0 for reason in reasons)
            setups.append(setup_s)
            estimates.append(estimated_s)
            converged.append(ok)
            labels.append(name if ok else name + "*")
            print(
                f"  setup={setup_s:.3f} s, estimated solves={estimated_s:.3f} s, "
                f"converged={ok}",
                flush=True,
            )
            ksp.destroy()

        x.destroy()
        for rhs in rhs_list:
            rhs.destroy()

        positions = np.arange(len(cases))
        ax.bar(positions, setups, color="#2563eb", label="Setup / factorization")
        bars = ax.bar(
            positions,
            estimates,
            bottom=setups,
            color="#f59e0b",
            label="Estimated Nc solves (mean of 3)",
        )
        for bar, ok in zip(bars, converged):
            if not ok:
                bar.set_hatch("//")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Time [s]")
        ax.set_title(f"{n} x {n} x {n} | {A.getSize()[0]:,} DOFs | Nc={nc}")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.text(
        0.5,
        0.02,
        "* At least one sampled solve did not converge; the time estimate is for incomplete solves.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    plt.show()
    Path("results").mkdir(exist_ok=True)
    fig.savefig("../results/max_prob/solver_time_breakdown.png", dpi=200)
    plt.close(fig)


def _memory_case(n, solver, kwargs, queue):
    mesh = create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.ones(3)],
        [n, n, n],
        cell_type=CellType.tetrahedron,
    )
    problem, mt, _, _, _ = problem_setup(
        mesh,
        element=("Lagrange", 1),
        material={"E": 1e9, "nu": 0.3},
        locate_u=lambda x: np.isclose(x[2], 0.0),
        locate_t=lambda x: np.isclose(x[0], 0.0),
        locate_c=lambda x: np.isclose(x[2], 1.0),
    )
    assemble(problem)

    A = problem.A
    V = problem.u.function_space
    bs = V.dofmap.index_map_bs
    contact_nodes = locate_dofs_topological(V, mesh.topology.dim - 1, mt.find(2))
    selected = np.random.default_rng(0).choice(contact_nodes, size=3, replace=False)

    rhs_list = []
    for node in selected:
        rhs = A.createVecLeft()
        rhs.set(0.0)
        rhs.setValue(bs * int(node) + 2, -1.0)
        rhs.assemble()
        rhs_list.append(rhs)

    ksp, _ = solver(A, rhs_list, **kwargs)
    x = A.createVecRight()
    reasons = []
    for rhs in rhs_list:
        x.set(0.0)
        ksp.solve(rhs, x)
        reasons.append(int(ksp.getConvergedReason()))

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
    ndofs = A.getSize()[0]

    x.destroy()
    ksp.destroy()
    for rhs in rhs_list:
        rhs.destroy()

    queue.put((ndofs, peak_mib, all(reason > 0 for reason in reasons)))


def test2():
    import multiprocessing as mp

    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("This benchmark requires one MPI process.")

    N = [10, 20, 30, 40]
    cases = [
        ("Cholesky", cholesky, {}),
        ("Cholesky BLR", cholesky, {"compression": True, "blr_tol": 1e-4}),
        ("CG", cg, {"max_iter": 500}),
        ("CG + BoomerAMG", cg, {"precond": True, "max_iter": 5000}),
    ]
    ctx = mp.get_context("spawn")
    fig, ax = plt.subplots(figsize=(8, 5))
    failed = False

    for name, solver, kwargs in cases:
        dofs, peaks, converged = [], [], []
        for n in N:
            print(f"Memory: mesh {n} x {n} x {n}, {name}", flush=True)
            queue = ctx.Queue()
            process = ctx.Process(target=_memory_case, args=(n, solver, kwargs, queue))
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"Memory run failed: {name}, mesh {n} x {n} x {n}")

            ndofs, peak_mib, ok = queue.get()
            queue.close()
            queue.join_thread()
            process.close()
            dofs.append(ndofs)
            peaks.append(peak_mib)
            converged.append(ok)
            print(f"  peak={peak_mib:.1f} MiB, converged={ok}", flush=True)

        dofs, peaks = np.asarray(dofs), np.asarray(peaks)
        ok = np.asarray(converged, dtype=bool)
        (line,) = ax.plot(dofs, peaks, "-o", label=name)
        ax.scatter(dofs[~ok], peaks[~ok], marker="x", s=90, color=line.get_color())
        failed = failed or bool(np.any(~ok))

    if failed:
        ax.plot([], [], "kx", markersize=8, label="Not all 3 solves converged")
    ax.set_xlabel("Number of FE DOFs")
    ax.set_ylabel("Peak process RSS [MiB]")
    ax.set_title("Memory footprint | meshes 10³, 20³, 30³, 40³")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.text(
        0.5,
        0.02,
        "Includes mesh, assembly, matrix, and solver; 3 solves per case.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()

    # Path("results").mkdir(exist_ok=True)
    # fig.savefig("results/solver_memory.png", dpi=200)
    # plt.close(fig)


def test3():
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("This benchmark requires one MPI process.")

    mesh = create_box(
        MPI.COMM_WORLD,
        [np.zeros(3), np.ones(3)],
        [40, 40, 40],
        cell_type=CellType.tetrahedron,
    )
    problem, mt, _, _, _ = problem_setup(
        mesh,
        element=("Lagrange", 1),
        material={"E": 1e9, "nu": 0.3},
        locate_u=lambda x: np.isclose(x[2], 0.0),
        locate_t=lambda x: np.isclose(x[0], 0.0),
        locate_c=lambda x: np.isclose(x[2], 1.0),
    )
    assemble(problem)

    A = problem.A
    V = problem.u.function_space
    bs = V.dofmap.index_map_bs
    contact_nodes = locate_dofs_topological(V, mesh.topology.dim - 1, mt.find(2))
    nc = contact_nodes.size
    selected = np.random.default_rng(0).choice(contact_nodes, size=3, replace=False)

    rhs_list = []
    for node in selected:
        rhs = A.createVecLeft()
        rhs.set(0.0)
        rhs.setValue(bs * int(node) + 2, -1.0)
        rhs.assemble()
        rhs_list.append(rhs)

    cases = [
        ("Cholesky", cholesky, {}),
        ("Cholesky BLR", cholesky, {"compression": True, "blr_tol": 1e-4}),
        ("CG", cg, {"max_iter": 500}),
        ("CG + BoomerAMG", cg, {"precond": True, "max_iter": 5000}),
    ]
    x = A.createVecRight()
    results = {}

    for name, setup, kwargs in cases:
        print(f"Speedup: mesh 40 x 40 x 40, {name}", flush=True)
        ksp, setup_s = setup(A, rhs_list, **kwargs)
        solves_s, reasons = [], []
        for rhs in rhs_list:
            x.set(0.0)
            t0 = perf_counter()
            ksp.solve(rhs, x)
            solves_s.append(perf_counter() - t0)
            reasons.append(int(ksp.getConvergedReason()))

        mean_s = float(np.mean(solves_s))
        ok = all(reason > 0 for reason in reasons)
        results[name] = (setup_s, mean_s, ok)
        print(
            f"  setup={setup_s:.3f} s, mean solve={mean_s:.6f} s, converged={ok}",
            flush=True,
        )
        ksp.destroy()

    x.destroy()
    for rhs in rhs_list:
        rhs.destroy()

    counts = np.arange(1, nc + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    failed = False
    for direct, iterative in [
        ("Cholesky", "CG"),
        ("Cholesky BLR", "CG + BoomerAMG"),
    ]:
        direct_setup, direct_solve, direct_ok = results[direct]
        iter_setup, iter_solve, iter_ok = results[iterative]
        speedup = (iter_setup + counts * iter_solve) / (
            direct_setup + counts * direct_solve
        )
        ok = direct_ok and iter_ok
        label = f"{direct} vs {iterative}" + ("" if ok else "*")
        ax.plot(counts, speedup, "-" if ok else ":", label=label)
        failed = failed or not ok

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ticks = sorted({1, 10, 100, 1000, nc})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(count) for count in ticks])
    ax.set_xlim(1, nc)
    ax.set_xlabel("Number of contact RHSs")
    ax.set_ylabel("Speedup = iterative total time / direct total time")
    ax.set_title(f"40 × 40 × 40 | {A.getSize()[0]:,} FE DOFs | Nc={nc}")
    ax.grid(alpha=0.3)
    ax.legend()
    note = "Totals = setup + RHS count × mean of 3 solves; speedup > 1 favors Cholesky."
    if failed:
        note += "\n* Dotted: sampled solves did not all converge; ratio uses incomplete solves."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    plt.show()
    # Path("results").mkdir(exist_ok=True)
    # fig.savefig("results/solver_speedup.png", dpi=200)
    # plt.close(fig)


def test4():
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("This benchmark requires one MPI process.")

    N = [10, 20, 30, 40]
    cases = [("CG", False), ("CG + BoomerAMG", True)]
    dofs, means, converged = [], [[], []], [[], []]

    for n in N:
        mesh = create_box(
            MPI.COMM_WORLD,
            [np.zeros(3), np.ones(3)],
            [n, n, n],
            cell_type=CellType.tetrahedron,
        )
        problem, mt, _, _, _ = problem_setup(
            mesh,
            element=("Lagrange", 1),
            material={"E": 1e9, "nu": 0.3},
            locate_u=lambda x: np.isclose(x[2], 0.0),
            locate_t=lambda x: np.isclose(x[0], 0.0),
            locate_c=lambda x: np.isclose(x[2], 1.0),
        )
        assemble(problem)

        A = problem.A
        V = problem.u.function_space
        bs = V.dofmap.index_map_bs
        dofs.append(A.getSize()[0])

        contact_nodes = locate_dofs_topological(V, mesh.topology.dim - 1, mt.find(2))
        selected = np.random.default_rng(0).choice(contact_nodes, size=3, replace=False)

        rhs_list = []
        for node in selected:
            rhs = A.createVecLeft()
            rhs.set(0.0)
            rhs.setValue(bs * int(node) + 2, -1.0)
            rhs.assemble()
            rhs_list.append(rhs)

        x = A.createVecRight()
        for i, (name, precond) in enumerate(cases):
            ksp, _ = cg(A, rhs_list, precond=precond)
            iterations, reasons = [], []

            for rhs in rhs_list:
                x.set(0.0)
                ksp.solve(rhs, x)
                iterations.append(int(ksp.getIterationNumber()))
                reasons.append(int(ksp.getConvergedReason()))

            mean_iter = float(np.mean(iterations))
            ok = all(reason > 0 for reason in reasons)
            means[i].append(mean_iter)
            converged[i].append(ok)

            print(
                f"Mesh {n} x {n} x {n}: {name}, "
                f"mean iterations={mean_iter:.1f}, converged={ok}",
                flush=True,
            )
            ksp.destroy()

        x.destroy()
        for rhs in rhs_list:
            rhs.destroy()

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(N))
    width = 0.35

    for i, (name, _) in enumerate(cases):
        bars = ax.bar(
            positions + (i - 0.5) * width,
            means[i],
            width,
            label=name,
        )
        ax.bar_label(
            bars,
            labels=[str(int(value) + 1) for value in means[i]],
            padding=4,
        )

        for bar, ok in zip(bars, converged[i]):
            if not ok:
                bar.set_hatch("//")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}³\n{ndofs:,} DOFs" for n, ndofs in zip(N, dofs)])
    ax.set_xlabel("Mesh")
    ax.set_ylabel("Mean iteration count (3 contact RHSs)")
    ax.set_yscale("log")
    ax.set_ylim(1, 2 * np.max(means))
    ax.set_title("CG vs CG + BoomerAMG")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    note = "rtol = 1e-6 | atol = 1e-8 | max iterations = 10,000"
    if not np.all(converged):
        note += "\nHatched bars: at least one sampled solve did not converge."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    plt.show()
    # Path("results").mkdir(exist_ok=True)
    # fig.savefig("results/cg_iterations.png", dpi=200)
    # plt.close(fig)


if __name__ == "__main__":
    test1()

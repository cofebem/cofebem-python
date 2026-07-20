from __future__ import annotations

import csv
import gc
import json
import multiprocessing as mp
from pathlib import Path
import signal
import tempfile
import time
from typing import Any

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    LinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
    sym,
    tr,
)

from cofebem.mesh.tyre_hex import tyre_hex_mesh

# =============================================================================
# CONFIGURATION
# =============================================================================

# Mesh sweep
NR = 3
NT_START = 20
NT_STEP = 20
NP_OVER_NT = 2.5

# Material
E = 2.5e8
NU = 0.48

# Tyre geometry
A0 = 0.20
B0 = 0.10
THICKNESS = 0.03
OX = 0.0
OZ = 0.5
THETA_CUT = np.pi / 6
FTOL = 5e-3

N_TIMING_SOLVES = 3
RHS_FORCE = 1.0e5

# Solvers
LU_FACTOR_SOLVER = "mumps"
CG_RTOL = 1.0e-6
CG_ATOL = 1.0e-8
CG_MAX_IT = 100

# Files
OUTPUT_PREFIX = "lu_vs_cg_hypre_serial_sweep"


# =============================================================================
# Problem construction
# =============================================================================


def build_problem(
    Nr: int,
    Nt: int,
    Np: int,
    work_directory: str,
) -> tuple[LinearProblem, dict[str, int], np.ndarray]:
    """Assemble the linear-elasticity problem for one tyre mesh."""

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "This benchmark must be run in serial with `python script.py`, "
            "not with mpiexec."
        )

    lmbda = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))

    mesh_filename = str(Path(work_directory) / f"tyre_hex_{Nr}_{Nt}_{Np}.xdmf")

    mesh_generator_ns = int(
        tyre_hex_mesh(
            A0,
            B0,
            THICKNESS,
            OX,
            OZ,
            nr=Nr,
            ntt=Nt,
            npp=Np,
            filename=mesh_filename,
        )
    )

    with XDMFFile(MPI.COMM_WORLD, mesh_filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    tdim = mesh.topology.dim
    fdim = tdim - 1

    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(tdim) + 2.0 * mu * epsilon(w)

    body_force = Constant(
        mesh,
        np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType),
    )

    a = inner(sigma(u), epsilon(v)) * dx
    L = inner(body_force, v) * dx

    # -------------------------------------------------------------------------
    # Dirichlet boundary: the two cut faces
    # -------------------------------------------------------------------------
    def gamma_u_locator(x):
        X = x[0]
        Y = x[1]
        Z = x[2]

        radius = np.sqrt(Y * Y + Z * Z)
        a_ref = A0 + 0.5 * THICKNESS
        b_ref = B0 + 0.5 * THICKNESS
        theta = np.arctan2((radius - OZ) / b_ref, (X - OX) / a_ref)

        theta1 = -np.pi + THETA_CUT
        theta2 = -THETA_CUT
        tolerance = 0.15

        def angular_distance(alpha, beta):
            return np.abs(np.arctan2(np.sin(alpha - beta), np.cos(alpha - beta)))

        return (angular_distance(theta, theta1) < tolerance) | (
            angular_distance(theta, theta2) < tolerance
        )

    gamma_u = locate_entities_boundary(mesh, fdim, gamma_u_locator)
    gamma_u_set = set(gamma_u.tolist())
    gamma_u_dofs = locate_dofs_topological(V, fdim, gamma_u)

    bc = dirichletbc(
        np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType),
        gamma_u_dofs,
        V,
    )
    bcs = [bc]

    # -------------------------------------------------------------------------
    # Inner pressure boundary
    # -------------------------------------------------------------------------
    def gamma_t_locator(x):
        X = x[0]
        Y = x[1]
        Z = x[2]

        radius = np.sqrt(Y * Y + Z * Z)
        level_set = ((X - OX) / A0) ** 2 + ((radius - OZ) / B0) ** 2 - 1.0
        return np.abs(level_set) < FTOL

    gamma_t = locate_entities_boundary(mesh, fdim, gamma_t_locator)
    gamma_t = np.array(
        [facet for facet in gamma_t if facet not in gamma_u_set],
        dtype=np.int32,
    )
    gamma_t_id = 1

    # -------------------------------------------------------------------------
    # Outer/contact boundary
    # -------------------------------------------------------------------------
    def gamma_c_locator(x):
        X = x[0]
        Y = x[1]
        Z = x[2]

        radius = np.sqrt(Y * Y + Z * Z)
        a_out = A0 + THICKNESS
        b_out = B0 + THICKNESS
        level_set = ((X - OX) / a_out) ** 2 + ((radius - OZ) / b_out) ** 2 - 1.0
        return np.abs(level_set) < FTOL

    gamma_c = locate_entities_boundary(mesh, fdim, gamma_c_locator)
    gamma_c = np.array(
        [facet for facet in gamma_c if facet not in gamma_u_set],
        dtype=np.int32,
    )
    gamma_c_id = 2

    contact_node_dofs = np.asarray(
        locate_dofs_topological(V, fdim, gamma_c),
        dtype=np.int32,
    )

    gamma_u_dofs = np.asarray(gamma_u_dofs, dtype=np.int32)

    free_contact_node_dofs = contact_node_dofs[
        ~np.isin(contact_node_dofs, gamma_u_dofs)
    ]

    if free_contact_node_dofs.size == 0:
        raise RuntimeError("No unconstrained contact nodes were detected on Gamma_c.")

    block_size = V.dofmap.index_map_bs

    # Probe the global x component of each unconstrained contact node.
    contact_probe_dofs = (block_size * free_contact_node_dofs).astype(np.int32)

    print(
        f"Contact nodes: {contact_node_dofs.size}, "
        f"free contact nodes: {free_contact_node_dofs.size}, "
        f"removed constrained nodes: "
        f"{contact_node_dofs.size - free_contact_node_dofs.size}"
    )

    traction = Function(V)
    traction.name = "contact_traction"

    facet_indices = np.hstack([gamma_t, gamma_c]).astype(np.int32)
    facet_values = np.hstack(
        [
            np.full(gamma_t.shape, gamma_t_id, dtype=np.int32),
            np.full(gamma_c.shape, gamma_c_id, dtype=np.int32),
        ]
    )

    order = np.argsort(facet_indices)
    facet_indices = facet_indices[order]
    facet_values = facet_values[order]

    _, first = np.unique(facet_indices, return_index=True)
    facet_indices = facet_indices[first]
    facet_values = facet_values[first]

    facet_tags = meshtags(mesh, fdim, facet_indices, facet_values)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)

    normal = FacetNormal(mesh)
    pressure = Constant(mesh, PETSc.ScalarType(1.5e5))

    L += inner(-pressure * normal, v) * ds(gamma_t_id)
    L += inner(traction, v) * ds(gamma_c_id)

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="assembly_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "none",
        },
    )

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

    for boundary_condition in problem.bcs:
        boundary_condition.set(problem._b.array_w)

    num_dofs = int(V.dofmap.index_map.size_global * block_size)
    num_contact_nodes = int(contact_node_dofs.size)

    metadata = {
        "num_nodes": int(mesh.geometry.x.shape[0]),
        "num_dofs": num_dofs,
        "Ns": num_contact_nodes,
        "num_contact_nodes": num_contact_nodes,
        "mesh_generator_ns": mesh_generator_ns,
    }

    return problem, metadata, contact_probe_dofs


# =============================================================================
# Right-hand sides and solvers
# =============================================================================


def make_timing_right_hand_sides(
    A: PETSc.Mat,
    contact_probe_dofs: np.ndarray,
) -> list[PETSc.Vec]:
    """Create evenly distributed sparse contact right-hand sides."""

    count = min(N_TIMING_SOLVES, len(contact_probe_dofs))
    sample_positions = np.linspace(0, len(contact_probe_dofs) - 1, count)
    selected = contact_probe_dofs[np.rint(sample_positions).astype(int)]

    right_hand_sides: list[PETSc.Vec] = []

    for dof in selected:
        b = A.createVecRight()
        b.set(0.0)
        b.setValue(int(dof), PETSc.ScalarType(RHS_FORCE))
        b.assemblyBegin()
        b.assemblyEnd()
        right_hand_sides.append(b)

    return right_hand_sides


def run_lu(
    A: PETSc.Mat,
    right_hand_sides: list[PETSc.Vec],
) -> dict[str, Any]:
    """Factorize once, then time representative triangular solves."""

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setType("preonly")

    pc = solver.getPC()
    pc.setType("lu")
    pc.setFactorSolverType(LU_FACTOR_SOLVER)

    start = time.perf_counter()
    solver.setUp()
    factorization_time = time.perf_counter() - start

    x = A.createVecRight()
    solve_times: list[float] = []
    reasons: list[int] = []

    for b in right_hand_sides:
        x.set(0.0)

        start = time.perf_counter()
        solver.solve(b, x)
        solve_times.append(time.perf_counter() - start)
        reasons.append(int(solver.getConvergedReason()))

    result = {
        "setup_time": float(factorization_time),
        "solve_time": float(np.mean(solve_times)),
        "solve_time_min": float(np.min(solve_times)),
        "solve_time_max": float(np.max(solve_times)),
        "iterations": 1.0,
        "reason": reasons[-1],
        "converged": all(reason > 0 for reason in reasons),
    }

    x.destroy()
    for b in right_hand_sides:
        b.destroy()
    solver.destroy()

    return result


def run_cg_hypre(
    A: PETSc.Mat,
    right_hand_sides: list[PETSc.Vec],
) -> dict[str, Any]:
    """Build Hypre BoomerAMG once, then time representative CG solves."""

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOptionsPrefix("serial_hypre_")
    solver.setOperators(A)
    solver.setType("cg")
    solver.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    solver.setTolerances(
        rtol=CG_RTOL,
        atol=CG_ATOL,
        max_it=CG_MAX_IT,
    )

    pc = solver.getPC()
    pc.setType("hypre")

    # Hard-coded Hypre type; no command-line PETSc options are needed.
    options = PETSc.Options()
    options["serial_hypre_pc_hypre_type"] = "boomeramg"
    solver.setFromOptions()

    start = time.perf_counter()
    solver.setUp()
    pc_setup_time = time.perf_counter() - start

    x = A.createVecRight()
    solve_times: list[float] = []
    iterations: list[int] = []
    reasons: list[int] = []
    residuals: list[float] = []

    for b in right_hand_sides:
        x.set(0.0)

        start = time.perf_counter()
        solver.solve(b, x)
        solve_times.append(time.perf_counter() - start)

        iterations.append(int(solver.getIterationNumber()))
        reasons.append(int(solver.getConvergedReason()))
        residuals.append(float(solver.getResidualNorm()))

    result = {
        "setup_time": float(pc_setup_time),
        "solve_time": float(np.mean(solve_times)),
        "solve_time_min": float(np.min(solve_times)),
        "solve_time_max": float(np.max(solve_times)),
        "iterations": float(np.mean(iterations)),
        "reason": reasons[-1],
        "residual": residuals[-1],
        "converged": all(reason > 0 for reason in reasons),
    }

    x.destroy()
    for b in right_hand_sides:
        b.destroy()
    solver.destroy()

    return result


# =============================================================================
# One isolated serial worker
# =============================================================================


def solver_worker(
    solver_name: str,
    Nt: int,
    Np: int,
    result_filename: str,
    work_directory: str,
) -> None:
    """Run one solver on one mesh in an independent process."""

    problem = None

    try:
        problem, metadata, contact_probe_dofs = build_problem(
            NR,
            Nt,
            Np,
            work_directory,
        )

        right_hand_sides = make_timing_right_hand_sides(
            problem.A,
            contact_probe_dofs,
        )

        if solver_name == "lu":
            timing = run_lu(problem.A, right_hand_sides)
        elif solver_name == "hypre":
            timing = run_cg_hypre(problem.A, right_hand_sides)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        status = "success" if timing["converged"] else "not_converged"

        result = {
            "status": status,
            "solver": solver_name,
            "Nr": NR,
            "Nt": Nt,
            "Np": Np,
            **metadata,
            **timing,
        }

        result["total_time"] = float(
            result["setup_time"] + result["Ns"] * result["solve_time"]
        )

        Path(result_filename).write_text(json.dumps(result, indent=2))

        # A KSP non-convergence is a valid benchmark result, not a worker
        # failure. In particular, CG+Hypre may reach CG_MAX_IT on one mesh
        # and still be tested on larger meshes. Hard failures (including an
        # operating-system OOM kill) are still handled by the exception/exit
        # code machinery below.

    except Exception as error:
        result_path = Path(result_filename)

        if not result_path.exists():
            failure = {
                "status": "failed",
                "solver": solver_name,
                "Nr": NR,
                "Nt": Nt,
                "Np": Np,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            result_path.write_text(json.dumps(failure, indent=2))

        raise

    finally:
        problem = None
        gc.collect()


def describe_exit_code(exit_code: int | None) -> str:
    if exit_code is None:
        return "worker exit code is unavailable"

    if exit_code < 0:
        signal_number = -exit_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"

        if signal_number == signal.SIGKILL:
            return (
                f"worker was killed by {signal_name}; this is usually the "
                "operating-system out-of-memory limit"
            )

        return f"worker was killed by {signal_name}"

    return f"worker exited with code {exit_code}"


def run_isolated_solver(
    solver_name: str,
    Nt: int,
    Np: int,
) -> dict[str, Any]:
    """Run one solver in a fresh process and return its JSON result."""

    context = mp.get_context("spawn")

    with tempfile.TemporaryDirectory(
        prefix=f"{solver_name}_{Nt}_{Np}_"
    ) as temporary_directory:
        result_filename = str(Path(temporary_directory) / "result.json")

        process = context.Process(
            target=solver_worker,
            args=(
                solver_name,
                Nt,
                Np,
                result_filename,
                temporary_directory,
            ),
        )

        process.start()
        process.join()
        exit_code = process.exitcode
        process.close()

        result_path = Path(result_filename)

        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {
                "status": "failed",
                "solver": solver_name,
                "Nr": NR,
                "Nt": Nt,
                "Np": Np,
                "error": describe_exit_code(exit_code),
            }

        result["exit_code"] = exit_code
        return result


# =============================================================================
# Saving and plotting
# =============================================================================


def merge_case_results(
    Nt: int,
    Np: int,
    lu_result: dict[str, Any] | None,
    hypre_result: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Nr": NR,
        "Nt": Nt,
        "Np": Np,
    }

    # Obtain mesh metadata from whichever solver completed the assembly.
    for result in (lu_result, hypre_result):
        if result is None:
            continue

        for key in (
            "num_nodes",
            "num_dofs",
            "Ns",
            "num_contact_nodes",
            "mesh_generator_ns",
        ):
            if key in result:
                row[key] = result[key]

        if "num_dofs" in row:
            break

    for prefix, result in (("lu", lu_result), ("hypre", hypre_result)):
        if result is None:
            row[f"{prefix}_status"] = "not_run"
            continue

        row[f"{prefix}_status"] = result.get("status", "failed")
        row[f"{prefix}_error"] = result.get("error", "")
        row[f"{prefix}_exit_code"] = result.get("exit_code")

        for key in (
            "setup_time",
            "solve_time",
            "solve_time_min",
            "solve_time_max",
            "total_time",
            "iterations",
            "reason",
            "residual",
            "converged",
        ):
            if key in result:
                row[f"{prefix}_{key}"] = result[key]

    return row


def save_results(results: list[dict[str, Any]]) -> None:
    """Checkpoint JSON and CSV after every mesh."""

    json_filename = Path(f"{OUTPUT_PREFIX}.json")
    csv_filename = Path(f"{OUTPUT_PREFIX}.csv")

    json_filename.write_text(json.dumps(results, indent=2, allow_nan=True))

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in results:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_filename.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def successful_curve(
    results: list[dict[str, Any]],
    prefix: str,
) -> tuple[list[int], list[float]]:
    dofs: list[int] = []
    total_times: list[float] = []

    for row in results:
        if row.get(f"{prefix}_status") != "success":
            continue

        num_dofs = row.get("num_dofs")
        total_time = row.get(f"{prefix}_total_time")

        if num_dofs is None or total_time is None:
            continue
        if not np.isfinite(num_dofs) or not np.isfinite(total_time):
            continue

        dofs.append(int(num_dofs))
        total_times.append(float(total_time))

    return dofs, total_times


def plot_results(results: list[dict[str, Any]]) -> None:
    """Plot total modeled time versus total displacement DOFs."""

    import matplotlib.pyplot as plt

    lu_dofs, lu_times = successful_curve(results, "lu")
    hypre_dofs, hypre_times = successful_curve(results, "hypre")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    if lu_dofs:
        ax.semilogx(
            lu_dofs,
            lu_times,
            marker="o",
            linestyle="-",
            label="LU: factorization + Ns triangular solves",
        )

    if hypre_dofs:
        ax.semilogx(
            hypre_dofs,
            hypre_times,
            marker="s",
            linestyle="-",
            label="CG+Hypre: PC setup + Ns CG solves",
        )

    ax.set_xlabel("Total number of displacement DOFs")
    ax.set_ylabel("Total modeled solve time [s]")
    ax.set_title("LU versus CG+Hypre")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    if lu_dofs or hypre_dofs:
        ax.legend()

    fig.tight_layout()
    figure_filename = f"{OUTPUT_PREFIX}_total_time.png"
    fig.savefig(figure_filename, dpi=200)
    plt.close(fig)


def format_value(row: dict[str, Any], key: str, fmt: str) -> str:
    value = row.get(key)

    if value is None:
        return "-"
    if not isinstance(value, (int, float)):
        return "-"
    if not np.isfinite(value):
        return "-"

    return format(value, fmt)


def print_results_table(results: list[dict[str, Any]]) -> None:
    headers = (
        "Nt",
        "Np",
        "DOFs",
        "Ns",
        "LU status",
        "LU total [s]",
        "Hypre status",
        "Hypre total [s]",
        "Hypre it",
    )
    widths = (8, 8, 14, 12, 14, 16, 16, 18, 12)

    print("\n" + "".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for row in results:
        cells = (
            str(row.get("Nt", "-")),
            str(row.get("Np", "-")),
            str(row.get("num_dofs", "-")),
            str(row.get("Ns", "-")),
            str(row.get("lu_status", "-")),
            format_value(row, "lu_total_time", ".3e"),
            str(row.get("hypre_status", "-")),
            format_value(row, "hypre_total_time", ".3e"),
            format_value(row, "hypre_iterations", ".1f"),
        )

        print("".join(cell.rjust(width) for cell, width in zip(cells, widths)))


# =============================================================================
# Sweep
# =============================================================================


def print_solver_result(label: str, result: dict[str, Any]) -> None:
    status = result.get("status", "failed")

    if status in ("success", "not_converged"):
        qualifier = "" if status == "success" else " [not converged]"
        print(
            f"{label}{qualifier}: DOFs={result['num_dofs']}, "
            f"Ns={result['Ns']}, "
            f"setup={result['setup_time']:.4e} s, "
            f"one solve={result['solve_time']:.4e} s, "
            f"total={result['total_time']:.4e} s"
        )

        if label == "CG+Hypre":
            print(
                f"           average iterations={result['iterations']:.1f}, "
                f"reason={result['reason']}, "
                f"residual={result['residual']:.3e}"
            )
            if status == "not_converged":
                print("           continuing the mesh sweep despite non-convergence")
    else:
        message = result.get("error") or describe_exit_code(result.get("exit_code"))
        print(f"{label}: hard failure at this mesh: {message}")


def main() -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "Run this benchmark with `python cg_hypre_vs_lu_simple_sweep.py`. "
            "Do not use mpiexec."
        )

    # Required by multiprocessing with the spawn method.
    mp.freeze_support()

    results: list[dict[str, Any]] = []
    lu_active = True
    hypre_active = True
    Nt = NT_START

    print("Serial LU versus CG+Hypre memory sweep")
    print(f"Nr={NR}, Nt starts at {NT_START}, Nt step={NT_STEP}")
    print(f"Np=round({NP_OVER_NT} * Nt)")
    print(
        "Each solver runs alone in a fresh process. "
        "The operating system/PETSc determines the memory limit."
    )

    while lu_active or hypre_active:
        Np = int(round(NP_OVER_NT * Nt))
        print(f"\n--- Mesh: Nr={NR}, Nt={Nt}, Np={Np} ---")

        lu_result = None
        hypre_result = None

        if lu_active:
            print("Running LU ...", flush=True)
            lu_result = run_isolated_solver("lu", Nt, Np)
            print_solver_result("LU", lu_result)

            if lu_result.get("status") != "success":
                lu_active = False

        if hypre_active:
            print("Running CG+Hypre ...", flush=True)
            hypre_result = run_isolated_solver("hypre", Nt, Np)
            print_solver_result("CG+Hypre", hypre_result)

            # Reaching CG_MAX_IT or another PETSc non-convergence reason
            # must not stop the mesh sweep. Stop only when the worker itself
            # fails, for example because it is killed by the OS at the memory
            # limit.
            if hypre_result.get("status") == "failed":
                hypre_active = False

        results.append(
            merge_case_results(
                Nt,
                Np,
                lu_result,
                hypre_result,
            )
        )

        save_results(results)
        plot_results(results)

        Nt += NT_STEP

    print_results_table(results)
    save_results(results)
    plot_results(results)

    print(f"\nSaved {OUTPUT_PREFIX}.csv")
    print(f"Saved {OUTPUT_PREFIX}.json")
    print(f"Saved {OUTPUT_PREFIX}_total_time.png")


if __name__ == "__main__":
    main()

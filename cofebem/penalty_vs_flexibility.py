import csv
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

if "--worker" in sys.argv:
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx.fem import (
        Constant,
        Function,
        assemble_scalar,
        assemble_vector,
        dirichletbc,
        form,
        functionspace,
        locate_dofs_topological,
    )
    from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix
    from dolfinx.io import VTKFile
    from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags


MESHES = (12, 20, 28, 36)
REPEATS = 3
E, NU = 1.0e9, 0.3
L, R, DELTA = 1.0, 0.5, 1.0e-3
PENALTY_FACTOR = 100.0
TOL = 1.0e-9
MAX_ITER = 10000
EXPORT_VTK = False
OUT = Path("results/hertz_comparison")

E_STAR = E / (1.0 - NU**2)
A_HERTZ = np.sqrt(R * DELTA)
F_HERTZ = 4.0 / 3.0 * E_STAR * np.sqrt(R) * DELTA**1.5
P0_HERTZ = 3.0 * F_HERTZ / (2.0 * np.pi * A_HERTZ**2)


class HertzProblem:
    def __init__(self, n):
        self.mesh = create_box(
            MPI.COMM_WORLD,
            [[0.0, 0.0, -L], [L, L, 0.0]],
            [n, n, n],
            cell_type=CellType.tetrahedron,
        )
        xyz = self.mesh.geometry.x
        xyz[:, :2] = xyz[:, :2] ** 2 / L
        xyz[:, 2] = -xyz[:, 2] ** 2 / L

        self.V = functionspace(self.mesh, ("Lagrange", 1, (3,)))
        self.u = Function(self.V)
        self.u.name = "u"
        self.du = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        fdim = self.mesh.topology.dim - 1
        bottom = locate_entities_boundary(
            self.mesh, fdim, lambda x: np.isclose(x[2], -L)
        )
        top = locate_entities_boundary(self.mesh, fdim, lambda x: np.isclose(x[2], 0.0))
        bottom_dofs = locate_dofs_topological(self.V, fdim, bottom)
        self.bcs = [
            dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), bottom_dofs, self.V)
        ]
        for axis in (0, 1):
            facets = locate_entities_boundary(
                self.mesh, fdim, lambda x, axis=axis: np.isclose(x[axis], 0.0)
            )
            dofs = locate_dofs_topological(self.V.sub(axis), fdim, facets)
            self.bcs.append(dirichletbc(PETSc.ScalarType(0.0), dofs, self.V.sub(axis)))

        top = np.sort(top)
        tags = meshtags(self.mesh, fdim, top, np.ones(top.size, dtype=np.int32))
        self.ds_c = ufl.Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=tags,
            metadata={"quadrature_degree": 6},
        )(1)
        dx = ufl.Measure("dx", domain=self.mesh)
        lmbda = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        mu = E / (2.0 * (1.0 + NU))
        eps = ufl.sym(ufl.grad(self.du))
        sigma = lmbda * ufl.tr(eps) * ufl.Identity(3) + 2.0 * mu * eps
        self.a = ufl.inner(sigma, ufl.sym(ufl.grad(self.v))) * dx
        x = ufl.SpatialCoordinate(self.mesh)
        self.r2 = x[0] ** 2 + x[1] ** 2
        self.g0 = self.r2 / (2.0 * R) - DELTA

        self.nodes = locate_dofs_topological(self.V, fdim, top)
        self.dofs = 3 * self.nodes + 2
        self.xy = self.V.tabulate_dof_coordinates()[self.nodes, :2]
        self.gap0 = np.sum(self.xy**2, axis=1) / (2.0 * R) - DELTA
        areas = np.zeros_like(self.u.x.array)
        assemble_vector(areas, form(self.v[2] * self.ds_c))
        self.areas = areas[self.dofs]
        self.h_contact = 2.0 * np.sqrt(L * A_HERTZ) / n


def solve_penalty(pb):
    stiffness = PENALTY_FACTOR * E / pb.h_contact
    eps_n = Constant(pb.mesh, PETSc.ScalarType(stiffness))
    pn = eps_n * ufl.max_value(pb.u[2] - pb.g0, 0.0)
    residual = ufl.action(pb.a, pb.u) + pn * pb.v[2] * pb.ds_c
    problem = NonlinearProblem(
        residual,
        pb.u,
        bcs=pb.bcs,
        J=ufl.derivative(residual, pb.u, pb.du),
        petsc_options_prefix="penalty_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_atol": 1.0e-10 * F_HERTZ / 4.0,
            "snes_rtol": TOL,
            "snes_stol": 0.0,
            "snes_max_it": 100,
            "snes_error_if_not_converged": True,
            "ksp_type": "preonly",
            "ksp_error_if_not_converged": True,
            "pc_type": "cholesky",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    problem.A.setOption(PETSc.Mat.Option.SPD, True)
    problem.solve()
    pb.u.x.scatter_forward()
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("Penalty solver did not converge.")
    forces = np.zeros_like(pb.u.x.array)
    assemble_vector(forces, form(pn * pb.v[2] * pb.ds_c))
    return forces[pb.dofs], problem.solver.getIterationNumber()


def gpcg(apply, q, tol=TOL, max_iter=MAX_ITER):
    z = np.zeros_like(q)
    active = q >= 0.0
    iterations = 0
    while iterations < max_iter:
        w = apply(z) + q
        if np.linalg.norm(np.minimum(z, w), np.inf) <= tol:
            return z, iterations
        r = np.where(active, 0.0, -w)
        if np.linalg.norm(r, np.inf) <= tol:
            candidates = np.where(active, w, np.inf)
            active[np.argmin(candidates)] = False
            iterations += 1
            continue
        d = r.copy()
        rr = r @ r
        while iterations < max_iter:
            Md = apply(d)
            curvature = d @ Md
            if not np.isfinite(curvature) or curvature <= 0.0:
                raise RuntimeError("Nonpositive curvature in GPCG.")
            alpha_cg = rr / curvature
            decreasing = d < 0.0
            alpha_max = np.min(-z[decreasing] / d[decreasing], initial=np.inf)
            alpha = min(alpha_cg, alpha_max)
            z += alpha * d
            w += alpha * Md
            iterations += 1
            if alpha_max <= alpha_cg:
                zero_tol = 100.0 * np.finfo(float).eps * max(1.0, np.max(z))
                hit = decreasing & (z <= zero_tol)
                active[hit] = True
                z[hit] = 0.0
                break
            r_new = np.where(active, 0.0, -w)
            if np.linalg.norm(r_new, np.inf) <= tol:
                break
            rr_new = r_new @ r_new
            d = r_new + (rr_new / rr) * d
            d[active] = 0.0
            rr = rr_new
    w = apply(z) + q
    if np.linalg.norm(np.minimum(z, w), np.inf) <= tol:
        return z, iterations
    raise RuntimeError("GPCG did not converge.")


def solve_flexibility(pb):
    K = assemble_matrix(form(pb.a), bcs=pb.bcs)
    K.assemble()
    K.setOption(PETSc.Mat.Option.SPD, True)
    ksp = PETSc.KSP().create(pb.mesh.comm)
    ksp.setType("preonly")
    ksp.getPC().setType("cholesky")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setOperators(K)
    ksp.setErrorIfNotConverged(True)
    ksp.setUp()
    rhs = K.createVecLeft()
    sol = K.createVecRight()
    force_scale = F_HERTZ / 4.0

    def apply(z):
        rhs.set(0.0)
        rhs.array[pb.dofs] = force_scale * z
        ksp.solve(rhs, sol)
        return sol.array_r[pb.dofs].copy() / DELTA

    z, iterations = gpcg(apply, pb.gap0 / DELTA)
    fc = force_scale * z
    rhs.set(0.0)
    rhs.array[pb.dofs] = -fc
    ksp.solve(rhs, sol)
    pb.u.x.array[:] = sol.array_r
    pb.u.x.scatter_forward()
    rhs.destroy()
    sol.destroy()
    ksp.destroy()
    K.destroy()
    return fc, iterations


def run_case(method, n, repeat):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run this benchmark with one MPI process.")
    if np.issubdtype(PETSc.ScalarType, np.complexfloating):
        raise RuntimeError("Real-valued PETSc is required.")

    start = perf_counter()
    pb = HertzProblem(n)
    solve = solve_penalty if method == "penalty" else solve_flexibility
    fc, iterations = solve(pb)
    pressure = fc / pb.areas
    elapsed = perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (2**20 if sys.platform == "darwin" else 1024.0)

    p = Function(pb.V)
    p.name = "pressure"
    p.x.array[pb.dofs] = pressure
    ph = P0_HERTZ * ufl.sqrt(ufl.max_value(1.0 - pb.r2 / A_HERTZ**2, 0.0))
    error_sq = assemble_scalar(form((p[2] - ph) ** 2 * pb.ds_c))
    norm_sq = np.pi * A_HERTZ**2 * P0_HERTZ**2 / 8.0
    force = 4.0 * np.sum(fc)
    gap = pb.gap0 - pb.u.x.array[pb.dofs]
    pressure_error = np.sqrt(max(float(error_sq), 0.0) / norm_sq)
    result = {
        "method": method,
        "n": n,
        "repeat": repeat,
        "ndofs": pb.u.x.array.size,
        "nc": pb.dofs.size,
        "pressure_l2_rel": float(pressure_error),
        "force_rel": float(abs(force - F_HERTZ) / F_HERTZ),
        "force_N": float(force),
        "force_hertz_N": float(F_HERTZ),
        "max_nodal_penetration_m": float(np.max(np.maximum(-gap, 0.0))),
        "total_s": elapsed,
        "peak_rss_mib": peak_mib,
        "iterations": int(iterations),
        "penalty_Pa_per_m": (
            PENALTY_FACTOR * E / pb.h_contact if method == "penalty" else 0.0
        ),
    }

    if EXPORT_VTK and repeat == 1:
        tc = Function(pb.V)
        tc.name = "tc"
        tc.x.array[pb.dofs] = -pressure
        with VTKFile(pb.mesh.comm, str(OUT / f"{method}_{n}.pvd"), "w") as vtk:
            vtk.write_mesh(pb.mesh)
            vtk.write_function([pb.u, tc], 0.0)
    return result


def plot_results(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = (("penalty", "Penalty"), ("flexibility", "Flexibility"))
    metrics = (
        ("pressure_l2_rel", "Relative pressure L2 error", "pressure_error.png"),
        ("force_rel", "Relative total force error", "force_error.png"),
        ("total_s", "Total solve time [s]", "solve_time.png"),
        ("peak_rss_mib", "Peak process RSS [MiB]", "memory.png"),
    )
    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        for method, label in methods:
            x, y = [], []
            for n in MESHES:
                selected = [r for r in rows if r["method"] == method and r["n"] == n]
                x.append(selected[0]["ndofs"])
                y.append(np.median([r[key] for r in selected]))
            ax.plot(x, y, "o-", label=label)
        ax.set_xscale("log")
        if key != "peak_rss_mib":
            ax.set_yscale("log")
        ax.set_xlabel("Displacement DOFs")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)


def benchmark():
    OUT.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    print(
        f"Hertz: a = {A_HERTZ:.6g} m, F = {F_HERTZ:.6g} N, p0 = {P0_HERTZ:.6g} Pa",
        flush=True,
    )
    print("Warming the form compilation cache for both methods...", flush=True)
    for method in ("penalty", "flexibility"):
        subprocess.run(
            [sys.executable, script, "--worker", method, "8", "0"], check=True
        )

    rows = []
    for n in MESHES:
        for repeat in range(1, REPEATS + 1):
            methods = (
                ("penalty", "flexibility") if repeat % 2 else ("flexibility", "penalty")
            )
            for method in methods:
                path = OUT / f"{method}_{n}_{repeat}.json"
                subprocess.run(
                    [
                        sys.executable,
                        script,
                        "--worker",
                        method,
                        str(n),
                        str(repeat),
                        str(path),
                    ],
                    check=True,
                )
                row = json.loads(path.read_text())
                rows.append(row)
                path.unlink()
                print(
                    f"{method:11s} n={n:2d} run={repeat} DOFs={row['ndofs']:7d} "
                    f"pressure={row['pressure_l2_rel']:.3e} force={row['force_rel']:.3e} "
                    f"time={row['total_s']:.3f} s RSS={row['peak_rss_mib']:.1f} MiB",
                    flush=True,
                )
                with (OUT / "results.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    plot_results(rows)
    print(f"Results: {OUT.resolve()}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        result = run_case(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        if len(sys.argv) == 6:
            Path(sys.argv[5]).write_text(json.dumps(result, indent=2))
    else:
        benchmark()

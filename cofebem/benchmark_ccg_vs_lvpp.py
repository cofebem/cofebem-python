import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def residual(x, w):
    return float(np.linalg.norm(np.minimum(x, w), np.inf))


def gpcg(M, q, tol=1e-8, max_iter=10000, z0=None, record=False):
    start = perf_counter()
    z = np.zeros_like(q) if z0 is None else np.maximum(z0, 0.0).copy()
    matvecs = 0
    iterations = 0
    changes = 0
    history = []
    status = "max_iter"

    def apply(v):
        nonlocal matvecs
        matvecs += 1
        return M @ v

    def observe(w):
        value = residual(z, w)
        if record:
            history.append((perf_counter() - start, matvecs, value))
        return value

    w = apply(z) + q
    fresh = True
    active = (z == 0.0) & (w >= 0.0)
    eps = np.finfo(float).eps

    while iterations < max_iter:
        if observe(w) <= tol:
            if fresh:
                status = "converged"
                break
            w = apply(z) + q
            fresh = True
            continue

        r = np.where(active, 0.0, -w)
        if np.linalg.norm(r, np.inf) <= tol:
            candidates = np.where(active & (w < -tol), w, np.inf)
            j = np.argmin(candidates)
            if not np.isfinite(candidates[j]):
                w = apply(z) + q
                fresh = True
                if observe(w) <= tol:
                    status = "converged"
                else:
                    status = "stagnated"
                break
            active[j] = False
            changes += 1
            iterations += 1
            continue

        d = r.copy()
        rr = r @ r
        while iterations < max_iter:
            d[active] = 0.0
            Md = apply(d)
            curvature = d @ Md
            if not np.isfinite(curvature) or curvature <= 0.0:
                status = "nonpositive_curvature"
                break
            alpha_cg = rr / curvature
            decreasing = d < 0.0
            alpha_max = np.min(-z[decreasing] / d[decreasing], initial=np.inf)
            alpha = min(alpha_cg, alpha_max)
            if not np.isfinite(alpha) or alpha < 0.0:
                status = "invalid_step"
                break

            z += alpha * d
            w += alpha * Md
            fresh = False
            iterations += 1
            zero_tol = 1000.0 * eps * max(1.0, np.linalg.norm(z, np.inf))
            if np.any(z < -zero_tol):
                status = "infeasible_step"
                break

            hit_boundary = np.isfinite(alpha_max) and alpha_max <= alpha_cg
            snap = z < 0.0
            if hit_boundary:
                hit = (z <= zero_tol) & decreasing
                active[hit] = True
                snap |= hit
                changes += 1
            changed = np.any(z[snap] != 0.0)
            z[snap] = 0.0
            if changed:
                w = apply(z) + q
                fresh = True
            observe(w)
            if hit_boundary:
                break

            r_new = np.where(active, 0.0, -w)
            if np.linalg.norm(r_new, np.inf) <= tol:
                break
            rr_new = r_new @ r_new
            if not np.isfinite(rr_new) or rr_new <= 0.0:
                status = "numerical_breakdown"
                break
            if changed:
                break
            d = r_new + (rr_new / rr) * d
            rr = rr_new

        if status != "max_iter":
            break

    w = apply(z) + q
    value = observe(w)
    converged = value <= tol and status in ("max_iter", "converged")
    if converged:
        status = "converged"
    return (
        z,
        w,
        {
            "converged": converged,
            "status": status,
            "iterations": iterations,
            "active_set_changes": changes,
            "matvecs": matvecs,
            "residual": value,
            "history": np.asarray(history).reshape(-1, 3),
        },
    )


def lvpp(
    M,
    q,
    z0=None,
    alpha=1.0,
    growth=2.0,
    tol=1e-8,
    max_iter=100,
    max_newton=50,
    cg_tol=1e-11,
    max_cg=None,
    record=False,
):
    start = perf_counter()
    n = q.size
    x = np.ones(n) if z0 is None else np.asarray(z0, dtype=float).copy()
    if np.any(x <= 0.0):
        raise ValueError("LVPP requires a strictly positive initial vector.")
    psi = np.log(x)
    matvecs = 0
    newton_iterations = 0
    cg_iterations = 0
    history = []
    status = "max_iter"

    def apply(v):
        nonlocal matvecs
        matvecs += 1
        return M @ v

    def observe(w):
        value = residual(x, w)
        if record:
            history.append((perf_counter() - start, matvecs, value))
        return value

    def count_cg(_):
        nonlocal cg_iterations
        cg_iterations += 1

    w = apply(x) + q
    value = observe(w)
    outer = 0
    inner_tol = 0.01 * tol

    while outer < max_iter and value > tol:
        outer += 1
        previous = psi.copy()
        F = w.copy()
        for _ in range(max_newton):
            if np.linalg.norm(F, np.inf) <= inner_tol:
                break
            root = np.exp(0.5 * psi)
            H = LinearOperator(
                (n, n),
                matvec=lambda v: v + alpha * root * apply(root * v),
                dtype=float,
            )
            y, flag = cg(
                H,
                -alpha * root * F,
                rtol=cg_tol,
                atol=0.0,
                maxiter=max_cg,
                callback=count_cg,
            )
            if flag != 0 or not np.all(np.isfinite(y)):
                status = "cg_failed"
                break

            direction = -alpha * (F + apply(root * y))
            step = 1.0
            norm_F = np.linalg.norm(F)
            for _ in range(50):
                trial_psi = psi + step * direction
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    trial_x = np.exp(trial_psi)
                if np.all(np.isfinite(trial_x)):
                    trial_w = apply(trial_x) + q
                    trial_F = trial_w + (trial_psi - previous) / alpha
                    if (
                        np.all(np.isfinite(trial_F))
                        and np.linalg.norm(trial_F) <= (1 - 1e-4 * step) * norm_F
                    ):
                        break
                step *= 0.5
            else:
                status = "line_search_failed"
                break

            psi, x, w, F = trial_psi, trial_x, trial_w, trial_F
            newton_iterations += 1
            value = observe(w)
            if value <= tol:
                break
        else:
            if np.linalg.norm(F, np.inf) > inner_tol:
                status = "newton_failed"

        value = residual(x, w)
        if value <= tol or status != "max_iter":
            break
        alpha *= growth

    w = apply(x) + q
    value = observe(w)
    if value <= tol:
        status = "converged"
    return (
        x,
        w,
        {
            "converged": status == "converged",
            "status": status,
            "iterations": outer,
            "newton_iterations": newton_iterations,
            "cg_iterations": cg_iterations,
            "matvecs": matvecs,
            "residual": value,
            "history": np.asarray(history).reshape(-1, 3),
        },
    )


def sample_sc(nx, nz, length, height, E, nu, radius, indentation):
    from mpi4py import MPI
    from petsc4py import PETSc
    from dolfinx.mesh import create_box, locate_entities_boundary, CellType
    from dolfinx.fem import functionspace, form, dirichletbc, locate_dofs_topological
    from dolfinx.fem.petsc import assemble_matrix
    from ufl import TrialFunction, TestFunction, Identity, sym, grad, tr, inner, dx

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Run this dense sampled benchmark with one MPI process.")
    if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
        raise ValueError("This benchmark requires real double-precision PETSc.")
    start = perf_counter()
    mesh = create_box(
        MPI.COMM_WORLD,
        [
            np.array([-length / 2, -length / 2, 0.0]),
            np.array([length / 2, length / 2, height]),
        ],
        [nx, nx, nz],
        cell_type=CellType.tetrahedron,
    )
    V = functionspace(mesh, ("Lagrange", 1, (3,)))
    bottom = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], 0.0))
    top = locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], height))
    fixed = locate_dofs_topological(V, 2, bottom)
    contact = locate_dofs_topological(V, 2, top)
    bc = dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), fixed, V)
    u, v = TrialFunction(V), TestFunction(V)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    strain = sym(grad(u))
    sigma = lmbda * tr(strain) * Identity(3) + 2.0 * mu * strain
    K = assemble_matrix(form(inner(sigma, sym(grad(v))) * dx), bcs=[bc])
    K.assemble()
    K.setOption(PETSc.Mat.Option.SPD, True)
    assembly_s = perf_counter() - start

    factor_start = perf_counter()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(K)
    ksp.setType("preonly")
    ksp.getPC().setType("cholesky")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setErrorIfNotConverged(True)
    ksp.setUp()
    factor_s = perf_counter() - factor_start

    # The top normal is +ez; compressive nodal forces act along -ez.
    scalar_dofs = 3 * contact + 2
    coordinates = V.tabulate_dof_coordinates()[contact]
    q = (coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2) / (2 * radius) - indentation
    nc = contact.size
    Sc = np.empty((nc, nc))
    rhs, displacement = K.createVecLeft(), K.createVecRight()
    sample_start = perf_counter()
    for j, dof in enumerate(scalar_dofs):
        rhs.set(0.0)
        rhs.array_w[dof] = -1.0
        ksp.solve(rhs, displacement)
        Sc[:, j] = -displacement.array_r[scalar_dofs]
    sampling_s = perf_counter() - sample_start

    symmetry = np.linalg.norm(Sc - Sc.T, np.inf) / np.linalg.norm(Sc, np.inf)
    if not np.all(np.isfinite(Sc)) or symmetry > 1e-10 or np.any(np.diag(Sc) <= 0):
        raise RuntimeError(
            f"Invalid sampled compliance: symmetry defect {symmetry:.3e}."
        )
    Sc = 0.5 * (Sc + Sc.T)
    ndofs = K.getSize()[0]
    rhs.destroy()
    displacement.destroy()
    ksp.destroy()
    K.destroy()
    return (
        Sc,
        q,
        {
            "ndofs": ndofs,
            "nc": nc,
            "mesh_assembly_s": assembly_s,
            "factor_s": factor_s,
            "sampling_s": sampling_s,
            "setup_s": perf_counter() - start,
            "symmetry_defect": symmetry,
        },
    )


def solve_case(Sc, q, tol, repeats, alpha, growth, record):
    gap_scale = np.linalg.norm(q, np.inf)
    force_scale = gap_scale / np.linalg.norm(Sc @ np.ones(q.size), np.inf)
    A, b = (force_scale / gap_scale) * Sc, q / gap_scale
    x0 = np.ones(q.size)
    solvers = {
        "GPCG": lambda trace: gpcg(A, b, tol=tol, z0=x0, record=trace),
        "LVPP": lambda trace: lvpp(
            A,
            b,
            tol=tol,
            z0=x0,
            alpha=alpha,
            growth=growth,
            max_cg=max(1000, 10 * q.size),
            record=trace,
        ),
    }
    results = {}
    times = {name: [] for name in solvers}
    passed = {name: True for name in solvers}

    for name, solve in solvers.items():
        solve(False)
        results[name] = {"history": np.empty((0, 3))}

    for repeat in range(repeats):
        order = list(solvers) if repeat % 2 == 0 else list(reversed(solvers))
        for name in order:
            start = perf_counter()
            x, w, info = solvers[name](False)
            elapsed = perf_counter() - start
            checked_residual = residual(x, A @ x + b)
            passed[name] &= info["converged"] and checked_residual <= tol
            times[name].append(elapsed)
            results[name].update(info)
            results[name]["residual"] = checked_residual
            results[name]["force"] = force_scale * x

    for name, solve in solvers.items():
        values = np.asarray(times[name])
        results[name].update(
            {
                "converged": bool(passed[name]),
                "status": results[name]["status"] if passed[name] else "failed_repeat",
                "solve_s": float(np.median(values)),
                "min_s": float(values.min()),
                "max_s": float(values.max()),
            }
        )
        if record:
            _, _, trace = solve(True)
            results[name]["history"] = trace["history"]

    return results


def make_plots(rows, traces, tol, out):
    colors = {"GPCG": "#2166ac", "LVPP": "#d6604d"}
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for name, result in traces.items():
        h = result["history"]
        label = name if h[-1, 2] <= tol else f"{name}: tolerance not reached"
        ax.semilogy(
            h[:, 0],
            np.maximum(h[:, 2], 1e-16),
            color=colors[name],
            label=label,
            linewidth=1.8,
        )
    ax.axhline(tol, color="0.35", linestyle="--", label=f"Tolerance = {tol:g}")
    ax.set(
        xlabel="Elapsed LCP solve time [s]",
        ylabel=r"$\|\min(x,\,Ax+b)\|_\infty$",
        title=f"Algebraic contact accuracy: {max(r['nc'] for r in rows)} contact nodes",
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.savefig(out / "accuracy.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for name, color in colors.items():
        subset = [r for r in rows if r["solver"] == name]
        n = np.array([r["nc"] for r in subset])
        valid = np.array([r["converged"] for r in subset])
        t = np.array([r["solve_s"] for r in subset])
        lo = np.array([r["min_s"] for r in subset])
        hi = np.array([r["max_s"] for r in subset])
        setup = np.array([r["setup_s"] for r in subset])
        for ax, common in zip(axes, (np.zeros_like(setup), setup)):
            displayed = np.where(valid, t + common, np.nan)
            ax.plot(n, displayed, "o-", color=color, label=name)
            ax.fill_between(
                n, lo + common, hi + common, where=valid, color=color, alpha=0.15
            )
            if np.any(~valid):
                ax.scatter(
                    n[~valid],
                    (t + common)[~valid],
                    marker="x",
                    s=80,
                    color=color,
                    label=f"{name}: failed tolerance",
                )
            ax.set(
                xlabel="Number of contact nodes", ylabel="Wall time [s]", yscale="log"
            )
            ax.grid(True, which="both", alpha=0.2)
    axes[0].set_title("LCP solve only")
    axes[1].set_title("Common FEM + sampling setup + LCP solve")
    fig.suptitle("Median solve time; shading shows minimum to maximum")
    for ax in axes:
        ax.legend()
    fig.savefig(out / "performance.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", nargs="+", type=int, default=[12, 20, 28, 36])
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=0.5)
    parser.add_argument("--E", type=float, default=1e7)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--indentation", type=float, default=0.01)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--growth", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("results/gpcg_lvpp"))
    args = parser.parse_args()
    if min(args.nx) < 2 or args.nz < 1 or args.repeats < 1:
        parser.error("Require nx >= 2, nz >= 1 and repeats >= 1.")
    if (
        min(
            args.length,
            args.height,
            args.E,
            args.radius,
            args.indentation,
            args.tol,
            args.alpha,
        )
        <= 0
        or not -1 < args.nu < 0.5
        or args.growth < 1
    ):
        parser.error("Invalid geometry, material or solver parameters.")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = None
    for nx in sorted(set(args.nx)):
        print(f"\nSampling Sc: mesh {nx} x {nx} x {args.nz}", flush=True)
        Sc, q, setup = sample_sc(
            nx,
            args.nz,
            args.length,
            args.height,
            args.E,
            args.nu,
            args.radius,
            args.indentation,
        )
        if q.min() >= 0:
            raise ValueError(
                "No initially penetrating contact node; refine the mesh or increase indentation."
            )
        print(
            f"ndofs={setup['ndofs']}, nc={setup['nc']}, "
            f"factor={setup['factor_s']:.3f}s, sample={setup['sampling_s']:.3f}s",
            flush=True,
        )
        record = nx == max(args.nx)
        results = solve_case(
            Sc, q, args.tol, args.repeats, args.alpha, args.growth, record
        )
        for name, result in results.items():
            row = {
                "nx": nx,
                "nz": args.nz,
                "E": args.E,
                "nu": args.nu,
                "length": args.length,
                "height": args.height,
                "radius": args.radius,
                "indentation": args.indentation,
                "tol": args.tol,
                "alpha": args.alpha,
                "growth": args.growth,
                "repeats": args.repeats,
                **setup,
                "solver": name,
                **{
                    key: result[key]
                    for key in (
                        "converged",
                        "status",
                        "solve_s",
                        "min_s",
                        "max_s",
                        "residual",
                        "iterations",
                        "matvecs",
                    )
                },
            }
            rows.append(row)
            print(
                f"{name:5s}: {result['solve_s']:.4f}s, "
                f"residual={result['residual']:.3e}, "
                f"matvecs={result['matvecs']}, {result['status']}",
                flush=True,
            )
        f0, f1 = results["GPCG"]["force"], results["LVPP"]["force"]
        relative_difference = np.linalg.norm(f0 - f1) / max(
            np.linalg.norm(f0), np.linalg.norm(f1)
        )
        print(
            f"Relative force-vector difference: {relative_difference:.3e}", flush=True
        )
        if record:
            traces = results
            for name, result in results.items():
                np.savetxt(
                    args.out / f"history_{name.lower()}.csv",
                    result["history"],
                    delimiter=",",
                    header="time_s,matvecs,residual",
                    comments="",
                )
        del Sc

    with (args.out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_plots(rows, traces, args.tol, args.out)
    print(f"\nSaved accuracy.png, performance.png and CSV data in {args.out}")


if __name__ == "__main__":
    main()

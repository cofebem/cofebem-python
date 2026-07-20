from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "This benchmark needs psutil for phase-local peak RSS measurements. "
        "Install it with: python -m pip install psutil"
    ) from exc

from dolfinx.mesh import (
    CellType,
    create_box,
    entities_to_geometry,
    locate_entities_boundary,
)
from dolfinx.fem import (
    Constant,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    LinearProblem,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
)
from ufl import Identity, TrialFunction, TestFunction, dx, grad, inner, sym, tr

from cofebem.contact.rigid_indenters import parabolic
from cofebem.lcp.problem import LCP
from cofebem.lcp.solve import solve

MIB = 1024.0**2
RESULT_PREFIX = "RESULT_JSON="


@dataclass
class DualTimes:
    wall_s: float = 0.0
    cpu_s: float = 0.0

    def add(self, wall_s: float, cpu_s: float) -> None:
        self.wall_s += float(wall_s)
        self.cpu_s += float(cpu_s)


class Timer:
    def __enter__(self):
        self._wall0 = time.perf_counter()
        self._cpu0 = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wall_s = time.perf_counter() - self._wall0
        self.cpu_s = time.process_time() - self._cpu0


class PeakRSSSampler:
    """Sample current-process RSS during one benchmark phase."""

    def __init__(self, interval_s: float = 0.005):
        self.interval_s = float(interval_s)
        self.process = psutil.Process(os.getpid())
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _rss(self) -> int:
        return int(self.process.memory_info().rss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.peak_bytes = max(self.peak_bytes, self._rss())

    def __enter__(self):
        gc.collect()
        self.baseline_bytes = self._rss()
        self.peak_bytes = self.baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.peak_bytes = max(self.peak_bytes, self._rss())
        self._stop.set()
        assert self._thread is not None
        self._thread.join()

    @property
    def peak_mib(self) -> float:
        return self.peak_bytes / MIB

    @property
    def incremental_peak_mib(self) -> float:
        return max(0, self.peak_bytes - self.baseline_bytes) / MIB


class ScNormalMatvec:
    """Matrix-free p -> N^T K^{-1} N p with detailed profiling."""

    def __init__(self, A, Ic, nrm, tdim=3, factor_solver: str | None = None):
        self.A = A
        self.comm = A.getComm()
        self.tdim = int(tdim)
        self.Ic = np.asarray(Ic, dtype=np.int64)
        self.nrm = np.asarray(nrm, dtype=float)
        self.nc = int(self.Ic.size)

        if self.nrm.shape != (self.nc, self.tdim):
            raise ValueError(
                f"nrm must have shape ({self.nc}, {self.tdim}), got {self.nrm.shape}"
            )

        lengths = np.linalg.norm(self.nrm, axis=1)
        if np.any(lengths == 0.0):
            raise ValueError("Zero normal detected")
        self.n = self.nrm / lengths[:, None]

        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        if factor_solver:
            pc.setFactorSolverType(factor_solver)
        self.ksp.setFromOptions()

        with Timer() as timer:
            self.ksp.setUp()
        self.factorization = DualTimes(timer.wall_s, timer.cpu_s)

        self.rhs = A.createVecRight()
        self.u = A.createVecRight()
        self.cd = np.stack(
            [self.Ic * self.tdim + c for c in range(self.tdim)], axis=1
        ).astype(np.int32)
        self.cdf = self.cd.reshape(-1).astype(np.int32)

        self.uc = PETSc.Vec().createSeq(self.tdim * self.nc, comm=PETSc.COMM_SELF)
        self.isf = PETSc.IS().createGeneral(self.cdf, comm=self.comm)
        self.ist = PETSc.IS().createStride(
            self.tdim * self.nc, first=0, step=1, comm=PETSc.COMM_SELF
        )
        self.scatter = PETSc.Scatter().create(self.u, self.isf, self.uc, self.ist)

        self.n_matvec = 0
        self.embedding = DualTimes()
        self.triangular_solves = DualTimes()
        self.extraction = DualTimes()

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if p.size != self.nc:
            raise ValueError(f"Expected p of size {self.nc}, got {p.size}")

        with Timer() as timer:
            self.rhs.set(0.0)
            values = np.asarray(
                (p[:, None] * self.n).reshape(-1), dtype=PETSc.ScalarType
            )
            self.rhs.setValues(self.cdf, values, addv=PETSc.InsertMode.INSERT_VALUES)
            self.rhs.assemble()
        self.embedding.add(timer.wall_s, timer.cpu_s)

        with Timer() as timer:
            self.ksp.solve(self.rhs, self.u)
        self.triangular_solves.add(timer.wall_s, timer.cpu_s)

        with Timer() as timer:
            self.uc.set(0.0)
            self.scatter.scatter(
                self.u,
                self.uc,
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )
            uc_array = (
                self.uc.getArray(readonly=True).copy().reshape(self.nc, self.tdim)
            )
            y = np.einsum("ij,ij->i", self.n, uc_array)
        self.extraction.add(timer.wall_s, timer.cpu_s)

        self.n_matvec += 1
        return y


def gpcg_matrix_free(apply_M, q, tol=1e-8, max_iter=1000, z0=None):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = q.size
    eps = np.finfo(np.float64).eps
    z = (
        np.zeros(n, dtype=np.float64)
        if z0 is None
        else np.maximum(np.asarray(z0, dtype=np.float64).reshape(-1), 0.0)
    )

    matvec_count = 0

    def Mv(x):
        nonlocal matvec_count
        y = np.asarray(apply_M(x), dtype=np.float64).reshape(-1)
        if y.size != n:
            raise ValueError(f"apply_M returned size {y.size}, expected {n}")
        matvec_count += 1
        return y

    w = Mv(z) + q
    zero_tol = 100.0 * eps * max(1.0, float(np.linalg.norm(z, ord=np.inf)))
    active = (z <= zero_tol) & (w >= 0.0)
    z[active] = 0.0

    cg_iterations = 0
    active_set_changes = 0
    work_count = 0
    converged = False
    numerical_breakdown = False

    while work_count < max_iter:
        scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
        rel_residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf)) / scale
        if rel_residual <= tol:
            converged = True
            break

        free = ~active
        r = np.zeros_like(z)
        r[free] = -w[free]

        if np.linalg.norm(r, ord=np.inf) <= tol * scale:
            violated = np.flatnonzero(active & (w < -tol * scale))
            if violated.size == 0:
                converged = True
                break
            active[violated[np.argmin(w[violated])]] = False
            active_set_changes += 1
            work_count += 1
            continue

        d = r.copy()
        rr = float(r @ r)

        while work_count < max_iter:
            d[active] = 0.0
            Md = Mv(d)
            curvature = float(d @ Md)
            if not np.isfinite(curvature) or curvature <= 0.0:
                numerical_breakdown = True
                break

            alpha_cg = rr / curvature
            decreasing = d < 0.0
            alpha_max = (
                float(np.min(-z[decreasing] / d[decreasing]))
                if np.any(decreasing)
                else np.inf
            )
            alpha = min(alpha_cg, alpha_max)
            if alpha < 0.0 or not np.isfinite(alpha):
                numerical_breakdown = True
                break

            z = z + alpha * d
            w = w + alpha * Md
            zero_tol = 1000.0 * eps * max(1.0, float(np.linalg.norm(z, ord=np.inf)))
            z[(z < 0.0) & (z >= -zero_tol)] = 0.0
            if np.any(z < -zero_tol):
                numerical_breakdown = True
                break

            cg_iterations += 1
            work_count += 1
            hit_boundary = np.isfinite(alpha_max) and alpha_max <= alpha_cg * (
                1.0 + 1e-14
            )
            if hit_boundary:
                hit = (z <= zero_tol) & (d < 0.0)
                z[hit] = 0.0
                active[hit] = True
                active_set_changes += 1
                break

            free = ~active
            r_new = np.zeros_like(z)
            r_new[free] = -w[free]
            scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)
            if np.linalg.norm(r_new, ord=np.inf) <= tol * scale:
                break

            rr_new = float(r_new @ r_new)
            if rr_new <= 0.0 or not np.isfinite(rr_new):
                numerical_breakdown = True
                break

            beta = rr_new / rr
            d = r_new + beta * d
            d[active] = 0.0
            r = r_new
            rr = rr_new

        if numerical_breakdown:
            break

    residual = float(np.linalg.norm(np.minimum(z, w), ord=np.inf))
    return (
        z,
        w,
        {
            "converged": converged,
            "numerical_breakdown": numerical_breakdown,
            "cg_iterations": cg_iterations,
            "active_set_changes": active_set_changes,
            "work_count": work_count,
            "matvec_count": matvec_count,
            "residual": residual,
        },
    )


def assemble_system(mesh, E=1.0e9, nu=0.3):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def eps(w):
        return sym(grad(w))

    def sig(w):
        return la * tr(eps(w)) * Identity(tdim) + 2.0 * mu * eps(w)

    f = Constant(mesh, np.zeros(tdim, dtype=PETSc.ScalarType))
    a = inner(sig(u), eps(v)) * dx
    Lform = inner(f, v) * dx

    gamma_u = locate_entities_boundary(
        mesh, fdim, lambda x: np.isclose(x[2], 0.0, atol=1.0e-12)
    )
    fixed_dofs = locate_dofs_topological(V, fdim, gamma_u)
    bc = dirichletbc(np.zeros(tdim, dtype=PETSc.ScalarType), fixed_dofs, V)

    pb = LinearProblem(
        a,
        Lform,
        bcs=[bc],
        petsc_options_prefix="benchmark_sc_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # Explicit assembly, matching the original script.
    pb._A.zeroEntries()
    assemble_matrix(pb._A, pb._a, bcs=pb.bcs)
    pb._A.assemble()
    with pb._b.localForm() as b0:
        b0.set(0.0)
    assemble_vector(pb._b, pb._L)
    apply_lifting(pb._b, [pb._a], bcs=[pb.bcs])
    pb._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for boundary_condition in pb.bcs:
        boundary_condition.set(pb._b.array_w)

    return V, pb._A, pb._b, pb


def unique_vertices(mesh, facets):
    fdim = mesh.topology.dim - 1
    facet_geometry = entities_to_geometry(
        mesh, fdim, np.asarray(facets, dtype=np.int32)
    )
    return np.sort(np.unique(np.asarray(facet_geometry, dtype=np.int32).ravel()))


def create_contact_data(mesh, side_fraction: float, L: float = 1.0):
    if not 0.0 < side_fraction <= 1.0:
        raise ValueError("contact side fraction must be in (0, 1]")

    fdim = mesh.topology.dim - 1
    half_side = 0.5 * side_fraction * L
    center = 0.5 * L
    tol = 10.0 * np.finfo(float).eps * max(1.0, L)

    def locator(x):
        return (
            np.isclose(x[2], L, atol=1.0e-12)
            & (np.abs(x[0] - center) <= half_side + tol)
            & (np.abs(x[1] - center) <= half_side + tol)
        )

    facets = locate_entities_boundary(mesh, fdim, locator)
    Ic = unique_vertices(mesh, facets).astype(np.int64)
    if Ic.size == 0:
        raise RuntimeError(
            "The selected contact patch contains no facets. Increase the "
            "contact fraction or refine the mesh."
        )

    coordinates = mesh.geometry.x[Ic].copy()
    normals = np.zeros((Ic.size, mesh.geometry.dim), dtype=np.float64)
    normals[:, 2] = 1.0  # exact for the planar top boundary
    return facets, Ic, coordinates, normals


def create_ksp(A, factor_solver: str | None):
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    if factor_solver:
        pc.setFactorSolverType(factor_solver)
    ksp.setFromOptions()
    return ksp


def construct_dense_sc(A, Ic, nrm, tdim=3, f0=1.0e9, factor_solver=None):
    Ic = np.asarray(Ic, dtype=np.int64)
    nrm = np.asarray(nrm, dtype=np.float64)
    nc = Ic.size
    n = nrm / np.linalg.norm(nrm, axis=1)[:, None]

    ksp = create_ksp(A, factor_solver)
    with Timer() as timer:
        ksp.setUp()
    factorization = DualTimes(timer.wall_s, timer.cpu_s)

    rhs = A.createVecRight()
    u = A.createVecRight()
    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    isf = PETSc.IS().createGeneral(cd.reshape(-1), comm=A.getComm())
    ist = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    scatter = PETSc.Scatter().create(u, isf, uc, ist)

    Sc = np.empty((nc, nc), dtype=np.float64)
    embedding = DualTimes()
    triangular_solves = DualTimes()
    extraction = DualTimes()

    for j in range(nc):
        with Timer() as timer:
            rhs.set(0.0)
            rhs.setValues(cd[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
            rhs.assemble()
        embedding.add(timer.wall_s, timer.cpu_s)

        with Timer() as timer:
            ksp.solve(rhs, u)
        triangular_solves.add(timer.wall_s, timer.cpu_s)

        with Timer() as timer:
            uc.set(0.0)
            scatter.scatter(
                u,
                uc,
                addv=PETSc.InsertMode.INSERT_VALUES,
                mode=PETSc.ScatterMode.FORWARD,
            )
            ux = uc.getArray(readonly=True).copy().reshape(nc, tdim)
            Sc[:, j] = np.einsum("ij,ij->i", n, ux) / f0
        extraction.add(timer.wall_s, timer.cpu_s)

    profile = {
        "factorization": factorization,
        "embedding": embedding,
        "triangular_solves": triangular_solves,
        "extraction": extraction,
        "n_triangular_solves": int(nc),
    }
    return Sc, profile


def build_gaps(
    coordinates,
    n_loads: int,
    side_fraction: float,
    load_path: str = "moving",
    L: float = 1.0,
):
    Rindenter = 0.3
    displacement = 0.2
    patch_side = side_fraction * L
    if load_path == "repeated" or n_loads == 1:
        x_positions = np.full(n_loads, 0.5 * L, dtype=float)
    elif load_path == "moving":
        # Keep the moving center inside the candidate contact patch.
        x_positions = np.linspace(
            0.5 * L - 0.25 * patch_side,
            0.5 * L + 0.25 * patch_side,
            n_loads,
        )
    else:
        raise ValueError(f"Unknown load path: {load_path}")

    gaps = []
    for x_center in x_positions:
        gap = (
            parabolic(
                coordinates[:, 0],
                coordinates[:, 1],
                x_center,
                0.5 * L,
                Rindenter,
                np.ones_like(coordinates[:, 2]) - displacement,
            )
            - coordinates[:, 2]
        )
        gaps.append(np.asarray(gap, dtype=np.float64).reshape(-1))
    return x_positions, gaps


def matrix_info(A, V, mesh):
    index_map = V.dofmap.index_map
    ndofs = int(index_map.size_global * V.dofmap.index_map_bs)
    ncells = int(mesh.topology.index_map(mesh.topology.dim).size_global)
    info = A.getInfo()
    nnz = int(info.get("nz_used", info.get("nz_allocated", 0)))
    return ndofs, ncells, nnz


def run_worker(args):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "The included RSS measurement is for serial workers. Run this "
            "sweep without mpirun. For MPI jobs use scheduler MaxRSS per step."
        )

    with Timer() as common_timer:
        mesh = create_box(
            MPI.COMM_WORLD,
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [args.nx, args.ny, args.nz],
            CellType.hexahedron,
        )
        V, A, b, pb = assemble_system(mesh)
        _, Ic, contact_x, nrm = create_contact_data(mesh, args.contact_fraction)
        _, gaps = build_gaps(
            contact_x,
            args.n_loads,
            args.contact_fraction,
            load_path=args.load_path,
        )
    ndofs, ncells, nnz = matrix_info(A, V, mesh)
    nc = int(Ic.size)

    base = {
        "method": args.method,
        "repeat": args.repeat,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "n_cells": ncells,
        "n_dofs": ndofs,
        "A_nnz": nnz,
        "contact_side_fraction": args.contact_fraction,
        "contact_area_fraction": args.contact_fraction**2,
        "n_contact_dofs": nc,
        "n_loads": args.n_loads,
        "load_path": args.load_path,
        "warm_start": args.warm_start,
        "common_setup_wall_s": common_timer.wall_s,
        "common_setup_cpu_s": common_timer.cpu_s,
        "dense_Sc_theoretical_mib": nc * nc * np.dtype(np.float64).itemsize / MIB,
    }

    with PeakRSSSampler(args.rss_interval) as rss:
        with Timer() as total_timer:
            if args.method == "flexibility":
                with Timer() as setup_timer:
                    Sc, profile = construct_dense_sc(
                        A,
                        Ic,
                        nrm,
                        tdim=mesh.topology.dim,
                        factor_solver=args.factor_solver,
                    )

                z = np.zeros(nc, dtype=np.float64)
                converged = []
                residuals = []
                iterations = []
                with Timer() as solve_timer:
                    for gap in gaps:
                        result = solve(
                            LCP(Sc, gap),
                            "ccg_v2",
                            z0=z if args.warm_start else None,
                            tol=args.tol,
                            max_iter=args.max_iter,
                        )
                        z = result.z
                        converged.append(bool(result.converged))
                        residuals.append(float(getattr(result, "residual", np.nan)))
                        iterations.append(float(getattr(result, "iterations", np.nan)))

                matvec_count = 0
                cg_iterations = np.nansum(iterations)
                active_set_changes = np.nan

            elif args.method == "matrix_free":
                with Timer() as setup_timer:
                    apply_Sc = ScNormalMatvec(
                        A,
                        Ic,
                        nrm,
                        tdim=mesh.topology.dim,
                        factor_solver=args.factor_solver,
                    )

                z = np.zeros(nc, dtype=np.float64)
                converged = []
                residuals = []
                cg_iterations = 0
                active_set_changes = 0
                with Timer() as solve_timer:
                    for gap in gaps:
                        z, _, info = gpcg_matrix_free(
                            apply_Sc,
                            gap,
                            tol=args.tol,
                            max_iter=args.max_iter,
                            z0=z if args.warm_start else None,
                        )
                        converged.append(bool(info["converged"]))
                        residuals.append(float(info["residual"]))
                        cg_iterations += int(info["cg_iterations"])
                        active_set_changes += int(info["active_set_changes"])

                profile = {
                    "factorization": apply_Sc.factorization,
                    "embedding": apply_Sc.embedding,
                    "triangular_solves": apply_Sc.triangular_solves,
                    "extraction": apply_Sc.extraction,
                    "n_triangular_solves": int(apply_Sc.n_matvec),
                }
                matvec_count = int(apply_Sc.n_matvec)
            else:  # pragma: no cover
                raise ValueError(args.method)

    tri = profile["triangular_solves"]
    embed = profile["embedding"]
    extract = profile["extraction"]
    fact = profile["factorization"]
    ntri = int(profile["n_triangular_solves"])

    base.update(
        {
            "method_setup_wall_s": setup_timer.wall_s,
            "method_setup_cpu_s": setup_timer.cpu_s,
            "lcp_solves_wall_s": solve_timer.wall_s,
            "lcp_solves_cpu_s": solve_timer.cpu_s,
            "mean_lcp_wall_per_load_s": solve_timer.wall_s / args.n_loads,
            "mean_lcp_cpu_per_load_s": solve_timer.cpu_s / args.n_loads,
            "method_total_wall_s": total_timer.wall_s,
            "method_total_cpu_s": total_timer.cpu_s,
            "cold_total_wall_s": common_timer.wall_s + total_timer.wall_s,
            "cold_total_cpu_s": common_timer.cpu_s + total_timer.cpu_s,
            "factorization_wall_s": fact.wall_s,
            "factorization_cpu_s": fact.cpu_s,
            "rhs_embedding_wall_s": embed.wall_s,
            "rhs_embedding_cpu_s": embed.cpu_s,
            "triangular_solve_wall_s": tri.wall_s,
            "triangular_solve_cpu_s": tri.cpu_s,
            "solution_extraction_wall_s": extract.wall_s,
            "solution_extraction_cpu_s": extract.cpu_s,
            "n_triangular_solves": ntri,
            "avg_triangular_solve_wall_s": tri.wall_s / max(1, ntri),
            "avg_triangular_solve_cpu_s": tri.cpu_s / max(1, ntri),
            "matvec_count": matvec_count,
            "cg_iterations": float(cg_iterations),
            "active_set_changes": float(active_set_changes),
            "all_converged": all(converged),
            "max_complementarity_residual": float(
                np.nanmax(residuals) if residuals else np.nan
            ),
            "final_force_l2": float(np.linalg.norm(z)),
            "final_force_sum": float(np.sum(z)),
            "final_force_checksum": float(
                z @ np.sin(np.arange(1, z.size + 1, dtype=np.float64))
            ),
            "rss_baseline_mib": rss.baseline_bytes / MIB,
            "rss_peak_mib": rss.peak_mib,
            "rss_incremental_peak_mib": rss.incremental_peak_mib,
            # Assuming an existing reusable LU, remove only factorization time.
            "warm_lu_total_wall_s": total_timer.wall_s - fact.wall_s,
            "warm_lu_total_cpu_s": total_timer.cpu_s - fact.cpu_s,
        }
    )

    print(RESULT_PREFIX + json.dumps(base, sort_keys=True))


def parse_mesh_case(value: str):
    try:
        nx, ny, nz = (int(x) for x in value.lower().split("x"))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid mesh case {value!r}; expected NxXNyXNz, e.g. 50x50x5"
        ) from exc
    if min(nx, ny, nz) <= 0:
        raise argparse.ArgumentTypeError("Mesh dimensions must be positive")
    return nx, ny, nz


def median_or_blank(values):
    clean = [float(v) for v in values if v not in ("", None) and np.isfinite(float(v))]
    return statistics.median(clean) if clean else ""


def make_summary(rows):
    keys = (
        "method",
        "nx",
        "ny",
        "nz",
        "contact_side_fraction",
        "n_loads",
        "load_path",
        "warm_start",
    )
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(row)

    fixed_columns = list(keys) + [
        "n_cells",
        "n_dofs",
        "A_nnz",
        "n_contact_dofs",
        "dense_Sc_theoretical_mib",
        "all_converged",
    ]
    metric_columns = [
        "common_setup_wall_s",
        "common_setup_cpu_s",
        "method_setup_wall_s",
        "method_setup_cpu_s",
        "lcp_solves_wall_s",
        "lcp_solves_cpu_s",
        "mean_lcp_wall_per_load_s",
        "mean_lcp_cpu_per_load_s",
        "method_total_wall_s",
        "method_total_cpu_s",
        "cold_total_wall_s",
        "cold_total_cpu_s",
        "warm_lu_total_wall_s",
        "warm_lu_total_cpu_s",
        "factorization_wall_s",
        "triangular_solve_wall_s",
        "avg_triangular_solve_wall_s",
        "rss_peak_mib",
        "rss_incremental_peak_mib",
        "n_triangular_solves",
        "matvec_count",
        "cg_iterations",
        "active_set_changes",
        "max_complementarity_residual",
        "final_force_l2",
        "final_force_sum",
        "final_force_checksum",
    ]

    summary = []
    for key, group in grouped.items():
        item = {name: value for name, value in zip(keys, key, strict=True)}
        for name in fixed_columns[len(keys) :]:
            if name == "all_converged":
                item[name] = all(bool(row[name]) for row in group)
            else:
                item[name] = group[0][name]
        item["repeats"] = len(group)
        for name in metric_columns:
            item[f"median_{name}"] = median_or_blank(row[name] for row in group)
        summary.append(item)
    return summary


def make_method_comparison(summary_rows):
    """Pair flexibility and matrix-free medians for each physical case."""
    case_keys = (
        "nx",
        "ny",
        "nz",
        "contact_side_fraction",
        "n_loads",
        "load_path",
        "warm_start",
    )
    paired = {}
    for row in summary_rows:
        key = tuple(row[name] for name in case_keys)
        paired.setdefault(key, {})[row["method"]] = row

    comparisons = []
    for key, methods in paired.items():
        if "flexibility" not in methods or "matrix_free" not in methods:
            continue
        flex = methods["flexibility"]
        matfree = methods["matrix_free"]

        flex_cold = float(flex["median_method_total_wall_s"])
        mf_cold = float(matfree["median_method_total_wall_s"])
        flex_warm = float(flex["median_warm_lu_total_wall_s"])
        mf_warm = float(matfree["median_warm_lu_total_wall_s"])
        flex_ram = float(flex["median_rss_incremental_peak_mib"])
        mf_ram = float(matfree["median_rss_incremental_peak_mib"])

        force_scale = max(
            1.0,
            abs(float(flex["median_final_force_l2"])),
            abs(float(matfree["median_final_force_l2"])),
        )
        force_norm_rel_diff = (
            abs(
                float(flex["median_final_force_l2"])
                - float(matfree["median_final_force_l2"])
            )
            / force_scale
        )
        checksum_scale = max(
            1.0,
            abs(float(flex["median_final_force_checksum"])),
            abs(float(matfree["median_final_force_checksum"])),
        )
        checksum_rel_diff = (
            abs(
                float(flex["median_final_force_checksum"])
                - float(matfree["median_final_force_checksum"])
            )
            / checksum_scale
        )

        item = {name: value for name, value in zip(case_keys, key, strict=True)}
        item.update(
            {
                "n_dofs": flex["n_dofs"],
                "n_contact_dofs": flex["n_contact_dofs"],
                "dense_Sc_theoretical_mib": flex["dense_Sc_theoretical_mib"],
                "flex_method_total_wall_s": flex_cold,
                "matrix_free_method_total_wall_s": mf_cold,
                "cold_faster_method": (
                    "flexibility" if flex_cold <= mf_cold else "matrix_free"
                ),
                "cold_speedup": max(flex_cold, mf_cold)
                / max(min(flex_cold, mf_cold), np.finfo(float).tiny),
                "flex_warm_lu_total_wall_s": flex_warm,
                "matrix_free_warm_lu_total_wall_s": mf_warm,
                "warm_lu_faster_method": (
                    "flexibility" if flex_warm <= mf_warm else "matrix_free"
                ),
                "warm_lu_speedup": max(flex_warm, mf_warm)
                / max(min(flex_warm, mf_warm), np.finfo(float).tiny),
                "flex_incremental_peak_mib": flex_ram,
                "matrix_free_incremental_peak_mib": mf_ram,
                "ram_ratio_larger_over_smaller": max(flex_ram, mf_ram)
                / max(min(flex_ram, mf_ram), np.finfo(float).tiny),
                "matrix_free_matvec_count": matfree["median_matvec_count"],
                "force_l2_norm_relative_difference": force_norm_rel_diff,
                "force_checksum_relative_difference": checksum_rel_diff,
                "both_converged": bool(flex["all_converged"])
                and bool(matfree["all_converged"]),
            }
        )
        comparisons.append(item)
    return comparisons


def make_break_even_table(summary_rows):
    """Estimate affine time models from the 1- and 10-solve measurements.

    T_method(m) ~= setup_intercept + m * marginal_solve_cost.  The estimate is
    cleanest with --load-path repeated --no-warm-start.
    """
    physical_keys = (
        "nx",
        "ny",
        "nz",
        "contact_side_fraction",
        "load_path",
        "warm_start",
    )
    indexed = {}
    for row in summary_rows:
        key = tuple(row[name] for name in physical_keys)
        indexed.setdefault(key, {}).setdefault(row["method"], {})[
            int(row["n_loads"])
        ] = row

    output = []
    for key, methods in indexed.items():
        if not all(method in methods for method in ("flexibility", "matrix_free")):
            continue
        if not all(
            1 in methods[method] and 10 in methods[method] for method in methods
        ):
            continue

        models = {}
        for method in ("flexibility", "matrix_free"):
            t1 = float(methods[method][1]["median_method_total_wall_s"])
            t10 = float(methods[method][10]["median_method_total_wall_s"])
            marginal = (t10 - t1) / 9.0
            intercept = t1 - marginal
            models[method] = (intercept, marginal, t1, t10)

        sf, cf = models["flexibility"][0], models["flexibility"][1]
        sm, cm = models["matrix_free"][0], models["matrix_free"][1]
        denominator = cm - cf
        if abs(denominator) <= np.finfo(float).eps * max(1.0, abs(cm), abs(cf)):
            break_even = np.nan
        else:
            break_even = (sf - sm) / denominator
            if break_even < 0.0:
                break_even = np.nan

        flex_row = methods["flexibility"][1]
        item = {name: value for name, value in zip(physical_keys, key, strict=True)}
        item.update(
            {
                "n_dofs": flex_row["n_dofs"],
                "n_contact_dofs": flex_row["n_contact_dofs"],
                "flex_setup_intercept_wall_s": sf,
                "flex_marginal_solve_wall_s": cf,
                "matrix_free_setup_intercept_wall_s": sm,
                "matrix_free_marginal_solve_wall_s": cm,
                "estimated_break_even_n_solves": break_even,
                "interpretation_is_clean": (
                    key[physical_keys.index("load_path")] == "repeated"
                    and not bool(key[physical_keys.index("warm_start")])
                ),
            }
        )
        output.append(item)
    return output


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_parent(args):
    rows = []
    script = Path(__file__).resolve()
    env = os.environ.copy()
    # Reproducible single-process CPU measurements unless the user overrides.
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["NUMEXPR_NUM_THREADS"] = str(args.threads)

    total = (
        len(args.mesh_cases)
        * len(args.contact_fractions)
        * len(args.solve_counts)
        * len(args.methods)
        * args.repeats
    )
    current = 0

    for nx, ny, nz in args.mesh_cases:
        for fraction in args.contact_fractions:
            for n_loads in args.solve_counts:
                for method in args.methods:
                    for repeat in range(args.repeats):
                        current += 1
                        print(
                            f"[{current}/{total}] {method:12s} mesh={nx}x{ny}x{nz} "
                            f"contact={fraction:.3f} loads={n_loads} repeat={repeat + 1}",
                            flush=True,
                        )
                        command = [
                            sys.executable,
                            str(script),
                            "--worker",
                            "--method",
                            method,
                            "--nx",
                            str(nx),
                            "--ny",
                            str(ny),
                            "--nz",
                            str(nz),
                            "--contact-fraction",
                            str(fraction),
                            "--n-loads",
                            str(n_loads),
                            "--load-path",
                            args.load_path,
                            "--repeat",
                            str(repeat),
                            "--tol",
                            str(args.tol),
                            "--max-iter",
                            str(args.max_iter),
                            "--rss-interval",
                            str(args.rss_interval),
                        ]
                        command.append(
                            "--warm-start" if args.warm_start else "--no-warm-start"
                        )
                        if args.factor_solver:
                            command.extend(["--factor-solver", args.factor_solver])

                        completed = subprocess.run(
                            command,
                            env=env,
                            text=True,
                            capture_output=True,
                        )
                        if completed.returncode != 0:
                            print(completed.stdout)
                            print(completed.stderr, file=sys.stderr)
                            raise RuntimeError(
                                f"Worker failed for {method}, mesh {nx}x{ny}x{nz}, "
                                f"contact fraction {fraction}, loads {n_loads}"
                            )

                        result_lines = [
                            line
                            for line in completed.stdout.splitlines()
                            if line.startswith(RESULT_PREFIX)
                        ]
                        if not result_lines:
                            print(completed.stdout)
                            raise RuntimeError("Worker did not produce RESULT_JSON")
                        row = json.loads(result_lines[-1][len(RESULT_PREFIX) :])
                        rows.append(row)
                        write_csv(Path(args.output), rows)

    summary = make_summary(rows)
    summary_path = Path(args.output).with_name(Path(args.output).stem + "_summary.csv")
    write_csv(summary_path, summary)

    comparisons = make_method_comparison(summary)
    comparison_path = Path(args.output).with_name(
        Path(args.output).stem + "_comparison.csv"
    )
    write_csv(comparison_path, comparisons)

    break_even_rows = make_break_even_table(summary)
    break_even_path = Path(args.output).with_name(
        Path(args.output).stem + "_break_even.csv"
    )
    write_csv(break_even_path, break_even_rows)

    print(f"\nRaw runs:   {Path(args.output).resolve()}")
    print(f"Medians:    {summary_path.resolve()}")
    print(f"Comparison: {comparison_path.resolve()}")
    print(f"Break-even: {break_even_path.resolve()}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    # Parent sweep options.
    parser.add_argument(
        "--mesh-cases",
        nargs="+",
        type=parse_mesh_case,
        default=[(20, 20, 5), (35, 35, 5), (50, 50, 5)],
    )
    parser.add_argument(
        "--contact-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0],
        help="Side length of square contact patch divided by L; area fraction is its square.",
    )
    parser.add_argument("--solve-counts", nargs="+", type=int, default=[1, 10])
    parser.add_argument(
        "--load-path",
        choices=["moving", "repeated"],
        default="moving",
        help=(
            "moving reproduces the original moving indenter; repeated uses the "
            "same centered gap for every solve and is cleaner for break-even fits."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["flexibility", "matrix_free"],
        default=["flexibility", "matrix_free"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default="sc_benchmark.csv")
    parser.add_argument("--threads", type=int, default=1)

    # Shared numerical options.
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--warm-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--factor-solver", default=None)
    parser.add_argument("--rss-interval", type=float, default=0.005)

    # Worker-only options.
    parser.add_argument("--method", choices=["flexibility", "matrix_free"])
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--contact-fraction", type=float)
    parser.add_argument("--n-loads", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.worker:
        required = (
            args.method,
            args.nx,
            args.ny,
            args.nz,
            args.contact_fraction,
            args.n_loads,
        )
        if any(value is None for value in required):
            raise ValueError("Missing worker arguments")
        run_worker(args)
    else:
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        if any(n not in (1, 10) for n in args.solve_counts):
            print(
                "Note: solve counts other than 1 and 10 are supported and will also be run."
            )
        run_parent(args)


if __name__ == "__main__":
    main()

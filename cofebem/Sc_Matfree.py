"""Matrix-free normal-contact benchmark: Sc_nn compliance + GPCG on a 3-D cube.

The normal contact compliance S = N^T K^{-1} N is applied through solves with a
factorized stiffness K (PETSc KSP preonly + LU/Cholesky). This script compares

  * compression=none : plain MUMPS multifrontal factorization, BLR explicitly
                       disabled (ICNTL(35)=0, CNTL(7)=0);
  * compression=blr  : MUMPS Block Low-Rank factorization with ICNTL(35)=2 and
                       CNTL(7)=--blr-tol.

BLR mode (verified against the MUMPS 5.9.1 user guide, Sec. 5.20, and the
installed MUMPS 5.8.2 through PETSc 3.25):
  ICNTL(35)=2  BLR during factorization AND solve: the factors are stored in
               low-rank form and the compressed factors are used by every
               triangular solve. (ICNTL(35)=3 would decompress for the solve and
               give no memory gain; =1 is "automatic", currently equal to 2. The
               explicit value 2 is used so the mode cannot change silently.)
  CNTL(7)      Dropping parameter eps. It is an ABSOLUTE threshold of the
               truncated rank-revealing QR, applied to the scaled matrix, so the
               scaling ICNTL(8) is fixed explicitly (default 7, iterative row/col
               scaling) for every configuration, compressed or not.
  ICNTL(36)    BLR variant (0=UFSC, the 5.8.2 default; 1=UCFS, the 5.9 default).
  ICNTL(37)    Contribution-block compression (default 0).

Common to every configuration (written explicitly, then read back from the
MUMPS instance after factorization; a mismatch aborts the run instead of
silently running something else): ordering ICNTL(7)/(28)/(29), scaling
ICNTL(8), iterative refinement ICNTL(10)=0, error analysis ICNTL(11)=0, the
factorization type (--pc) and the thread environment. ICNTL(10)=0 is required
for a fair comparison: MUMPS disables refinement whenever the solution is kept
distributed (PETSc does this on >1 rank), so refinement could not be common to
all rank counts. Pivoting thresholds CNTL(1) are left at the MUMPS defaults for
the chosen factorization type and are recorded.

GPCG needs a symmetric positive definite compliance. With --pc cholesky (LDL^T)
the BLR approximation K~ = L~ D L~^T is symmetric by construction; with LU the
BLR factors are not exactly transposes and S~ is only symmetric to O(eps). Both
are allowed; every run records a measured symmetry defect and Rayleigh quotient
of the operator actually used.

Statistics: INFOG(9) (factor storage actually used, includes low-rank factors
with ICNTL(35)=2), INFOG(29) (full-rank factor entries), INFOG(35) (BLR entries),
RINFOG(3)/RINFOG(14) (full-rank / effective flops). With MUMPS 5.8.x INFOG(35)
is known to be inaccurate and INFOG(21-22) overestimated under BLR (fixed in
5.9.0), and RINFOG(14) reads 0 through the PETSc API in this build, so the
statistics block printed by MUMPS (ICNTL(4)=2, factorization only) is also
parsed from the run log by the sweep. Process peak RSS is reported separately.

Modes: run (one MPI job), sweep (fresh MPI process per configuration and
repetition, validation, summaries, plots), plot (offline regeneration).
"""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import random
import re
import resource
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

# Sequential (ICNTL(28)=1, method in ICNTL(7)) or parallel (ICNTL(28)=2, method
# in ICNTL(29)) analysis.
ORDERINGS = {
    "amd": (1, 0),
    "amf": (1, 2),
    "scotch": (1, 3),
    "pord": (1, 4),
    "metis": (1, 5),
    "qamd": (1, 6),
    "auto": (1, 7),
    "ptscotch": (2, 1),
    "parmetis": (2, 2),
}
# ICNTL entries readable through MatMumpsGetIcntl in PETSc 3.25.
MUMPS_ICNTL_READABLE = tuple(range(1, 39)) + (48, 49, 56, 58)
FACTOR_EVENTS = {
    "lu": ("MatLUFactorSym", "MatLUFactorNum"),
    "cholesky": ("MatCholFctrSym", "MatCholFctrNum"),
}


def mumps_settings(args, compression=None, print_level=None):
    """Explicit ICNTL/CNTL values; only ICNTL(35)/CNTL(7) differ for BLR."""
    compression = args.compression if compression is None else compression
    analysis, method = ORDERINGS[args.ordering]
    icntl = {
        4: args.mumps_print_level if print_level is None else print_level,
        28: analysis,
        7 if analysis == 1 else 29: method,
        8: args.scaling,
        10: 0,
        11: 0,
        35: 2 if compression == "blr" else 0,
        36: args.blr_variant,
        37: int(args.blr_cb_compression),
    }
    for item in args.mumps_icntl:
        key, value = item.split("=")
        if int(key) in (35, 4):
            raise ValueError("ICNTL(35) and ICNTL(4) are controlled by the script.")
        icntl[int(key)] = int(value)
    cntl = {7: float(args.blr_tol) if compression == "blr" else 0.0}
    return {"icntl": icntl, "cntl": cntl}


def mumps_version():
    """Best-effort MUMPS version from the installed C header."""
    for root in (sys.prefix, os.environ.get("PETSC_DIR", "")):
        header = Path(root) / "include" / "dmumps_c.h"
        try:
            match = re.search(r'#define MUMPS_VERSION "([^"]+)"', header.read_text())
        except OSError:
            continue
        if match:
            return match.group(1)
    return None


def load_fem():
    global MPI, PETSc, create_box, locate_entities_boundary, meshtags, CellType
    global VTKFile, Constant, Function, functionspace, form, dirichletbc
    global locate_dofs_topological, LinearProblem, assemble_matrix
    global assemble_vector, apply_lifting, Identity, TrialFunction, TestFunction
    global sym, grad, inner, tr, dx, FacetNormal, Measure
    from mpi4py import MPI
    from petsc4py import PETSc

    from dolfinx.mesh import create_box, locate_entities_boundary, meshtags, CellType
    from dolfinx.io import VTKFile
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
        FacetNormal,
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


def boundary_normals(problem, mt, tag):
    V = problem.u.function_space
    mesh = V.mesh
    fdim, gdim = mesh.topology.dim - 1, mesh.geometry.dim

    facets = mt.find(tag)
    dofs = locate_dofs_topological(V, fdim, facets, remote=True)

    ds = Measure("ds", domain=mesh, subdomain_data=mt)
    v = TestFunction(V)

    b = assemble_vector(form(inner(FacetNormal(mesh), v) * ds(tag)))
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    b.ghostUpdate(
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )

    with b.localForm() as local:
        normals = local.array_r.reshape(-1, gdim)[dofs]
    b.destroy()

    norms = np.linalg.norm(normals, axis=1)
    if mesh.comm.allreduce(int(np.any(norms == 0)), op=MPI.MAX):
        raise ValueError(f"Boundary tag {tag} has an undefined nodal normal.")

    normals /= norms[:, None]
    return dofs, normals


def timed(comm, operation, local=None, key=None):
    """Barrier before the phase, then maximum elapsed time across ranks."""
    comm.Barrier()
    start = MPI.Wtime()
    result = operation()
    elapsed = MPI.Wtime() - start
    if local is not None:
        local[key] = elapsed
    return result, comm.allreduce(elapsed, op=MPI.MAX)


class Sc_nn:
    """Normal compliance N^T K^{-1} N through a factorized K.

    mumps: {"icntl": {i: v}, "cntl": {i: v}} applied to the factor matrix via the
    KSP options prefix before analysis, then verified against the MUMPS instance.
    ksp/contact: reuse an existing operator's contact layout with another KSP
    (validation reference); K is then factorized again but not reassembled.
    """

    def __init__(self, problem, mt, Gamma_c_id=2, mumps=None, ksp=None, contact=None):
        self.problem = problem
        self.K = problem.A
        self.ksp = problem.solver if ksp is None else ksp
        self.mumps = mumps
        self.mumps_report = None
        self.dtype = np.dtype(PETSc.ScalarType)
        V = problem.u.function_space
        self.comm = V.mesh.comm
        self.gdim = V.mesh.geometry.dim
        self.times = {}
        self.local_times = {}
        self.profile = None

        if contact is None:
            (dofs_c, normals), self.times["normals_s"] = timed(
                self.comm,
                lambda: boundary_normals(problem, mt, Gamma_c_id),
                self.local_times,
                "normals_s",
            )
            owned = dofs_c < V.dofmap.index_map.size_local
            self.dofs_c = dofs_c[owned]
            self.normals = np.ascontiguousarray(normals[owned], dtype=self.dtype)
        else:
            self.dofs_c, self.normals = contact
        self.nc_loc = self.dofs_c.size
        self.nc = self.comm.allreduce(self.nc_loc, op=MPI.SUM)
        self.shape = (self.nc, self.nc)

        self.assemble(build=contact is None)
        self.f = self.K.createVecLeft()
        self.u = self.K.createVecRight()
        self.f.set(0.0)
        self.fc = np.empty_like(self.normals)

    def assemble(self, build=True):
        pc = self.ksp.getPC()
        if self.ksp.getType() != "preonly" or pc.getType() not in FACTOR_EVENTS:
            raise ValueError("Sc_nn requires preonly with LU or Cholesky.")

        def build_K():
            self.K.zeroEntries()
            assemble_matrix(self.K, self.problem.a, bcs=self.problem.bcs)
            self.K.assemble()

        def factor_K():
            self.ksp.setOperators(self.K)
            self.ksp.setErrorIfNotConverged(True)
            self.ksp.setUp()

        if build:
            _, self.times["assembly_s"] = timed(
                self.comm, build_K, self.local_times, "assembly_s"
            )

        # DOLFINx removes its prefixed options after setFromOptions, but MUMPS
        # reads -<prefix>mat_mumps_* during symbolic factorization: set them now.
        opts = PETSc.Options()
        prefix = self.ksp.getOptionsPrefix() or ""
        keys = []
        if self.mumps is not None:
            if pc.getFactorSolverType() != "mumps":
                raise ValueError("MUMPS controls require the MUMPS backend.")
            for kind in ("icntl", "cntl"):
                for i, value in self.mumps[kind].items():
                    keys.append(f"{prefix}mat_mumps_{kind}_{i}")
                    opts[keys[-1]] = value

        # PETSc event times are rank-local and need no extra collective.
        events = [PETSc.Log.Event(e) for e in FACTOR_EVENTS[pc.getType()]]
        before = [e.getPerfInfo()["time"] for e in events]
        _, self.times["factor_s"] = timed(
            self.comm, factor_K, self.local_times, "factor_s"
        )
        after = [e.getPerfInfo()["time"] for e in events]
        self.local_times["analysis_s"] = after[0] - before[0]
        self.local_times["numeric_factor_s"] = after[1] - before[1]
        for key in keys:
            del opts[key]

        if self.mumps is not None:
            F = pc.getFactorMatrix()
            self.mumps_report = mumps_report(F, self.mumps, self.dtype.itemsize)
            # Statistics are printed for analysis/factorization only, not per solve.
            F.setMumpsIcntl(4, 0)

    def __matmul__(self, fc_n):
        # All callers supply one value per owned contact node.
        # GPCG validates the layout once, avoiding a validation reduction here.
        fc_n = np.asarray(fc_n, dtype=self.dtype).reshape(-1)
        np.multiply(fc_n[:, None], self.normals, out=self.fc)
        f = self.f.array_w.reshape(-1, self.gdim)
        f[self.dofs_c] = self.fc
        del f
        if self.profile is None:
            self.ksp.solve(self.f, self.u)
        else:
            start = MPI.Wtime()
            self.ksp.solve(self.f, self.u)
            self.profile["solve_s"] += MPI.Wtime() - start
        uc = self.u.array_r.reshape(-1, self.gdim)[self.dofs_c]
        np.multiply(uc, self.normals, out=uc)
        return uc.sum(axis=1)

    def destroy(self):
        self.f.destroy()
        self.u.destroy()


def mumps_report(F, requested, itemsize):
    """Effective MUMPS controls and statistics; abort on any substitution."""

    def read(getter, i):
        try:
            return getter(i)
        except PETSc.Error:
            return None

    icntl = {i: read(F.getMumpsIcntl, i) for i in MUMPS_ICNTL_READABLE}
    cntl = {i: read(F.getMumpsCntl, i) for i in range(1, 8)}
    infog = [int(F.getMumpsInfog(i)) for i in range(1, 81)]
    rinfog = [float(F.getMumpsRinfog(i)) for i in range(1, 41)]

    mismatch = [
        f"ICNTL({i}) requested {v}, effective {icntl.get(i)}"
        for i, v in requested["icntl"].items()
        if icntl.get(i) is not None and int(icntl[i]) != int(v)
    ] + [
        f"CNTL({i}) requested {v:g}, effective {cntl.get(i)}"
        for i, v in requested["cntl"].items()
        if cntl.get(i) is None or not np.isclose(cntl[i], v, rtol=1e-12, atol=0.0)
    ]
    if mismatch:
        raise RuntimeError(
            "MUMPS did not accept the requested controls; refusing to run a "
            "substituted configuration: " + "; ".join(mismatch)
        )

    def entries(v):  # Negative INFOG sizes are in millions.
        return float(v) if v >= 0 else -1e6 * v

    blr = requested["icntl"].get(35, 0) > 0
    stats = {
        "analysis_type_INFOG32": infog[31],
        "ordering_INFOG7": infog[6],
        "scaling_INFOG33": infog[32],
        "factor_storage_entries_INFOG9": entries(infog[8]),
        "factor_storage_MiB": entries(infog[8]) * itemsize / 2**20,
        "fullrank_factor_entries_INFOG29": entries(infog[28]),
        "blr_factor_entries_INFOG35": entries(infog[34]),
        "flops_fullrank_RINFOG3": rinfog[2],
        "flops_effective_RINFOG14": rinfog[13],
        "mem_alloc_max_MB_INFOG18": infog[17],
        "mem_alloc_sum_MB_INFOG19": infog[18],
        "mem_used_max_MB_INFOG21": infog[20],
        "mem_used_sum_MB_INFOG22": infog[21],
        "max_front_INFOG11": infog[10],
        "pivots_INFOG12": infog[11],
        "delayed_pivots_INFOG13": infog[12],
    }
    warnings = []
    if infog[32] != requested["icntl"].get(8, infog[32]):
        warnings.append(
            f"MUMPS applied scaling INFOG(33)={infog[32]} instead of "
            f"ICNTL(8)={requested['icntl'][8]}."
        )
    if blr:
        warnings.append(
            "MUMPS 5.8.x: INFOG(35) may be inaccurate and INFOG(21-22) "
            "overestimated with BLR (fixed in 5.9.0)."
        )
        if rinfog[13] == 0.0:
            warnings.append(
                "RINFOG(14) reads 0 through the PETSc API; use the printed BLR "
                "statistics (mumps_printed) instead."
            )
    return {
        "icntl": icntl,
        "cntl": cntl,
        "infog": infog,
        "rinfog": rinfog,
        "stats": stats,
        "warnings": warnings,
    }


def reference_operator(Sc, args):
    """Uncompressed MUMPS compliance on a separate KSP (validation only)."""
    ksp = PETSc.KSP().create(Sc.comm)
    ksp.setOptionsPrefix("scref_")
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType(args.pc)
    pc.setFactorSolverType(args.backend)
    ref = Sc_nn(
        Sc.problem,
        None,
        mumps=mumps_settings(args, compression="none", print_level=0),
        ksp=ksp,
        contact=(Sc.dofs_c, Sc.normals),
    )
    return ref


def global_inf(comm, *vectors):
    local = np.array([np.max(np.abs(v), initial=0.0) for v in vectors])
    total = np.empty_like(local)
    comm.Allreduce(local, total, op=MPI.MAX)
    return total


def global_dot(comm, x, y):
    return comm.allreduce(float(np.dot(x, y)), op=MPI.SUM)


def linear_residual(K, ksp, rhs):
    """||K x - r|| / ||r|| for x from ksp; K is the original assembled matrix."""
    x, r = K.createVecRight(), K.createVecLeft()
    try:
        ksp.solve(rhs, x)
        K.mult(x, r)
        r.axpy(-1.0, rhs)
        return float(r.norm() / rhs.norm())
    finally:
        x.destroy()
        r.destroy()


class ScaledOperator:
    def __init__(self, M, factor):
        self.M = M
        self.factor = factor
        self.comm = M.comm
        self.nc_loc = M.nc_loc

    def __matmul__(self, z):
        return self.factor * (self.M @ z)


def gpcg(M, q, tol=1e-8, max_iter=10000, z0=None):
    comm = M.comm

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = q.size
    eps = np.finfo(np.float64).eps

    z = (
        np.zeros(n, dtype=np.float64)
        if z0 is None
        else np.maximum(np.asarray(z0, dtype=np.float64).reshape(-1), 0.0)
    )

    invalid = (
        n != M.nc_loc
        or z.size != n
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(z))
    )
    if comm.allreduce(invalid, op=MPI.LOR):
        raise ValueError(
            "q and z0 must be finite vectors matching the local operator size."
        )

    if tol < 0 or max_iter < 0:
        raise ValueError("tol and max_iter must be nonnegative.")

    def norms(*vectors):
        # Compute several global infinity norms in one reduction.
        local = np.array([np.max(np.abs(v), initial=0.0) for v in vectors])
        total = np.empty_like(local)
        comm.Allreduce(local, total, op=MPI.MAX)
        return total

    def dot(x, y):
        return comm.allreduce(float(x @ y), op=MPI.SUM)

    matvec_count = 0
    cg_iterations = 0
    active_set_changes = 0
    work_count = 0
    numerical_breakdown = False
    profile = {"cg_s": 0.0, "matvec_cg_s": 0.0, "matvec_other_s": 0.0}

    def apply(x, in_cg=False):
        # No additional MPI collective in this fine-grained timer.
        nonlocal matvec_count
        start = MPI.Wtime()
        y = M @ x
        profile["matvec_cg_s" if in_cg else "matvec_other_s"] += MPI.Wtime() - start
        matvec_count += 1
        return y

    w = apply(z) + q
    fresh = True

    active = (z == 0.0) & (w >= 0.0)

    while work_count < max_iter:
        z_norm, w_norm, residual = norms(z, w, np.minimum(z, w))
        scale = 1.0 + z_norm + w_norm

        if residual <= tol * scale:
            if fresh:
                break

            w = apply(z) + q
            fresh = True
            continue

        r = np.where(active, 0.0, -w)

        if norms(r)[0] <= tol * scale:
            candidates = np.where(active & (w < -tol * scale), w, np.inf)
            j = int(np.argmin(candidates)) if n else -1
            value = float(candidates[j]) if n else np.inf

            value, rank = comm.allreduce((value, comm.rank), op=MPI.MINLOC)

            if not np.isfinite(value):
                if fresh:
                    break

                w = apply(z) + q
                fresh = True
                continue

            # Only its owning rank changes the selected constraint.
            if comm.rank == rank:
                active[j] = False

            active_set_changes += 1
            work_count += 1
            continue

        d = r.copy()
        rr = dot(r, r)

        # CG on the current free face.
        cg_start = MPI.Wtime()
        while work_count < max_iter:
            d[active] = 0.0

            Md = apply(d, in_cg=True)
            curvature = dot(d, Md)

            if not np.isfinite(curvature) or curvature <= 0.0:
                numerical_breakdown = True
                break

            alpha_cg = rr / curvature

            # Global feasible step length.
            decreasing = d < 0.0
            local_step = np.min(
                -z[decreasing] / d[decreasing],
                initial=np.inf,
            )
            alpha_max = comm.allreduce(float(local_step), op=MPI.MIN)
            alpha = min(alpha_cg, alpha_max)

            if alpha < 0.0 or not np.isfinite(alpha):
                numerical_breakdown = True
                break

            z += alpha * d
            w += alpha * Md
            fresh = False

            cg_iterations += 1
            work_count += 1

            z_norm = norms(z)[0]
            zero_tol = 1000.0 * eps * max(1.0, z_norm)

            if comm.allreduce(bool(np.any(z < -zero_tol)), op=MPI.LOR):
                numerical_breakdown = True
                break

            hit_boundary = np.isfinite(alpha_max) and alpha_max <= alpha_cg

            snap = z < 0.0

            if hit_boundary:
                hit = (z <= zero_tol) & decreasing
                active[hit] = True
                snap |= hit
                active_set_changes += 1

            changed = comm.allreduce(bool(np.any(z[snap] != 0.0)), op=MPI.LOR)
            z[snap] = 0.0

            if changed:
                w = apply(z, in_cg=True) + q
                fresh = True

            if hit_boundary:
                break

            r_new = np.where(active, 0.0, -w)
            z_norm, w_norm, r_norm = norms(z, w, r_new)

            if r_norm <= tol * (1.0 + z_norm + w_norm):
                break

            rr_new = dot(r_new, r_new)

            if not np.isfinite(rr_new) or rr_new <= 0.0:
                numerical_breakdown = True
                break

            if changed:
                break

            d *= rr_new / rr
            d += r_new
            rr = rr_new

        profile["cg_s"] += MPI.Wtime() - cg_start
        if numerical_breakdown:
            break

    # Compute the final residual from the actual operator action.
    if not fresh:
        w = apply(z) + q

    z_norm, w_norm, residual = norms(z, w, np.minimum(z, w))
    rel_residual = residual / (1.0 + z_norm + w_norm)

    info = {
        "converged": bool(not numerical_breakdown and rel_residual <= tol),
        "numerical_breakdown": numerical_breakdown,
        "cg_iterations": cg_iterations,
        "active_set_changes": active_set_changes,
        "work_count": work_count,
        "matvec_count": matvec_count,
        "residual": float(residual),
        "rel_residual": float(rel_residual),
        "local_times": profile,
    }

    return z, w, info


def external_load(problem):
    """Assemble external data once; tc is zero in this benchmark."""
    b = assemble_vector(problem.L)
    apply_lifting(b, [problem.a], bcs=[problem.bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    for bc in problem.bcs:
        bc.set(b.array_w)
    return b


def export_vtk(Sc, fc_n, b, filename):
    V = Sc.problem.u.function_space
    u = Function(V, name="u")
    fc = Function(V, name="fc")
    fc.x.array.reshape(-1, Sc.gdim)[Sc.dofs_c] = -fc_n[:, None] * Sc.normals
    fc.x.scatter_forward()
    f = b.copy()
    try:
        f.axpy(1.0, fc.x.petsc_vec)
        Sc.ksp.solve(f, u.x.petsc_vec)
        u.x.scatter_forward()
        with VTKFile(Sc.comm, str(filename), "w") as vtk:
            vtk.write_mesh(V.mesh)
            vtk.write_function([u, fc], 0.0)
    finally:
        f.destroy()


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0**2 if sys.platform == "darwin" else 1024.0)


def exclusive_times(local):
    """Disjoint intervals on ONE rank; never subtract independent rank maxima."""
    parts = {
        k: local[k]
        for k in (
            "mesh_s",
            "forms_s",
            "normals_s",
            "assembly_s",
            "factor_s",
            "load_assembly_s",
            "external_solve_s",
        )
    }
    if "analysis_s" in local:
        # Split factor_s into PETSc symbolic/numeric events plus the remainder.
        parts["analysis_s"] = local["analysis_s"]
        parts["numeric_factor_s"] = local["numeric_factor_s"]
        parts["factor_other_s"] = max(
            0.0, parts.pop("factor_s") - local["analysis_s"] - local["numeric_factor_s"]
        )
    parts["operator_other_s"] = max(
        0.0,
        local["operator_setup_s"]
        - sum(local[k] for k in ("normals_s", "assembly_s", "factor_s")),
    )
    parts["cg_matvec_s"] = local["matvec_cg_s"]
    parts["cg_other_s"] = max(0.0, local["cg_s"] - local["matvec_cg_s"])
    parts["other_matvec_s"] = local["matvec_other_s"]
    parts["gpcg_other_s"] = max(
        0.0, local["gpcg_s"] - local["cg_s"] - local["matvec_other_s"]
    )
    parts["pipeline_other_s"] = max(0.0, local["pipeline_s"] - sum(parts.values()))
    return parts


def worker(args):
    load_fem()
    comm = MPI.COMM_WORLD
    if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
        raise RuntimeError("This real contact LCP benchmark needs real-scalar PETSc.")
    if args.backend == "mumps" and not PETSc.Sys.hasExternalPackage("mumps"):
        raise RuntimeError("PETSc was built without MUMPS; use a build with MUMPS.")
    if args.compression == "blr" and args.backend != "mumps":
        raise RuntimeError(
            f"BLR compression is only implemented for MUMPS, not '{args.backend}'."
        )

    out = Path(args.out).resolve()
    if comm.rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    # Default PETSc logging is always on (identical overhead for every
    # configuration); it provides the symbolic/numeric factorization events.
    PETSc.Log.begin()
    mumps = mumps_settings(args) if args.backend == "mumps" else None

    L = args.length
    times = {}
    local_times = {}
    comm.Barrier()
    start = MPI.Wtime()
    mesh, times["mesh_s"] = timed(
        comm,
        lambda: create_box(
            comm, [[0.0, 0.0, 0.0], [L, L, L]], args.mesh, CellType.hexahedron
        ),
        local_times,
        "mesh_s",
    )

    def locate_u(x):
        return np.isclose(x[2], 0.0, rtol=0.0, atol=1e-10 * L)

    def locate_c(x):
        return np.isclose(x[2], L, rtol=0.0, atol=1e-10 * L)

    def locate_t(x):
        return np.isclose(x[1], L, rtol=0.0, atol=1e-10 * L) & (
            (x[0] - L / 2) ** 2 + (x[2] - L / 2) ** 2 <= (0.2 * L) ** 2
        )

    result, times["forms_s"] = timed(
        comm,
        lambda: problem_setup(
            mesh,
            ("Lagrange", 1),
            {"E": args.E, "nu": args.nu},
            locate_u,
            locate_t,
            locate_c,
            t0=[0.0, 0.0, args.traction_z],
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": args.pc,
                "pc_factor_mat_solver_type": args.backend,
            },
        ),
        local_times,
        "forms_s",
    )
    problem, mt, f_v, t0, tc = result
    Sc, times["operator_setup_s"] = timed(
        comm, lambda: Sc_nn(problem, mt, mumps=mumps), local_times, "operator_setup_s"
    )
    times.update(Sc.times)
    local_times.update(Sc.local_times)
    V = problem.u.function_space
    x_c = V.tabulate_dof_coordinates()[Sc.dofs_c, : Sc.gdim]
    if Sc.nc == 0:
        raise RuntimeError("The top face contains no contact nodes.")

    # A parabolic rigid obstacle, a local spherical-profile approximation.
    r2 = (x_c[:, 0] - L / 2) ** 2 + (x_c[:, 1] - L / 2) ** 2
    g0 = r2 / (2.0 * args.radius) - args.indentation

    b, times["load_assembly_s"] = timed(
        comm, lambda: external_load(problem), local_times, "load_assembly_s"
    )
    u_t = Function(V, name="u_t")
    _, times["external_solve_s"] = timed(
        comm, lambda: Sc.ksp.solve(b, u_t.x.petsc_vec), local_times, "external_solve_s"
    )
    # Only owned nodes are read, so no ghost update is needed for this projection.
    uc_t = u_t.x.array.reshape(-1, Sc.gdim)[Sc.dofs_c]
    uc_t_n = np.sum(uc_t * Sc.normals, axis=1)
    g_t = g0 - uc_t_n

    # Nondimensionalization avoids comparing forces [N] with gaps [m].
    # fc_n = force_scale*z, g = gap_scale*w.
    gap_scale = args.indentation
    force_scale = args.E * L * gap_scale / Sc.nc
    M = ScaledOperator(Sc, force_scale / gap_scale)
    Sc.profile = {"solve_s": 0.0}
    (z, w, info), times["gpcg_s"] = timed(
        comm,
        lambda: gpcg(M, g_t / gap_scale, tol=args.tol, max_iter=args.max_iter),
        local_times,
        "gpcg_s",
    )
    local_times["triangular_solve_s"] = Sc.profile["solve_s"]
    Sc.profile = None
    fc_n, g = force_scale * z, gap_scale * w
    local_times["pipeline_s"] = MPI.Wtime() - start
    times["pipeline_s"] = comm.allreduce(local_times["pipeline_s"], op=MPI.MAX)
    # Peak memory of the measured pipeline, before microbenchmarks, export and
    # the reference validation.
    pipeline_peak_rss = peak_rss_mib()
    local_times.update(info.pop("local_times"))
    local_times["gpcg_matvec_s"] = (
        local_times["matvec_cg_s"] + local_times["matvec_other_s"]
    )
    local_times["gpcg_matvec_mean_s"] = (
        local_times["gpcg_matvec_s"] / info["matvec_count"]
    )
    # Collect once AFTER the timed pipeline. Fine timers add no collectives.
    profiles = comm.allgather({"rank": comm.rank, "times": local_times})
    critical = max(profiles, key=lambda r: r["times"]["pipeline_s"])
    breakdown = exclusive_times(critical["times"])
    for key in (
        "cg_s",
        "gpcg_matvec_s",
        "gpcg_matvec_mean_s",
        "triangular_solve_s",
        "analysis_s",
        "numeric_factor_s",
    ):
        times[key] = max(r["times"][key] for r in profiles)
    # Cumulative triangular-solve time inside GPCG (max over ranks).
    times["gpcg_solve_s"] = times["triangular_solve_s"]

    # Fixed-work benchmarks with identical nonzero data in every configuration;
    # they exclude setup and the configuration-dependent GPCG iteration count.
    probe = 1.0 + 0.2 * np.cos(2 * np.pi * x_c[:, 0] / L)
    probe *= 1.0 + 0.1 * np.sin(2 * np.pi * x_c[:, 1] / L)

    def benchmark(operation):
        for _ in range(args.matvec_warmup):
            operation()
        samples = []

        def batch():
            for _ in range(args.matvecs):
                operation()

        for _ in range(args.batches):
            _, elapsed = timed(comm, batch)
            samples.append(elapsed / args.matvecs)
        return samples

    # (a) Complete Sc_nn application: scatter, triangular solves, projection.
    batch_times = benchmark(lambda: Sc @ probe)
    times["matvec_s"] = statistics.median(batch_times)
    # (b) Linear solve only, with the contact load N * probe as right-hand side.
    rhs = Sc.K.createVecLeft()
    rhs.set(0.0)
    rhs.array_w.reshape(-1, Sc.gdim)[Sc.dofs_c] = probe[:, None] * Sc.normals
    x_solve = Sc.K.createVecRight()
    solve_batch_times = benchmark(lambda: Sc.ksp.solve(rhs, x_solve))
    times["solve_s"] = statistics.median(solve_batch_times)
    x_solve.destroy()

    # Representative linear-solve residuals against the original assembled K.
    b_norm = b.norm()
    linear_residuals = {
        "contact_probe": linear_residual(Sc.K, Sc.ksp, rhs),
        "external_load": linear_residual(Sc.K, Sc.ksp, b) if b_norm > 0 else None,
        "external_load_norm": float(b_norm),
    }

    # Is the compliance actually applied symmetric positive definite (CG)?
    y_probe = 1.0 + 0.3 * np.sin(np.pi * x_c[:, 0] / L) * np.cos(np.pi * x_c[:, 1] / L)
    Sx, Sy = Sc @ probe, Sc @ y_probe
    xSx, ySy = global_dot(comm, probe, Sx), global_dot(comm, y_probe, Sy)
    xSy, ySx = global_dot(comm, probe, Sy), global_dot(comm, y_probe, Sx)
    operator_check = {
        "symmetry_defect": abs(xSy - ySx) / np.sqrt(abs(xSx * ySy)),
        "min_rayleigh_m_per_N": min(
            xSx / global_dot(comm, probe, probe),
            ySy / global_dot(comm, y_probe, y_probe),
        ),
    }

    # Numerical checks and peak memory are outside the pipeline timing.
    diagnostics = {
        "total_force_N": comm.allreduce(float(fc_n.sum()), op=MPI.SUM),
        "min_gap_m": comm.allreduce(float(np.min(g, initial=np.inf)), op=MPI.MIN),
        "min_force_N": comm.allreduce(float(np.min(fc_n, initial=np.inf)), op=MPI.MIN),
        "max_complementarity_Nm": comm.allreduce(
            float(np.max(np.abs(fc_n * g), initial=0.0)), op=MPI.MAX
        ),
    }
    diagnostics["feasibility_scaled"] = max(0.0, -diagnostics["min_gap_m"]) / gap_scale
    diagnostics["complementarity_scaled"] = diagnostics["max_complementarity_Nm"] / (
        force_scale * gap_scale
    )
    local_layout = {
        "rank": comm.rank,
        "cells": mesh.topology.index_map(mesh.topology.dim).size_local,
        "dofs": V.dofmap.index_map.size_local * V.dofmap.index_map_bs,
        "contact_nodes": Sc.nc_loc,
        "peak_rss_mib": pipeline_peak_rss,
    }
    layouts = comm.allgather(local_layout)
    # Coordinate-based ordering removes MPI partition-dependent DOF numbering.
    local_contact = np.column_stack((x_c, fc_n, g, g0, uc_t_n))
    pieces = comm.gather(local_contact, root=0)
    if comm.rank == 0:
        contact = np.vstack(pieces)
        order = np.lexsort((contact[:, 2], contact[:, 1], contact[:, 0]))
        contact = contact[order]
        np.savez(
            out / "contact.npz",
            x=contact[:, :3],
            fc_n=contact[:, 3],
            g=contact[:, 4],
            g0=contact[:, 5],
            uc_t_n=contact[:, 6],
        )

    times["export_s"] = 0.0
    if args.export:
        _, times["export_s"] = timed(
            comm, lambda: export_vtk(Sc, fc_n, b, out / "solution.pvd")
        )
    if args.petsc_log:
        viewer = PETSc.Viewer().createASCII(str(out / "petsc.log"), comm=comm)
        PETSc.Log.view(viewer)
        viewer.destroy()

    # Accuracy for the ORIGINAL problem: recompute the external gap and the
    # contact gaps of the BLR forces with an uncompressed factorization. This
    # runs after every timing and the RSS peak were recorded.
    reference_check = None
    if args.compression == "blr" and args.reference_check:
        Sc.ksp.getPC().reset()  # Release the BLR factors first.
        comm.Barrier()
        ref_start = MPI.Wtime()
        ref = reference_operator(Sc, args)
        u_ref = Function(V, name="u_ref")
        ref.ksp.solve(b, u_ref.x.petsc_vec)
        uc_ref_n = np.sum(
            u_ref.x.array.reshape(-1, Sc.gdim)[Sc.dofs_c] * Sc.normals, axis=1
        )
        g_t_ref = g0 - uc_ref_n
        g_true = g_t_ref + ref @ fc_n
        w_true = g_true / gap_scale
        z_norm, w_norm, res = global_inf(comm, z, w_true, np.minimum(z, w_true))
        min_gap = comm.allreduce(float(np.min(g_true, initial=np.inf)), op=MPI.MIN)
        compl = comm.allreduce(
            float(np.max(np.abs(fc_n * g_true), initial=0.0)), op=MPI.MAX
        )
        reference_check = {
            "rel_residual": float(res / (1.0 + z_norm + w_norm)),
            "min_gap_m": min_gap,
            "feasibility_scaled": max(0.0, -min_gap) / gap_scale,
            "complementarity_scaled": compl / (force_scale * gap_scale),
            "gap_change_scaled_linf": float(
                global_inf(comm, g_true - g)[0] / gap_scale
            ),
            "external_gap_change_scaled_linf": float(
                global_inf(comm, g_t_ref - g_t)[0] / gap_scale
            ),
            "reference_linear_residual": linear_residual(Sc.K, ref.ksp, rhs),
            "reference_mumps_stats": ref.mumps_report["stats"],
            "reference_factor_s": ref.times["factor_s"],
        }
        ref.destroy()
        ref.ksp.destroy()
        reference_check["elapsed_s"] = comm.allreduce(
            MPI.Wtime() - ref_start, op=MPI.MAX
        )
    rhs.destroy()

    import dolfinx

    pc = Sc.ksp.getPC()
    record = {
        "schema_version": 3,
        "status": "ok",
        "ranks": comm.size,
        "mesh": args.mesh,
        "ndofs": V.dofmap.index_map.size_global * V.dofmap.index_map_bs,
        "nc": Sc.nc,
        "model": {
            "length": L,
            "E": args.E,
            "nu": args.nu,
            "radius": args.radius,
            "indentation": args.indentation,
            "traction_z": args.traction_z,
        },
        "pc": args.pc,
        "backend": args.backend,
        "compression": args.compression,
        "blr_tol": args.blr_tol if args.compression == "blr" else None,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "initial_guess": "zero",
        "gap_scale": gap_scale,
        "force_scale": force_scale,
        "solver": {
            "options_prefix": Sc.ksp.getOptionsPrefix(),
            "ksp_type": Sc.ksp.getType(),
            "pc_type": pc.getType(),
            "factor_solver_type": args.backend,
            "ordering": args.ordering,
            "requested": mumps,
            "mumps": (
                None
                if Sc.mumps_report is None
                else {k: v for k, v in Sc.mumps_report.items()}
            ),
        },
        "times": times,
        "profile_by_rank": profiles,
        "breakdown_rank": critical["rank"],
        "breakdown": breakdown,
        "breakdown_total_s": critical["times"]["pipeline_s"],
        "matvec_batches_s": batch_times,
        "solve_batches_s": solve_batch_times,
        "info": info,
        "diagnostics": diagnostics,
        "linear_residuals": linear_residuals,
        "operator_check": operator_check,
        "reference_check": reference_check,
        "layout": layouts,
        "rss_peak_max_mib": max(row["peak_rss_mib"] for row in layouts),
        "rss_peak_sum_mib": sum(row["peak_rss_mib"] for row in layouts),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "dolfinx": dolfinx.__version__,
            "petsc": list(PETSc.Sys.getVersion()),
            "mumps": mumps_version(),
            "mpi": MPI.Get_library_version().strip().rstrip("\x00"),
        },
        "threads": {key: os.environ.get(key) for key in THREAD_VARS},
        "host": platform.node(),
    }
    if comm.rank == 0:
        (out / "result.json").write_text(json.dumps(record, indent=2))
        label = "none" if args.compression == "none" else f"blr eps={args.blr_tol:g}"
        print(
            f"ranks={comm.size} ndofs={record['ndofs']} nc={Sc.nc} "
            f"pc={args.pc} compression={label} "
            f"converged={info['converged']} matvecs={info['matvec_count']}",
            flush=True,
        )
        print(
            "  "
            + "  ".join(
                f"{k}={times[k]:.6g}"
                for k in (
                    "assembly_s",
                    "analysis_s",
                    "numeric_factor_s",
                    "factor_s",
                    "solve_s",
                    "matvec_s",
                    "gpcg_s",
                    "pipeline_s",
                )
            ),
            flush=True,
        )
        if Sc.mumps_report is not None:
            s = Sc.mumps_report["stats"]
            print(
                f"  MUMPS: factor storage INFOG(9)={s['factor_storage_MiB']:.1f} MiB, "
                f"full-rank entries INFOG(29)={s['fullrank_factor_entries_INFOG29']:.4g}, "
                f"scaling INFOG(33)={s['scaling_INFOG33']}; "
                f"peak RSS max/rank={record['rss_peak_max_mib']:.1f} MiB",
                flush=True,
            )
            for warning in Sc.mumps_report["warnings"]:
                print("  note: " + warning, flush=True)
        print(
            f"  total force={diagnostics['total_force_N']:.9g} N; "
            f"scaled residual={info['rel_residual']:.3e}; "
            f"K-residual(probe)={linear_residuals['contact_probe']:.3e}; "
            f"symmetry defect={operator_check['symmetry_defect']:.2e}",
            flush=True,
        )
        if reference_check is not None:
            print(
                "  reference (uncompressed) contact residual="
                f"{reference_check['rel_residual']:.3e}; scaled gap change="
                f"{reference_check['gap_change_scaled_linf']:.3e}",
                flush=True,
            )
        print(f"  output={out}", flush=True)
    b.destroy()
    Sc.destroy()
    return 0 if info["converged"] else 2


def run_worker(args):
    """Write error_rank<r>.json on failure and abort so no rank hangs."""
    try:
        return worker(args)
    except Exception as exc:
        from mpi4py import MPI as _MPI

        comm = _MPI.COMM_WORLD
        out = Path(args.out).resolve()
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"error_rank{comm.rank}.json").write_text(
                json.dumps(
                    {
                        "rank": comm.rank,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
            )
        except OSError:
            pass
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        if comm.size > 1:
            comm.Abort(1)
        raise


THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def compare_contact(reference, candidate, gap_scale, length):
    from scipy.spatial import cKDTree

    with np.load(reference) as ref, np.load(candidate) as cur:
        check = {
            "coordinates_match": False,
            "force_rel_l2": None,
            "gap_scaled_linf": None,
        }

        if ref["x"].shape != cur["x"].shape:
            return check

        dist, order = cKDTree(cur["x"]).query(ref["x"])

        if np.any(dist > 1e-11 * length) or np.unique(order).size != order.size:
            return check

        df = cur["fc_n"][order] - ref["fc_n"]
        dg = cur["g"][order] - ref["g"]

        check.update(
            coordinates_match=True,
            force_rel_l2=float(
                np.linalg.norm(df)
                / max(np.linalg.norm(ref["fc_n"]), np.finfo(float).tiny)
            ),
            gap_scaled_linf=float(np.max(np.abs(dg), initial=0.0) / gap_scale),
        )
        return check


TIME_KEYS = (
    "mesh_s",
    "forms_s",
    "normals_s",
    "assembly_s",
    "analysis_s",
    "numeric_factor_s",
    "factor_s",
    "load_assembly_s",
    "external_solve_s",
    "gpcg_solve_s",
    "triangular_solve_s",
    "cg_s",
    "gpcg_matvec_s",
    "gpcg_matvec_mean_s",
    "gpcg_s",
    "pipeline_s",
    "solve_s",
    "matvec_s",
    "export_s",
)
COUNT_KEYS = ("work_count", "cg_iterations", "active_set_changes", "matvec_count")
MUMPS_KEYS = (
    "factor_storage_MiB",
    "factor_storage_entries_INFOG9",
    "fullrank_factor_entries_INFOG29",
    "blr_factor_entries_INFOG35",
    "flops_fullrank_RINFOG3",
    "mem_used_max_MB_INFOG21",
    "mem_used_sum_MB_INFOG22",
)
PRINTED_KEYS = (
    "blr_fronts",
    "blr_fraction_pct",
    "blr_entries_printed",
    "blr_flops_printed",
    "mumps_analysis_elapsed_s",
    "mumps_factor_elapsed_s",
)
ACCURACY_KEYS = (
    "force_rel_l2_seq",
    "gap_scaled_linf_seq",
    "force_rel_l2_rank",
    "gap_scaled_linf_rank",
    "gpcg_rel_residual",
    "reference_rel_residual",
    "reference_feasibility_scaled",
    "reference_complementarity_scaled",
    "reference_gap_change_scaled",
    "linear_residual_probe",
    "linear_residual_load",
    "symmetry_defect",
    "min_rayleigh",
    "total_force_N",
    "feasibility_scaled",
    "complementarity_scaled",
)
DEFAULT_THRESHOLDS = {
    "check_rtol": 1e-5,
    "accuracy_force_rtol": 1e-3,
    "accuracy_gap_tol": 1e-3,
    "accuracy_residual_tol": 1e-3,
}


def normalize(record):
    """Defaults that make schema-2 (pre-BLR) records readable."""
    record.setdefault("status", "ok")
    record.setdefault("compression", "none")
    record.setdefault("blr_tol", None)
    record.setdefault("backend", "mumps")
    record["mesh"] = [int(n) for n in record["mesh"]]
    return record


def mesh_tag(mesh):
    return "x".join(str(n) for n in mesh)


def config_tag(compression, tol):
    return "none" if compression == "none" else f"blr{tol:.0e}"


def config_label(compression, tol, pc=None):
    label = "no compression" if compression == "none" else f"BLR ε={tol:.0e}"
    if pc is None:
        return label
    return f"{label}, {'LDLᵀ' if pc == 'cholesky' else 'LU'}"


def config_key(r):
    return (tuple(r["mesh"]), r["pc"], r["backend"], r["compression"], r["blr_tol"])


def run_metrics(r):
    """Flat scalar metrics of one successful run (None when not available)."""
    m = {k: r["times"].get(k) for k in TIME_KEYS}
    m.update({k: r["info"].get(k) for k in COUNT_KEYS})
    m["rss_peak_max_mib"] = r.get("rss_peak_max_mib")
    m["rss_peak_sum_mib"] = r.get("rss_peak_sum_mib")
    stats = (((r.get("solver") or {}).get("mumps")) or {}).get("stats") or {}
    m.update({k: stats.get(k) for k in MUMPS_KEYS})
    printed = r.get("mumps_printed") or {}
    m.update({k: printed.get(k) for k in PRINTED_KEYS})
    lin = r.get("linear_residuals") or {}
    m["linear_residual_probe"] = lin.get("contact_probe")
    m["linear_residual_load"] = lin.get("external_load")
    op = r.get("operator_check") or {}
    m["symmetry_defect"] = op.get("symmetry_defect")
    m["min_rayleigh"] = op.get("min_rayleigh_m_per_N")
    d = r.get("diagnostics") or {}
    for k in ("total_force_N", "feasibility_scaled", "complementarity_scaled"):
        m[k] = d.get(k)
    m["gpcg_rel_residual"] = r["info"].get("rel_residual")
    ref = r.get("reference_check") or {}
    m["reference_rel_residual"] = ref.get("rel_residual")
    m["reference_feasibility_scaled"] = ref.get("feasibility_scaled")
    m["reference_complementarity_scaled"] = ref.get("complementarity_scaled")
    m["reference_gap_change_scaled"] = ref.get("gap_change_scaled_linf")
    for name, check in (("seq", r.get("check")), ("rank", r.get("check_rank"))):
        check = check or {}
        match = check.get("coordinates_match")
        m["force_rel_l2_" + name] = check.get("force_rel_l2") if match else None
        m["gap_scaled_linf_" + name] = check.get("gap_scaled_linf") if match else None
    return m


def validate(records, thresholds):
    """Compare every run with the sequential and the same-rank uncompressed run."""
    records = [normalize(r) for r in records]
    ok = [r for r in records if r["status"] == "ok"]

    def first(candidates):
        return min(
            candidates,
            key=lambda q: (q.get("repeat", 0), q.get("directory", "")),
            default=None,
        )

    def baselines(r, ranks):
        return [
            q
            for q in ok
            if tuple(q["mesh"]) == tuple(r["mesh"])
            and q["pc"] == r["pc"]
            and q["backend"] == r["backend"]
            and q["compression"] == "none"
            and q["ranks"] == ranks
        ]

    for r in records:
        reasons, notes = [], []
        if r["status"] != "ok":
            reasons.append(f"run {r['status']} (exit {r.get('exit_code')})")
        else:
            blr = r["compression"] == "blr"
            if not r["info"]["converged"]:
                reasons.append("GPCG not converged")
            if r["info"].get("numerical_breakdown"):
                reasons.append("GPCG numerical breakdown")
            same_rank = baselines(r, r["ranks"])
            same_repeat = [q for q in same_rank if q.get("repeat") == r.get("repeat")]
            refs = {
                "seq": first(baselines(r, 1)),
                "rank": first(same_repeat or same_rank),
            }
            if blr:
                ftol = thresholds["accuracy_force_rtol"]
                gtol = thresholds["accuracy_gap_tol"]
            else:
                ftol = gtol = thresholds["check_rtol"]
            for name, ref in refs.items():
                key = "check" if name == "seq" else "check_rank"
                what = "sequential" if name == "seq" else "same-rank"
                if ref is None:
                    r[key] = None
                    reasons.append(f"no successful {what} uncompressed reference")
                    continue
                if ref["ndofs"] != r["ndofs"] or ref["nc"] != r["nc"]:
                    reasons.append(f"problem size differs from {what} reference")
                files = [Path(q["directory"]) / "contact.npz" for q in (ref, r)]
                if all(f.exists() for f in files):
                    r[key] = compare_contact(
                        files[0], files[1], r["gap_scale"], r["model"]["length"]
                    )
                elif r.get(key) is None:
                    reasons.append(f"contact.npz missing for {what} comparison")
                    continue
                check = r[key]
                if not check["coordinates_match"]:
                    reasons.append(f"contact coordinates differ from {what} reference")
                    continue
                if check["force_rel_l2"] > ftol:
                    reasons.append(
                        f"force error vs {what} {check['force_rel_l2']:.2e} > {ftol:g}"
                    )
                if check["gap_scaled_linf"] > gtol:
                    reasons.append(
                        f"gap error vs {what} {check['gap_scaled_linf']:.2e} > {gtol:g}"
                    )
            if blr:
                ref_check = r.get("reference_check")
                if ref_check is None:
                    notes.append("original-problem residual not checked")
                elif ref_check["rel_residual"] > thresholds["accuracy_residual_tol"]:
                    reasons.append(
                        "original-problem contact residual "
                        f"{ref_check['rel_residual']:.2e} > "
                        f"{thresholds['accuracy_residual_tol']:g}"
                    )
                printed = r.get("mumps_printed") or {}
                if printed.get("blr_fronts") == 0:
                    notes.append("BLR active but no front was compressed")
        r["valid"] = not reasons
        r["validation"] = {"reasons": reasons, "notes": notes, "thresholds": thresholds}
    return records


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    return statistics.median(values), min(values), max(values)


def summarize(records, out, settings=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = [normalize(r) for r in records]
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        raise ValueError("No successful measured runs found.")
    # Only the factorization configuration may differ between compared runs.
    for key in ("model", "tol", "threads", "max_iter"):
        if any(r.get(key) != ok[0].get(key) for r in ok):
            raise ValueError(f"Cannot combine different {key} values in one summary.")
    sizes = {}
    for r in ok:
        if sizes.setdefault(tuple(r["mesh"]), (r["ndofs"], r["nc"])) != (
            r["ndofs"],
            r["nc"],
        ):
            raise ValueError(f"Inconsistent problem size for mesh {r['mesh']}.")

    groups = {}
    for r in records:
        groups.setdefault(config_key(r) + (r["ranks"],), []).append(r)

    def order(key):
        mesh, pc, backend, compression, tol, ranks = key
        return (
            int(np.prod(mesh)),
            mesh,
            pc,
            backend,
            compression != "none",
            tol or 0.0,
            ranks,
        )

    summary_keys = (
        TIME_KEYS
        + COUNT_KEYS
        + ("rss_peak_max_mib", "rss_peak_sum_mib")
        + MUMPS_KEYS
        + PRINTED_KEYS
        + ACCURACY_KEYS
    )
    rows = []
    for key in sorted(groups, key=order):
        mesh, pc, backend, compression, tol, ranks = key
        group = groups[key]
        good = [r for r in group if r["status"] == "ok"]
        ndofs, nc = sizes.get(mesh, (None, None))
        issues = sorted(
            {
                reason
                for r in group
                for reason in (r.get("validation") or {}).get("reasons", [])
            }
            | {
                note
                for r in group
                for note in (r.get("validation") or {}).get("notes", [])
            }
        )
        row = {
            "mesh": mesh_tag(mesh),
            "mesh_nx": mesh[0],
            "mesh_ny": mesh[1],
            "mesh_nz": mesh[2],
            "ranks": ranks,
            "pc": pc,
            "backend": backend,
            "compression": compression,
            "blr_tol": tol,
            "config": config_label(compression, tol),
            "repeats": len(good),
            "failed_runs": len(group) - len(good),
            "valid": all(r.get("valid", False) for r in group),
            "converged_all": bool(good) and all(r["info"]["converged"] for r in good),
            "breakdown_any": any(r["info"].get("numerical_breakdown") for r in good),
            "issues": "; ".join(issues),
            "ndofs": ndofs,
            "nc": nc,
            "length": ok[0]["model"]["length"],
        }
        metrics = [run_metrics(r) for r in good]
        for k in summary_keys:
            row[k], row[k + "_min"], row[k + "_max"] = _stats(m[k] for m in metrics)
        # Means preserve additivity. Independently computed medians need not.
        if good and all("breakdown" in r for r in good):
            for k in sorted({k for r in good for k in r["breakdown"]}):
                row["breakdown_" + k] = statistics.mean(
                    r["breakdown"].get(k, 0.0) for r in good
                )
            row["breakdown_total_s"] = statistics.mean(
                r["breakdown_total_s"] for r in good
            )
            row["breakdown_triangular_solve_s"] = statistics.mean(
                next(
                    p["times"]["triangular_solve_s"]
                    for p in r["profile_by_rank"]
                    if p["rank"] == r["breakdown_rank"]
                )
                for r in good
            )
        rows.append(row)

    def find(row, **changes):
        target = {
            **{
                k: row[k]
                for k in ("mesh", "pc", "backend", "compression", "blr_tol", "ranks")
            },
            **changes,
        }
        return next(
            (q for q in rows if all(q[k] == v for k, v in target.items())), None
        )

    for row in rows:
        # Strong scaling within one configuration (unchanged definition).
        base = find(row, ranks=1)
        for key in ("pipeline", "factor", "gpcg", "matvec", "solve"):
            usable = (
                base is not None
                and base["valid"]
                and row["valid"]
                and base.get(key + "_s")
                and row.get(key + "_s")
            )
            row[key + "_speedup"] = (
                base[key + "_s"] / row[key + "_s"] if usable else None
            )
            row[key + "_efficiency"] = (
                row[key + "_speedup"] / row["ranks"] if usable else None
            )
        # BLR versus the uncompressed run at the SAME mesh and rank count.
        if row["compression"] != "blr":
            continue
        base = find(row, compression="none", blr_tol=None)
        row["blr_baseline_valid"] = bool(base and base["valid"])

        def ratio(a, b):
            return a / b if a is not None and b else None

        for key in (
            "assembly",
            "analysis",
            "numeric_factor",
            "factor",
            "solve",
            "matvec",
            "gpcg_solve",
            "gpcg",
            "pipeline",
        ):
            row["blr_speedup_" + key] = (
                ratio(base.get(key + "_s"), row.get(key + "_s")) if base else None
            )
        for key in ("rss_peak_max_mib", "rss_peak_sum_mib"):
            r = ratio(row.get(key), base.get(key)) if base else None
            row["blr_reduction_" + key] = None if r is None else 1.0 - r
        row["blr_factor_storage_ratio"] = (
            ratio(
                row.get("factor_storage_entries_INFOG9"),
                base.get("factor_storage_entries_INFOG9"),
            )
            if base
            else None
        )
        row["blr_flops_ratio"] = ratio(
            row.get("blr_flops_printed"), row.get("flops_fullrank_RINFOG3")
        )
        fronts = row.get("blr_fronts")
        storage = row["blr_factor_storage_ratio"]
        row["blr_compressed"] = (
            fronts > 0
            if fronts is not None
            else (storage < 1.0 if storage is not None else None)
        )

    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "scaling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    blr_columns = [
        k
        for k in fieldnames
        if k
        in (
            "mesh",
            "ndofs",
            "ranks",
            "pc",
            "config",
            "valid",
            "issues",
            "factor_s",
            "solve_s",
            "gpcg_s",
            "pipeline_s",
            "rss_peak_max_mib",
            "factor_storage_MiB",
            "blr_fronts",
            "force_rel_l2_seq",
            "reference_rel_residual",
        )
        or k.startswith("blr_")
    ]
    with (out / "blr_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=blr_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "scaling.json").write_text(
        json.dumps(
            {"summary": rows, "runs": records, "settings": settings or {}}, indent=2
        )
    )
    return rows


def plot_scaling(rows, filename):
    """Write all figures beside filename. Import neither MPI nor DOLFINx."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: CSV/JSON summaries were still generated.")
        return

    out = Path(filename).parent
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["ranks"])
    p = np.array([r["ranks"] for r in rows])
    r = rows[0]
    mesh = " × ".join(str(r[k]) for k in ("mesh_nx", "mesh_ny", "mesh_nz"))
    factor = r"$LDL^T$" if r["pc"] == "cholesky" else "LU"
    nmin, nmax = min(x["repeats"] for x in rows), max(x["repeats"] for x in rows)
    repeats = str(nmin) if nmin == nmax else f"{nmin}–{nmax}"
    subtitle = (
        f"Mesh: {mesh} cells · cube side {r['length']:g} m · "
        f"{r['ndofs']:,} DOFs · {r['nc']:,} contact nodes\n"
        f"{r['backend']} / {factor} · {r.get('config', 'no compression')} · "
        f"{repeats} repeats"
    )
    invalid = [str(x["ranks"]) for x in rows if not x["valid"]]
    warning = (
        "  |  FAILED validation at ranks: " + ", ".join(invalid) if invalid else ""
    )
    colors = [
        "#2563a6",
        "#dc8624",
        "#16816a",
        "#9257a4",
        "#d45352",
        "#58a0bb",
        "#92734e",
        "#77818b",
    ]
    median_note = (
        "Median across runs; whiskers = min–max. Wall time = maximum over MPI ranks."
    )
    stack_note = "Means across runs; all components use the rank with the longest pipeline in each run."

    def values(key):
        return np.array(
            [np.nan if x.get(key) is None else x[key] for x in rows], dtype=float
        )

    def line(ax, key, label, color, scale=1.0):
        y = values(key) * scale
        lo, hi = values(key + "_min") * scale, values(key + "_max") * scale
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            ax.errorbar(
                p,
                y,
                yerr=np.maximum(0.0, [y - lo, hi - y]),
                fmt="o-",
                color=color,
                capsize=4,
                lw=1.8,
                ms=5,
                label=label,
            )
        else:
            ax.plot(p, y, "o-", color=color, lw=1.8, ms=5, label=label)
        failed = np.array([not x["valid"] for x in rows])
        if np.any(failed):
            ax.scatter(
                p[failed], y[failed], marker="x", color="#b91c1c", s=80, zorder=8
            )

    def axes_style(ax, title, ylabel):
        ax.set(title=title, xlabel="MPI ranks", ylabel=ylabel)
        ax.set_xticks(p)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        if len(p) == 1:
            ax.set_xlim(p[0] - 0.5, p[0] + 0.5)

    def save(fig, name, title, note, legend=None):
        foot = note + warning
        # Wrap long notes so they stay within the figure width.
        import textwrap

        fig.text(
            0.5,
            0.018,
            "\n".join(textwrap.wrap(foot, 125)),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
        )
        bottom = 0.13
        if legend is not None:
            handles, labels = legend
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.075),
                ncol=min(4, len(labels)),
                frameon=False,
                fontsize=9,
            )
            bottom = 0.24
        fig.tight_layout(rect=(0.01, bottom, 0.99, 0.82))
        fig.suptitle(title + "\n" + subtitle, fontsize=12, y=0.98)
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=190, facecolor="white")
        plt.close(fig)

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10}):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
        for a, key, label, color in (
            (ax[0, 0], "pipeline_s", "Complete pipeline", colors[0]),
            (ax[0, 1], "factor_s", factor + " setup + factorization", colors[1]),
            (ax[1, 0], "gpcg_s", "GPCG total", colors[2]),
        ):
            line(a, key, label, color)
            axes_style(a, label, "Wall time [s]")
        if "cg_s" in rows[0]:
            line(ax[1, 0], "cg_s", "CG phase (included in GPCG)", colors[3])
            ax[1, 0].legend(fontsize=8)
        line(ax[1, 1], "matvec_s", "Fixed-work benchmark", colors[0], 1000)
        if "gpcg_matvec_mean_s" in rows[0]:
            line(ax[1, 1], "gpcg_matvec_mean_s", "During GPCG", colors[2], 1000)
        axes_style(ax[1, 1], "One compliance matvec", "Wall time [ms / matvec]")
        ax[1, 1].legend(fontsize=8)
        save(fig, "wall_time", "Wall time versus MPI ranks", median_note)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6.5))
        for key, label, color in zip(
            ("pipeline", "factor", "gpcg", "matvec"),
            ("Pipeline", factor + " setup + factorization", "GPCG", "Matvec"),
            colors,
        ):
            line(ax[0], key + "_speedup", label, color)
            line(ax[1], key + "_efficiency", label, color, 100)
        ax[0].plot(p, p, "--", color="0.5", label="Ideal")
        ax[1].axhline(100, ls="--", color="0.5", label="Ideal")
        axes_style(ax[0], "Speedup", r"$T_1 / T_p$")
        axes_style(ax[1], "Parallel efficiency", r"$100 T_1 / (p T_p)$ [%]")
        save(
            fig,
            "scaling",
            "Strong scaling",
            "Ratios of median wall times. Speedup and efficiency omitted when validation fails or the 1-rank baseline is missing.",
            ax[0].get_legend_handles_labels(),
        )

        fig, ax = plt.subplots(1, 2, figsize=(12, 6.5))
        for a, key, title, color in (
            (ax[0], "rss_peak_max_mib", "Largest process peak RSS", colors[0]),
            (ax[1], "rss_peak_sum_mib", "Sum of process peak RSS values", colors[3]),
        ):
            line(a, key, title, color)
            axes_style(a, title, "Memory [MiB]")
        save(
            fig,
            "memory",
            "Peak memory versus MPI ranks",
            "Median and min–max across runs. Peaks include runtime + factors; summed process peaks are not simultaneous total memory.",
        )

        fig, ax = plt.subplots(1, 3, figsize=(13, 6.5))
        for a, key, title, color in zip(
            ax,
            ("cg_iterations", "matvec_count", "active_set_changes"),
            ("CG iterations", "GPCG matvecs", "Active-set changes"),
            colors,
        ):
            line(a, key, title, color)
            axes_style(a, title, "Count")
        save(
            fig,
            "iterations",
            "Solver work versus MPI ranks",
            "Median and min–max across runs. GPCG matvecs exclude the separate fixed-work benchmark and VTK export.",
        )

        fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
        for a, key, title, color in zip(
            ax.flat,
            (
                "mesh_s",
                "forms_s",
                "normals_s",
                "assembly_s",
                "load_assembly_s",
                "external_solve_s",
            ),
            (
                "Mesh construction",
                "Forms / JIT",
                "Boundary normals",
                "K assembly",
                "External load assembly",
                "External load solve",
            ),
            colors,
        ):
            line(a, key, title, color)
            axes_style(a, title, "Wall time [s]")
        save(fig, "phase_times", "Setup and external-load costs", median_note)

        if not all("breakdown_total_s" in r for r in rows):
            print(
                "Detailed breakdown unavailable in this older JSON: rerun the sweep with this script to measure CG/matvec intervals."
            )
            return

        def part(*keys):
            # Older JSON has factor_s only; newer splits it into three parts.
            return sum(
                (np.nan_to_num(values("breakdown_" + k)) for k in keys),
                np.zeros(len(rows)),
            )

        def stack(ax, series, percent=False):
            x = np.arange(len(p))
            total = sum((y for _, y, _ in series), np.zeros(len(p)))
            bottom = np.zeros(len(p))
            for label, y, color in series:
                height = (
                    np.divide(100 * y, total, out=np.zeros_like(y), where=total > 0)
                    if percent
                    else y
                )
                ax.bar(
                    x,
                    height,
                    bottom=bottom,
                    label=label,
                    color=color,
                    width=0.65,
                    edgecolor="white",
                    linewidth=0.4,
                )
                bottom += height
            ax.set_xticks(x, p)
            ax.set_xlabel("MPI ranks")
            ax.set_ylabel("Share of measured total [%]" if percent else "Wall time [s]")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if percent:
                ax.set_ylim(0, 100)
            else:
                ax.set_ylim(0, max(float(total.max()) * 1.18, 1e-12))
                for i, y in enumerate(total):
                    ax.annotate(
                        f"{y:.3g} s",
                        (i, y),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                    )

        pipeline = [
            ("Mesh + forms / JIT", part("mesh_s", "forms_s"), colors[0]),
            ("Boundary normals", part("normals_s"), colors[5]),
            ("K assembly", part("assembly_s"), colors[2]),
            (
                factor + " setup + factorization",
                part("factor_s", "analysis_s", "numeric_factor_s", "factor_other_s"),
                colors[1],
            ),
            (
                "External load + solve",
                part("load_assembly_s", "external_solve_s"),
                colors[6],
            ),
            ("GPCG matvecs", part("cg_matvec_s", "other_matvec_s"), colors[3]),
            ("Other GPCG work", part("cg_other_s", "gpcg_other_s"), colors[4]),
            (
                "Other setup / synchronization",
                part("operator_other_s", "pipeline_other_s"),
                colors[7],
            ),
        ]
        fig, ax = plt.subplots(1, 2, figsize=(13, 7.5))
        stack(ax[0], pipeline)
        stack(ax[1], pipeline, percent=True)
        ax[0].set_title("Complete pipeline")
        ax[1].set_title("Pipeline composition")
        save(
            fig,
            "time_breakdown",
            "Pipeline wall-time breakdown",
            stack_note,
            ax[0].get_legend_handles_labels(),
        )

        gpcg_parts = [
            ("Matvecs inside CG", part("cg_matvec_s"), colors[3]),
            ("Other CG work", part("cg_other_s"), colors[4]),
            ("Matvecs outside CG", part("other_matvec_s"), colors[0]),
            ("Other GPCG work", part("gpcg_other_s"), colors[7]),
        ]
        mv = part("cg_matvec_s", "other_matvec_s")
        triangular = values("breakdown_triangular_solve_s")
        matvec_parts = [
            ("Triangular solves", triangular, colors[2]),
            ("Other matvec work", np.maximum(0.0, mv - triangular), colors[5]),
        ]
        fig, ax = plt.subplots(1, 2, figsize=(13, 7.5))
        stack(ax[0], gpcg_parts)
        stack(ax[1], matvec_parts)
        ax[0].set_title("GPCG total: four disjoint components")
        ax[1].set_title("Detail of GPCG matvec time")
        h1, l1 = ax[0].get_legend_handles_labels()
        h2, l2 = ax[1].get_legend_handles_labels()
        save(
            fig,
            "gpcg_breakdown",
            "CG and compliance matvec costs",
            stack_note + " The right panel is a subset of the left panel.",
            (h1 + h2, l1 + l2),
        )


# Reference categorical order (blue, orange, aqua, yellow, magenta, green,
# violet, red); a series keeps its slot in every figure.
PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
FAIL_COLOR = "#d03b3b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
LINESTYLES = {"lu": "-", "cholesky": "--"}


def plot_comparison(rows, out, thresholds, tol=None):
    """Compression comparison figures (PNG + SVG) in out/figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib unavailable: CSV/JSON summaries were still generated.")
        return []
    import textwrap

    figdir = Path(out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    tols = sorted({r["blr_tol"] for r in rows if r["compression"] == "blr"})
    style = {("none", None): 0}
    style.update({("blr", t): i + 1 for i, t in enumerate(tols)})
    pcs = sorted({r["pc"] for r in rows})
    meshes = sorted(
        {r["mesh"] for r in rows},
        key=lambda m: int(np.prod([int(n) for n in m.split("x")])),
    )
    ranks_all = sorted({r["ranks"] for r in rows})

    def num(row, key, scale=1.0):
        v = row.get(key)
        return np.nan if v is None else float(v) * scale

    def config_series(subset, include_baseline=True):
        """One series per (factorization, compression, tolerance)."""
        series = []
        for pc in pcs:
            for (comp, t), slot in sorted(style.items(), key=lambda kv: kv[1]):
                if comp == "none" and not include_baseline:
                    continue
                part = [
                    r
                    for r in subset
                    if r["pc"] == pc and r["compression"] == comp and r["blr_tol"] == t
                ]
                if part:
                    label = config_label(comp, t, pc if len(pcs) > 1 else None)
                    series.append((label, part, slot, LINESTYLES.get(pc, "-")))
        return series

    def draw(ax, series, xkey, key, scale=1.0, ylog=False, whiskers=True):
        for label, part, slot, ls in series:
            part = sorted(part, key=lambda r: r[xkey] or 0)
            x = np.array([num(r, xkey) for r in part])
            y = np.array([num(r, key, scale) for r in part])
            lo = np.array([num(r, key + "_min", scale) for r in part])
            hi = np.array([num(r, key + "_max", scale) for r in part])
            if ylog:
                y = np.where(y > 0, y, np.nan)
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            color, marker = PALETTE[slot % len(PALETTE)], MARKERS[slot % len(MARKERS)]
            if (
                whiskers
                and np.all(np.isfinite(lo[mask]))
                and np.all(np.isfinite(hi[mask]))
            ):
                err = np.maximum(0.0, [y[mask] - lo[mask], hi[mask] - y[mask]])
                if ylog:
                    err[0] = np.minimum(err[0], 0.999 * y[mask])
                ax.errorbar(
                    x[mask],
                    y[mask],
                    yerr=err,
                    color=color,
                    marker=marker,
                    ls=ls,
                    lw=2,
                    ms=7,
                    capsize=3,
                    label=label,
                )
            else:
                ax.plot(
                    x[mask],
                    y[mask],
                    color=color,
                    marker=marker,
                    ls=ls,
                    lw=2,
                    ms=7,
                    label=label,
                )
            bad = mask & np.array([not r["valid"] for r in part])
            if bad.any():
                ax.scatter(
                    x[bad],
                    y[bad],
                    marker="x",
                    color=FAIL_COLOR,
                    s=110,
                    lw=2.2,
                    zorder=9,
                )

    def style_axes(ax, title, ylabel, xlabel, xkey, xs, ylog, zero=True):
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=GRID_COLOR, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if xkey == "ranks":
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.minorticks_off()
            if len(xs) == 1:
                ax.set_xlim(xs[0] / 1.5, xs[0] * 1.5)
        elif xkey == "ndofs":
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(xs, [f"{x / 1e3:.0f}k" if x >= 1e4 else f"{x:,}" for x in xs])
        elif xkey == "blr_tol":
            ax.set_xscale("log")
        if not ax.has_data():
            ax.text(
                0.5,
                0.5,
                "no data (not recorded or no successful run)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=INK_SECONDARY,
                fontsize=9,
            )
            ax.set_yticks([])
            return
        if ylog:
            ax.set_yscale("log")
        elif zero:
            ax.set_ylim(bottom=0)

    def save(fig, axes, name, title, subtitle, note, extra_handles=()):
        handles, labels = {}, []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handles:
                    handles[l] = h
                    labels.append(l)
        handles = [handles[l] for l in labels] + list(extra_handles)
        labels = labels + [h.get_label() for h in extra_handles]
        handles.append(
            Line2D(
                [],
                [],
                color=FAIL_COLOR,
                marker="x",
                ls="",
                ms=9,
                mew=2.2,
                label="failed validation",
            )
        )
        labels.append("failed validation")
        foot = "\n".join(textwrap.wrap(note, 150))
        legend_rows = int(np.ceil(len(labels) / 4))
        height = fig.get_size_inches()[1]
        foot_h = (0.16 * (foot.count("\n") + 1) + 0.3) / height
        legend_h = (0.27 * legend_rows + 0.15) / height
        fig.text(
            0.5, 0.01, foot, ha="center", va="bottom", fontsize=8, color=INK_SECONDARY
        )
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, foot_h),
            ncol=min(4, len(labels)),
            frameon=False,
            fontsize=9,
        )
        top = 1.0 - 0.75 / height
        fig.tight_layout(rect=(0.01, foot_h + legend_h, 0.99, top))
        fig.suptitle(title + "\n" + subtitle, fontsize=12, y=0.995, va="top")
        for ext in ("png", "svg"):
            fig.savefig(figdir / f"{name}.{ext}", dpi=170, facecolor="white")
        plt.close(fig)
        written.append(name)

    def failed_note(subset):
        bad = sorted(
            {
                f"{r['config']}{'' if len(pcs) == 1 else ' ' + r['pc']} @ {r['ranks']} rank(s)"
                + (f", mesh {r['mesh']}" if len(meshes) > 1 else "")
                for r in subset
                if not r["valid"]
            }
        )
        return (
            (" FAILED/INVALID: " + "; ".join(bad) + ".")
            if bad
            else " All configurations passed validation."
        )

    def figure(
        name,
        panels,
        series,
        xkey,
        xlabel,
        xs,
        title,
        subtitle,
        note,
        ncols=None,
        extra=(),
    ):
        ncols = ncols or min(len(panels), 4)
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows + 2.6), squeeze=False
        )
        for ax, panel in zip(axes.flat, panels):
            key, ptitle, ylabel, scale, ylog = panel[:5]
            refline = panel[5] if len(panel) > 5 else None
            draw(
                ax, series, xkey, key, scale, ylog, whiskers=not key.startswith("blr_")
            )
            if refline is not None:
                ax.axhline(refline, color="0.45", ls=":", lw=1.3)
            # Ratio panels autoscale around their no-change line (may be < 0).
            style_axes(ax, ptitle, ylabel, xlabel, xkey, xs, ylog, zero=refline is None)
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        save(
            fig,
            [a for a in axes.flat if a.get_visible()],
            name,
            title,
            subtitle,
            note,
            extra,
        )

    median_note = (
        "Median over repeats, whiskers = min–max. Wall time = maximum over MPI ranks "
        "(analysis/numeric factorization: max over ranks of rank-local PETSc event times)."
    )

    def subtitle_for(subset, ranks=None):
        rr = subset[0]
        reps = sorted({r["repeats"] for r in subset})
        rep = str(reps[0]) if len(reps) == 1 else f"{reps[0]}–{reps[-1]}"
        if ranks is not None:
            mesh = f"{ranks} MPI rank(s) · meshes " + ", ".join(
                sorted({r["mesh"] for r in subset})
            )
        elif rr.get("ndofs"):
            mesh = f"mesh {rr['mesh']} cells · {rr['ndofs']:,} DOFs · {rr['nc']:,} contact nodes"
        else:
            mesh = f"mesh {rr['mesh']} cells · no successful run"
        fact = " / ".join("LDLᵀ" if p == "cholesky" else "LU" for p in pcs)
        return f"{mesh} · {rr['backend']} {fact} · {rep} repeats"

    # ---- Versus MPI ranks, one set per mesh ---------------------------------
    for mesh in meshes:
        subset = [r for r in rows if r["mesh"] == mesh]
        series = config_series(subset)
        xs = sorted({r["ranks"] for r in subset})
        sub = subtitle_for(subset)
        note = median_note + failed_note(subset)
        common = dict(xkey="ranks", xlabel="MPI ranks", xs=xs, subtitle=sub, note=note)
        figure(
            f"ranks_{mesh}_factorization",
            [
                ("assembly_s", "K assembly", "Wall time [s]", 1, False),
                (
                    "analysis_s",
                    "Symbolic analysis (MUMPS JOB=1)",
                    "Wall time [s]",
                    1,
                    False,
                ),
                (
                    "numeric_factor_s",
                    "Numerical factorization (JOB=2)",
                    "Wall time [s]",
                    1,
                    False,
                ),
                (
                    "factor_s",
                    "Setup + factorization (PCSetUp)",
                    "Wall time [s]",
                    1,
                    False,
                ),
            ],
            series,
            title="Factorization cost versus MPI ranks",
            **common,
        )
        figure(
            f"ranks_{mesh}_solve",
            [
                (
                    "solve_s",
                    "One linear solve (fixed RHS)",
                    "Wall time [ms]",
                    1e3,
                    False,
                ),
                ("matvec_s", "One Sc_nn application", "Wall time [ms]", 1e3, False),
                (
                    "external_solve_s",
                    "External-load solve",
                    "Wall time [ms]",
                    1e3,
                    False,
                ),
            ],
            series,
            title="Fixed-work solve and compliance application",
            **common,
        )
        figure(
            f"ranks_{mesh}_gpcg",
            [
                (
                    "gpcg_solve_s",
                    "Cumulative solves inside GPCG",
                    "Wall time [s]",
                    1,
                    False,
                ),
                ("gpcg_s", "GPCG total", "Wall time [s]", 1, False),
                ("pipeline_s", "Complete pipeline", "Wall time [s]", 1, False),
            ],
            series,
            title="GPCG and pipeline wall time",
            **common,
        )
        figure(
            f"ranks_{mesh}_memory",
            [
                (
                    "rss_peak_max_mib",
                    "Largest per-rank peak RSS",
                    "Memory [MiB]",
                    1,
                    False,
                ),
                (
                    "rss_peak_sum_mib",
                    "Sum of per-rank peak RSS",
                    "Memory [MiB]",
                    1,
                    False,
                ),
                (
                    "factor_storage_MiB",
                    "MUMPS factor storage, INFOG(9)",
                    "Memory [MiB]",
                    1,
                    False,
                ),
                (
                    "mem_used_sum_MB_INFOG22",
                    "MUMPS memory used, INFOG(22)",
                    "Memory [MB]",
                    1,
                    False,
                ),
            ],
            series,
            title="Peak memory versus MPI ranks",
            xkey="ranks",
            xlabel="MPI ranks",
            xs=xs,
            subtitle=sub,
            note=(
                "Process RSS peaks are measured after the pipeline, before microbenchmarks, export "
                "and reference validation; the sum of rank peaks is not a simultaneous total. "
                "INFOG(9)/INFOG(22) are solver-reported (MUMPS 5.8.x overestimates INFOG(22) "
                "under BLR)." + failed_note(subset)
            ),
        )
        figure(
            f"ranks_{mesh}_work",
            [
                ("work_count", "GPCG work count", "Count", 1, False),
                ("cg_iterations", "CG iterations", "Count", 1, False),
                ("matvec_count", "Operator applications", "Count", 1, False),
                ("active_set_changes", "Active-set changes", "Count", 1, False),
            ],
            series,
            title="GPCG work versus MPI ranks",
            **common,
        )
        blr_series = config_series(subset, include_baseline=False)
        if blr_series:
            figure(
                f"ranks_{mesh}_blr_relative",
                [
                    (
                        "blr_speedup_factor",
                        "Setup + factorization",
                        "Speedup vs uncompressed",
                        1,
                        False,
                        1.0,
                    ),
                    (
                        "blr_speedup_solve",
                        "One linear solve",
                        "Speedup vs uncompressed",
                        1,
                        False,
                        1.0,
                    ),
                    (
                        "blr_speedup_gpcg",
                        "GPCG total",
                        "Speedup vs uncompressed",
                        1,
                        False,
                        1.0,
                    ),
                    (
                        "blr_speedup_pipeline",
                        "Complete pipeline",
                        "Speedup vs uncompressed",
                        1,
                        False,
                        1.0,
                    ),
                    (
                        "blr_reduction_rss_peak_max_mib",
                        "Largest per-rank peak RSS",
                        "Reduction [%]",
                        100,
                        False,
                        0.0,
                    ),
                    (
                        "blr_reduction_rss_peak_sum_mib",
                        "Sum of per-rank peak RSS",
                        "Reduction [%]",
                        100,
                        False,
                        0.0,
                    ),
                    (
                        "blr_factor_storage_ratio",
                        "Factor storage INFOG(9)",
                        "BLR / uncompressed",
                        1,
                        False,
                        1.0,
                    ),
                    (
                        "blr_flops_ratio",
                        "Factorization flops RINFOG(14)/RINFOG(3)",
                        "Effective / full-rank",
                        1,
                        False,
                        1.0,
                    ),
                ],
                blr_series,
                title="BLR relative to uncompressed MUMPS at the same mesh and rank count",
                xkey="ranks",
                xlabel="MPI ranks",
                xs=xs,
                subtitle=sub,
                note=(
                    "Ratios of medians against the uncompressed run with the same mesh, ranks and "
                    "factorization (not strong scaling). Dotted line: no change. Flops from the "
                    "MUMPS-printed BLR statistics." + failed_note(subset)
                ),
            )

        # ---- Accuracy versus BLR tolerance ---------------------------------
        blr_rows = [r for r in subset if r["compression"] == "blr"]
        if blr_rows:
            acc_series = []
            for pc in pcs:
                for i, p in enumerate(sorted({r["ranks"] for r in blr_rows})):
                    part = [r for r in blr_rows if r["ranks"] == p and r["pc"] == pc]
                    if part:
                        label = f"{p} rank(s)" + ("" if len(pcs) == 1 else f", {pc}")
                        acc_series.append((label, part, i, LINESTYLES.get(pc, "-")))
            base_rows = [r for r in subset if r["compression"] == "none"]

            def base_level(key):
                vals = [r.get(key) for r in base_rows if r.get(key) is not None]
                return max(vals) if vals else None

            fig_panels = [
                (
                    "force_rel_l2_seq",
                    "Force error vs sequential uncompressed",
                    "rel. L2",
                    1,
                    True,
                    thresholds["accuracy_force_rtol"],
                ),
                (
                    "force_rel_l2_rank",
                    "Force error vs same-rank uncompressed",
                    "rel. L2",
                    1,
                    True,
                    thresholds["accuracy_force_rtol"],
                ),
                (
                    "gap_scaled_linf_seq",
                    "Gap error vs sequential uncompressed",
                    "max |Δg| / indentation",
                    1,
                    True,
                    thresholds["accuracy_gap_tol"],
                ),
                (
                    "gap_scaled_linf_rank",
                    "Gap error vs same-rank uncompressed",
                    "max |Δg| / indentation",
                    1,
                    True,
                    thresholds["accuracy_gap_tol"],
                ),
                (
                    "reference_rel_residual",
                    "Contact residual, ORIGINAL operator",
                    "scaled LCP residual",
                    1,
                    True,
                    thresholds["accuracy_residual_tol"],
                ),
                (
                    "gpcg_rel_residual",
                    "GPCG residual, compressed operator",
                    "scaled LCP residual",
                    1,
                    True,
                    tol,
                ),
                (
                    "linear_residual_probe",
                    "Linear solve ‖Kx − r‖/‖r‖",
                    "relative residual",
                    1,
                    True,
                    None,
                ),
                (
                    "symmetry_defect",
                    "Compliance symmetry defect",
                    "|xᵀSy − yᵀSx| / √(xᵀSx·yᵀSy)",
                    1,
                    True,
                    None,
                ),
            ]
            ncols = 4
            fig, axes = plt.subplots(
                2, ncols, figsize=(4.3 * ncols, 2 * 3.5 + 2.6), squeeze=False
            )
            for ax, (key, ptitle, ylabel, scale, ylog, thr) in zip(
                axes.flat, fig_panels
            ):
                draw(ax, acc_series, "blr_tol", key, scale, ylog)
                if thr is not None:
                    ax.axhline(thr, color=FAIL_COLOR, ls="--", lw=1.2)
                level = base_level(
                    "gpcg_rel_residual" if key == "reference_rel_residual" else key
                )
                if level is not None and level > 0:
                    ax.axhline(level, color="0.45", ls=":", lw=1.3)
                style_axes(
                    ax, ptitle, ylabel, "BLR tolerance CNTL(7)", "blr_tol", None, ylog
                )
            extra = (
                Line2D(
                    [],
                    [],
                    color=FAIL_COLOR,
                    ls="--",
                    lw=1.2,
                    label="acceptance threshold",
                ),
                Line2D(
                    [],
                    [],
                    color="0.45",
                    ls=":",
                    lw=1.3,
                    label="uncompressed level (worst rank)",
                ),
            )
            save(
                fig,
                list(axes.flat),
                f"accuracy_{mesh}",
                "BLR accuracy versus tolerance",
                subtitle_for(subset),
                "Errors are measured against uncompressed MUMPS (coordinate-matched). The original-"
                "operator residual recomputes the gaps of the BLR forces with an uncompressed "
                "factorization; the GPCG residual is convergence for the compressed operator. "
                "Thresholds are independent of CNTL(7). Zero values are not drawn on log axes."
                + failed_note(subset),
                extra,
            )

    # ---- Versus problem size at fixed ranks --------------------------------
    if len(meshes) > 1:
        for p in ranks_all:
            subset = [r for r in rows if r["ranks"] == p and r.get("ndofs")]
            if len({r["mesh"] for r in subset}) < 2:
                continue
            series = config_series(subset)
            xs = sorted({r["ndofs"] for r in subset})
            common = dict(
                xkey="ndofs",
                xlabel="Degrees of freedom",
                xs=xs,
                subtitle=subtitle_for(subset, ranks=p),
            )
            figure(
                f"size_p{p:03d}",
                [
                    ("factor_s", "Setup + factorization", "Wall time [s]", 1, True),
                    (
                        "solve_s",
                        "One linear solve (fixed RHS)",
                        "Wall time [ms]",
                        1e3,
                        True,
                    ),
                    ("matvec_s", "One Sc_nn application", "Wall time [ms]", 1e3, True),
                    (
                        "gpcg_solve_s",
                        "Cumulative solves inside GPCG",
                        "Wall time [s]",
                        1,
                        True,
                    ),
                    ("gpcg_s", "GPCG total", "Wall time [s]", 1, True),
                    ("pipeline_s", "Complete pipeline", "Wall time [s]", 1, True),
                    (
                        "rss_peak_max_mib",
                        "Largest per-rank peak RSS",
                        "Memory [MiB]",
                        1,
                        True,
                    ),
                    ("work_count", "GPCG work count", "Count", 1, False),
                ],
                series,
                title="Cost versus problem size",
                note=median_note + failed_note(subset),
                **common,
            )
            blr_series = config_series(subset, include_baseline=False)
            if blr_series:
                figure(
                    f"size_p{p:03d}_blr_relative",
                    [
                        (
                            "blr_speedup_factor",
                            "Setup + factorization",
                            "Speedup vs uncompressed",
                            1,
                            False,
                            1.0,
                        ),
                        (
                            "blr_speedup_solve",
                            "One linear solve",
                            "Speedup vs uncompressed",
                            1,
                            False,
                            1.0,
                        ),
                        (
                            "blr_speedup_pipeline",
                            "Complete pipeline",
                            "Speedup vs uncompressed",
                            1,
                            False,
                            1.0,
                        ),
                        (
                            "blr_reduction_rss_peak_max_mib",
                            "Largest per-rank peak RSS",
                            "Reduction [%]",
                            100,
                            False,
                            0.0,
                        ),
                    ],
                    blr_series,
                    title="When does BLR pay off? Relative to uncompressed at the same size and ranks",
                    note=(
                        "Ratios of medians against the uncompressed run with the same mesh, ranks and "
                        "factorization. Dotted line: no change." + failed_note(subset)
                    ),
                    **common,
                )
    return written


def plot_all(rows, out, thresholds, tol=None):
    """Comparison figures plus the original strong-scaling set per configuration."""
    written = plot_comparison(rows, out, thresholds, tol)
    configs = sorted(
        {(r["mesh"], r["pc"], r["compression"], r["blr_tol"] or 0.0) for r in rows}
    )
    for mesh, pc, comp, t in configs:
        subset = [
            r
            for r in rows
            if (r["mesh"], r["pc"], r["compression"], r["blr_tol"] or 0.0)
            == (mesh, pc, comp, t)
            and r["repeats"] > 0
        ]
        if not subset:
            continue
        # A single uncompressed configuration keeps the historic layout.
        target = (
            Path(out)
            if len(configs) == 1
            else Path(out)
            / "figures"
            / "strong_scaling"
            / f"m{mesh}_{pc}_{config_tag(comp, t or None)}"
        )
        plot_scaling(subset, target / "scaling.png")
    return written


def load_records(source):
    if source.suffix == ".jsonl":
        records = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
        return records, {}
    payload = json.loads(source.read_text())
    if isinstance(payload, dict):
        return payload.get("runs") or [], payload.get("settings") or {}
    return payload, {}


def thresholds_from(args, settings=None):
    stored = (settings or {}).get("thresholds", {})
    return {
        key: (
            getattr(args, key)
            if getattr(args, key, None) is not None
            else stored.get(key, DEFAULT_THRESHOLDS[key])
        )
        for key in DEFAULT_THRESHOLDS
    }


def replot(args):
    source = Path(args.input).resolve()
    records, settings = load_records(source)
    if not records:
        raise ValueError(
            "Expected scaling.json with full 'runs' records, or runs.jsonl."
        )
    out = Path(args.out).resolve() if args.out else source.parent
    thresholds = thresholds_from(args, settings)
    if args.revalidate or any("valid" not in r for r in records):
        records = validate(records, thresholds)
    settings = {**settings, "thresholds": thresholds}
    rows = summarize(records, out, settings)
    tol = next((r.get("tol") for r in records if r.get("tol") is not None), None)
    plot_all(rows, out, thresholds, tol)
    print_table(rows, out)
    print(f"Plots and summaries: {out}")
    return 0


def print_table(rows, out=None):
    def fmt(v, spec, scale=1.0):
        width = int(re.match(r"\d*", spec).group() or 0)
        return "--".rjust(width) if v is None else format(v * scale, spec)

    header = (
        f"{'Mesh':>10} {'p':>3} {'Fact':>4} {'Config':>16} {'Factor(s)':>9} "
        f"{'Solve(ms)':>9} {'GPCG(s)':>8} {'Pipe(s)':>8} {'RSSmax':>8} "
        f"{'xFact':>6} {'xPipe':>6} {'dRSS%':>6} {'ForceErr':>8} {'TrueRes':>8} Valid"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['mesh']:>10} {r['ranks']:>3} {r['pc'][:4]:>4} {r['config']:>16} "
            f"{fmt(r.get('factor_s'), '9.3f')} {fmt(r.get('solve_s'), '9.3f', 1e3)} "
            f"{fmt(r.get('gpcg_s'), '8.3f')} {fmt(r.get('pipeline_s'), '8.3f')} "
            f"{fmt(r.get('rss_peak_max_mib'), '8.0f')} "
            f"{fmt(r.get('blr_speedup_factor'), '6.2f')} {fmt(r.get('blr_speedup_pipeline'), '6.2f')} "
            f"{fmt(r.get('blr_reduction_rss_peak_max_mib'), '6.1f', 100)} "
            f"{fmt(r.get('force_rel_l2_seq_max'), '8.1e')} "
            f"{fmt(r.get('reference_rel_residual_max'), '8.1e')} "
            f"{'yes' if r['valid'] else 'NO'}"
            + ("" if r["valid"] or not r.get("issues") else f"  [{r['issues']}]")
        )
    lines.append(
        "xFact/xPipe/dRSS%: BLR vs uncompressed at the same mesh and ranks. "
        "ForceErr: max force error vs sequential uncompressed. "
        "TrueRes: max contact residual with the uncompressed operator."
    )
    text = "\n".join(lines)
    print("\n" + text)
    if out is not None:
        (Path(out) / "comparison_table.txt").write_text(text + "\n")


def parse_mumps_log(path):
    """BLR statistics and phase times printed by MUMPS (ICNTL(4)=2)."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None
    number = r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)"

    def grab(pattern, source, cast=float):
        match = re.search(pattern, source)
        return (
            None
            if match is None
            else cast(match.group(1).replace("D", "E").replace("d", "e"))
        )

    result = {
        "mumps_analysis_elapsed_s": grab(
            r"Elapsed time in analysis driver\s*=\s*" + number, text
        ),
        "mumps_factor_elapsed_s": grab(
            r"Elapsed time for factorization\s*=\s*" + number, text
        ),
    }
    start = text.find("Beginning of BLR statistics")
    if start >= 0:
        end = text.find("End of BLR statistics", start)
        block = text[start : end if end > 0 else None]
        result.update(
            {
                "blr_variant": grab(r"ICNTL\(36\)[^=]*=\s*(\d+)", block, int),
                "blr_dropping": grab(r"CNTL\(7\)[^=]*=\s*" + number, block),
                "blr_fronts": grab(r"Number of BLR fronts\s*=\s*(\d+)", block, int),
                "blr_fraction_pct": grab(
                    r"Fraction of factors in BLR fronts\s*=\s*" + number, block
                ),
                "blr_entries_printed": grab(r"INFOG\(35\)[^=]*=\s*" + number, block),
                "blr_flops_printed": grab(r"RINFOG\(14\)[^=]*=\s*" + number, block),
            }
        )
    return result if any(v is not None for v in result.values()) else None


def parse_meshes(args):
    if not args.meshes:
        return [list(args.mesh)]
    meshes = []
    for item in args.meshes:
        parts = item.lower().split("x")
        if len(parts) != 3 or not all(p.isdigit() and int(p) > 0 for p in parts):
            raise ValueError(f"Mesh '{item}' must be NXxNYxNZ, e.g. 50x50x5.")
        meshes.append([int(p) for p in parts])
    return meshes


def kill_group(proc):
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def sweep(args):
    if any(
        key in os.environ for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK")
    ):
        raise RuntimeError("Run 'sweep' with plain python, not inside mpirun/srun.")
    ranks = sorted(set([1] + args.ranks))
    meshes = parse_meshes(args)
    pcs = args.pcs or [args.pc]
    tols = sorted(set(args.blr_tols))
    configs = [(pc, "none", None) for pc in pcs] + [
        (pc, "blr", t) for pc in pcs for t in tols
    ]
    launcher = shlex.split(args.mpi_command)
    if not launcher or shutil.which(launcher[0]) is None:
        raise RuntimeError(f"MPI launcher not found: {args.mpi_command}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in THREAD_VARS:
        env[key] = str(args.threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONUNBUFFERED"] = "1"
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    if cores and max(ranks) * args.threads > cores:
        print(
            f"Note: requested up to {max(ranks) * args.threads} threads for {cores} visible CPUs."
        )

    script = str(Path(__file__).resolve())
    base = [sys.executable, script, "run"]
    for name in (
        "length",
        "E",
        "nu",
        "radius",
        "indentation",
        "traction_z",
        "tol",
        "max_iter",
        "backend",
        "ordering",
        "scaling",
        "blr_variant",
        "mumps_print_level",
        "matvecs",
        "batches",
        "matvec_warmup",
    ):
        base += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    for flag in ("export", "petsc_log", "blr_cb_compression"):
        if getattr(args, flag):
            base.append("--" + flag.replace("_", "-"))
    if not args.reference_check:
        base.append("--no-reference-check")
    for item in args.mumps_icntl:
        base += ["--mumps-icntl", item]

    thresholds = thresholds_from(args)
    settings = {
        "meshes": meshes,
        "ranks": ranks,
        "configs": configs,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "seed": args.seed,
        "threads": args.threads,
        "mpi_command": args.mpi_command,
        "mpi_args": args.mpi_args,
        "timeout": args.timeout,
        "thresholds": thresholds,
        "common_command": base,
    }
    (out / "sweep_settings.json").write_text(json.dumps(settings, indent=2))

    def launch(mesh, p, cfg, label, repeat, index, total):
        pc, comp, tol = cfg
        folder = (
            out
            / "runs"
            / f"m{mesh_tag(mesh)}"
            / f"{pc}_{config_tag(comp, tol)}"
            / f"p{p:03d}_{label}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        # Never accept stale output from an earlier or failed run.
        for stale in [folder / "result.json", folder / "contact.npz"] + list(
            folder.glob("error_rank*.json")
        ):
            stale.unlink(missing_ok=True)
        cmd = (
            launcher
            + shlex.split(args.mpi_args)
            + ["-n", str(p)]
            + base
            + ["--mesh", *map(str, mesh), "--pc", pc, "--compression", comp]
            + (["--blr-tol", repr(tol)] if comp == "blr" else [])
            + ["--out", str(folder)]
        )
        (folder / "command.json").write_text(json.dumps(cmd, indent=2))
        print(
            f"[{index}/{total}] {label}: mesh {mesh_tag(mesh)}, {pc}, "
            f"{config_label(comp, tol)}, {p} rank(s) x {args.threads} thread(s)",
            flush=True,
        )
        started = time.perf_counter()
        status, code = "ok", None
        with (folder / "run.log").open("w") as log:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = proc.wait(timeout=args.timeout if args.timeout > 0 else None)
            except subprocess.TimeoutExpired:
                status = "timeout"
                kill_group(proc)
                code = proc.returncode
            except KeyboardInterrupt:
                kill_group(proc)
                raise
        meta = {
            "label": label,
            "repeat": repeat,
            "directory": str(folder),
            "exit_code": code,
            "job_s": time.perf_counter() - started,
            "command": cmd,
        }
        result_path = folder / "result.json"
        if status == "ok" and code in (0, 2) and result_path.exists():
            record = json.loads(result_path.read_text())
            record.update(meta)
            record["mumps_printed"] = parse_mumps_log(folder / "run.log")
            return record
        if status == "ok":
            status = "failed"
        errors = []
        for path in sorted(folder.glob("error_rank*.json")):
            try:
                errors.append(json.loads(path.read_text())["error"])
            except (OSError, ValueError, KeyError):
                pass
        try:
            tail = (folder / "run.log").read_text(errors="replace").splitlines()[-15:]
        except OSError:
            tail = []
        print(
            f"  -> {status} (exit {code}): {errors[0] if errors else 'see log'}; "
            f"{folder / 'run.log'}",
            flush=True,
        )
        return {
            "schema_version": 3,
            "status": status,
            "mesh": list(mesh),
            "ranks": p,
            "pc": pc,
            "backend": args.backend,
            "compression": comp,
            "blr_tol": tol,
            "error": errors[0] if errors else None,
            "log_tail": tail,
            **meta,
        }

    jobs = [(m, p, c) for m in meshes for p in ranks for c in configs]
    warm = [(m, p, c, i) for (m, p, c) in jobs for i in range(args.warmup_runs)]
    with (out / "warmups.jsonl").open("w") as log:
        for k, (m, p, c, i) in enumerate(warm, 1):
            rec = launch(m, p, c, f"warmup{i + 1:02d}", 0, k, len(warm))
            log.write(
                json.dumps(
                    {
                        k: rec.get(k)
                        for k in ("status", "directory", "exit_code", "error")
                    }
                )
                + "\n"
            )

    # Reproducibly randomized order inside each repetition block.
    rng = random.Random(args.seed)
    schedule = []
    for repeat in range(1, args.repeats + 1):
        block = jobs.copy()
        rng.shuffle(block)
        schedule += [(repeat, m, p, c) for (m, p, c) in block]
    (out / "schedule.json").write_text(
        json.dumps(
            [
                {
                    "repeat": r,
                    "mesh": m,
                    "ranks": p,
                    "pc": c[0],
                    "compression": c[1],
                    "blr_tol": c[2],
                }
                for r, m, p, c in schedule
            ],
            indent=2,
        )
    )
    records = []
    with (out / "runs.jsonl").open("w") as log:
        for k, (repeat, m, p, c) in enumerate(schedule, 1):
            record = launch(m, p, c, f"run{repeat:02d}", repeat, k, len(schedule))
            records.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()

    records = validate(records, thresholds)
    rows = summarize(records, out, settings)
    plot_all(rows, out, thresholds, args.tol)
    print_table(rows, out)
    print(f"\nResults: {out / 'scaling.csv'} and {out / 'blr_comparison.csv'}")
    print("Per-run records: runs.jsonl / scaling.json; logs and commands under runs/.")
    print(
        "Timings are medians of maximum-rank wall times. Microbenchmarks, export and the\n"
        "uncompressed reference validation are excluded from pipeline_s and peak RSS.\n"
        "Summed RSS peaks are a sum of individual rank peaks, not simultaneous usage."
    )
    print(f"Figures: {out / 'figures'} (PNG + SVG).")
    return 0 if all(r["valid"] for r in records) else 2


def add_threshold_args(s, defaults=True):
    d = DEFAULT_THRESHOLDS if defaults else {k: None for k in DEFAULT_THRESHOLDS}
    s.add_argument(
        "--check-rtol",
        type=float,
        default=d["check_rtol"],
        help="Uncompressed runs: force-relative and gap-scaled agreement with the sequential run",
    )
    s.add_argument(
        "--accuracy-force-rtol",
        type=float,
        default=d["accuracy_force_rtol"],
        help="BLR runs: max relative L2 force error vs uncompressed (independent of --blr-tol)",
    )
    s.add_argument(
        "--accuracy-gap-tol",
        type=float,
        default=d["accuracy_gap_tol"],
        help="BLR runs: max gap error / indentation vs uncompressed",
    )
    s.add_argument(
        "--accuracy-residual-tol",
        type=float,
        default=d["accuracy_residual_tol"],
        help="BLR runs: max scaled LCP residual recomputed with the uncompressed operator",
    )


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)
    plot = sub.add_parser(
        "plot", help="Regenerate summaries and figures without MPI/FEniCSx"
    )
    plot.add_argument(
        "--input",
        required=True,
        help="scaling.json with full run records (any schema), or runs.jsonl",
    )
    plot.add_argument(
        "--out", default=None, help="Output directory; defaults to the input directory"
    )
    plot.add_argument(
        "--revalidate",
        action="store_true",
        help="Recompute validation (needs the runs' contact.npz files)",
    )
    add_threshold_args(plot, defaults=False)
    for name in ("run", "sweep"):
        s = sub.add_parser(
            name,
            help=(
                "One MPI simulation"
                if name == "run"
                else "Repeated compression / strong-scaling comparison"
            ),
        )
        s.add_argument(
            "--mesh", nargs=3, type=int, default=[50, 50, 5], metavar=("NX", "NY", "NZ")
        )
        s.add_argument("--length", type=float, default=1.0)
        s.add_argument("--E", type=float, default=1e9)
        s.add_argument("--nu", type=float, default=0.3)
        s.add_argument(
            "--radius",
            type=float,
            default=2.0,
            help="Parabolic indenter curvature radius [m]",
        )
        s.add_argument(
            "--indentation", type=float, default=0.002, help="Rigid indentation [m]"
        )
        s.add_argument(
            "--traction-z",
            type=float,
            default=0.0,
            help="Upward traction on Gamma_t [Pa]",
        )
        s.add_argument(
            "--tol",
            type=float,
            default=1e-9,
            help="Dimensionless LCP relative tolerance",
        )
        s.add_argument("--max-iter", type=int, default=10000)
        s.add_argument("--pc", choices=("lu", "cholesky"), default="lu")
        s.add_argument("--backend", default="mumps")
        s.add_argument(
            "--ordering",
            choices=tuple(ORDERINGS),
            default="metis",
            help="MUMPS ordering, common to all configurations (ICNTL(7)/(28)/(29))",
        )
        s.add_argument(
            "--scaling",
            type=int,
            default=7,
            help="MUMPS ICNTL(8), common to all configurations (7 works centralized and distributed)",
        )
        s.add_argument(
            "--blr-variant",
            type=int,
            choices=(0, 1),
            default=0,
            help="ICNTL(36): 0=UFSC (MUMPS 5.8 default), 1=UCFS",
        )
        s.add_argument(
            "--blr-cb-compression",
            action="store_true",
            help="ICNTL(37)=1: also compress contribution blocks",
        )
        s.add_argument(
            "--mumps-print-level",
            type=int,
            default=2,
            help="ICNTL(4) during analysis/factorization only (2 prints the BLR statistics)",
        )
        s.add_argument(
            "--mumps-icntl",
            action="append",
            default=[],
            metavar="I=V",
            help="Extra MUMPS ICNTL applied to every configuration, e.g. 14=40",
        )
        s.add_argument(
            "--reference-check",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="BLR: recompute gaps/residual with an uncompressed factorization after all measurements",
        )
        s.add_argument(
            "--matvecs",
            type=int,
            default=20,
            help="Operator applications / solves in each timing batch",
        )
        s.add_argument("--batches", type=int, default=3)
        s.add_argument("--matvec-warmup", type=int, default=2)
        s.add_argument(
            "--export",
            action="store_true",
            help="Write solution.pvd outside timed pipeline",
        )
        s.add_argument(
            "--petsc-log",
            action="store_true",
            help="Write the PETSc log view (event logging itself is always on)",
        )
        s.add_argument(
            "--out",
            default=(
                "results/matfree_run" if name == "run" else "results/matfree_scaling"
            ),
        )
        if name == "run":
            s.add_argument("--compression", choices=("none", "blr"), default="none")
            s.add_argument(
                "--blr-tol",
                type=float,
                default=1e-6,
                help="MUMPS CNTL(7) absolute dropping parameter (with --compression blr)",
            )
        if name == "sweep":
            s.add_argument(
                "--meshes",
                nargs="+",
                default=None,
                metavar="NXxNYxNZ",
                help="Several meshes, e.g. 40x40x10 60x60x15; overrides --mesh",
            )
            s.add_argument(
                "--pcs",
                nargs="+",
                choices=("lu", "cholesky"),
                default=None,
                help="Factorization types to compare; defaults to --pc",
            )
            s.add_argument(
                "--blr-tols",
                nargs="*",
                type=float,
                default=[1e-8, 1e-6, 1e-4],
                help="BLR tolerances compared with the uncompressed baseline (none: baseline only)",
            )
            s.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8])
            s.add_argument("--repeats", type=int, default=3)
            s.add_argument(
                "--warmup-runs",
                type=int,
                default=1,
                help="Discarded full runs per configuration (warms JIT and file caches)",
            )
            s.add_argument("--seed", type=int, default=17, help="Execution-order seed")
            s.add_argument("--threads", type=int, default=1)
            s.add_argument("--mpi-command", default="mpirun")
            s.add_argument(
                "--mpi-args",
                default="",
                help="Launcher arguments, e.g. --mpi-args='--bind-to core'",
            )
            s.add_argument(
                "--timeout",
                type=float,
                default=0,
                help="Timeout per job in seconds; 0 disables",
            )
            add_threshold_args(s)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.mode == "plot":
        return replot(args)
    if (
        min(args.mesh) < 1
        or args.length <= 0
        or args.E <= 0
        or not -1 < args.nu < 0.5
        or args.radius <= 0
        or args.indentation <= 0
        or args.tol <= 0
        or args.max_iter < 1
        or args.matvecs < 1
        or args.batches < 1
        or args.matvec_warmup < 0
    ):
        p.error(
            "Mesh counts, dimensions, modulus, radius, indentation and iteration counts must be valid positive values; -1 < nu < 0.5."
        )
    if any(not re.fullmatch(r"\d+=-?\d+", item) for item in args.mumps_icntl):
        p.error("--mumps-icntl expects I=V with integers, e.g. 14=40.")
    if args.mumps_icntl and args.backend != "mumps":
        p.error("--mumps-icntl requires --backend mumps.")
    if args.mode == "sweep":
        if (
            min(args.ranks) < 1
            or args.repeats < 1
            or args.threads < 1
            or args.warmup_runs < 0
            or args.check_rtol <= 0
            or args.accuracy_force_rtol <= 0
            or args.accuracy_gap_tol <= 0
            or args.accuracy_residual_tol <= 0
            or any(t <= 0 for t in args.blr_tols)
        ):
            p.error("Invalid scaling parameters.")
        if args.blr_tols and args.backend != "mumps":
            p.error("BLR compression requires --backend mumps.")
        try:
            parse_meshes(args)
        except ValueError as exc:
            p.error(str(exc))
        return sweep(args)
    if args.compression == "blr" and (args.blr_tol <= 0 or args.backend != "mumps"):
        p.error("--compression blr needs --backend mumps and --blr-tol > 0.")
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

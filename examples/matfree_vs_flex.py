from pathlib import Path
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm

from dolfinx.mesh import create_box, locate_entities_boundary, meshtags, CellType
from dolfinx.io import VTKFile
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
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
from dolfinx.mesh import entities_to_geometry

from cofebem.contact.rigid_indenters import parabolic
from cofebem.lcp.solve import solve
from cofebem.lcp.problem import LCP


class ScNormalMatvec:

    def __init__(self, A, Ic, nrm, tdim=2):
        self.A = A
        self.comm = A.getComm()
        self.tdim = int(tdim)

        self.Ic = np.asarray(Ic, dtype=np.int64)
        self.nrm = np.asarray(nrm, dtype=float)
        self.nc = int(self.Ic.size)

        if self.nrm.shape != (self.nc, self.tdim):
            raise ValueError(
                f"nrm must have shape ({self.nc}, {self.tdim}), "
                f"got {self.nrm.shape}"
            )

        m = np.linalg.norm(self.nrm, axis=1)
        if np.any(m == 0.0):
            raise ValueError("Zero normal detected.")

        self.n = self.nrm / m[:, None]

        # Build KSP once.
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.setFromOptions()
        self.ksp.setUp()

        self.rhs = A.createVecRight()
        self.u = A.createVecRight()

        # Vector dofs associated with contact scalar dofs.
        self.cd = np.stack(
            [self.Ic * self.tdim + c for c in range(self.tdim)],
            axis=1,
        ).astype(np.int32)

        self.cdf = self.cd.reshape(-1).astype(np.int32)

        # Scatter solution values at contact vector dofs into a sequential vector.
        self.uc = PETSc.Vec().createSeq(
            self.tdim * self.nc,
            comm=PETSc.COMM_SELF,
        )

        self.isf = PETSc.IS().createGeneral(self.cdf, comm=self.comm)
        self.ist = PETSc.IS().createStride(
            self.tdim * self.nc,
            first=0,
            step=1,
            comm=PETSc.COMM_SELF,
        )

        self.scatter = PETSc.Scatter().create(
            self.u,
            self.isf,
            self.uc,
            self.ist,
        )

        self.n_matvec = 0

    def __call__(self, p):
        # TODO : Measure the overhead from the embedding of the vector
        p = np.asarray(p, dtype=np.float64).reshape(-1)

        if p.size != self.nc:
            raise ValueError(f"Expected p of size {self.nc}, got {p.size}")

        self.rhs.set(0.0)

        values = (p[:, None] * self.n).reshape(-1)
        values = np.asarray(values, dtype=PETSc.ScalarType)

        self.rhs.setValues(
            self.cdf,
            values,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        self.rhs.assemble()

        self.ksp.solve(self.rhs, self.u)

        self.uc.set(0.0)
        self.scatter.scatter(
            self.u,
            self.uc,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        uc_array = self.uc.getArray(readonly=True).copy()
        uc_array = uc_array.reshape(self.nc, self.tdim)

        y = np.einsum("ij,ij->i", self.n, uc_array)

        self.n_matvec += 1

        return y


def gpcg_matrix_free(
    apply_M,
    q,
    tol=1e-10,
    max_iter=10000,
    z0=None,
):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = q.size

    eps = np.finfo(np.float64).eps

    if z0 is None:
        z = np.zeros(n, dtype=np.float64)
    else:
        z = np.maximum(np.asarray(z0, dtype=np.float64).reshape(-1), 0.0)

    matvec_count = 0

    def Mv(x):
        nonlocal matvec_count
        y = np.asarray(apply_M(x), dtype=np.float64).reshape(-1)
        if y.size != n:
            raise ValueError(f"apply_M returned size {y.size}, expected {n}")
        matvec_count += 1
        return y

    # Initial complementarity variable
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

            # Release the most violated active constraint.
            j = violated[np.argmin(w[violated])]
            active[j] = False

            active_set_changes += 1
            work_count += 1

            continue

        d = r.copy()
        rr = float(r @ r)

        # Inner CG loop on the current free face.
        while work_count < max_iter:
            d[active] = 0.0

            Md = Mv(d)
            curvature = float(d @ Md)

            if not np.isfinite(curvature) or curvature <= 0.0:
                numerical_breakdown = True
                break

            alpha_cg = rr / curvature

            decreasing = d < 0.0
            if np.any(decreasing):
                alpha_max = float(np.min(-z[decreasing] / d[decreasing]))
            else:
                alpha_max = np.inf

            alpha = min(alpha_cg, alpha_max)

            if alpha < 0.0 or not np.isfinite(alpha):
                numerical_breakdown = True
                break

            z = z + alpha * d

            # Since w = M z + q and M is linear:
            # avoid an extra expensive solve by updating w incrementally.
            w = w + alpha * Md

            zero_tol = (
                1000.0
                * eps
                * max(
                    1.0,
                    float(np.linalg.norm(z, ord=np.inf)),
                )
            )

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

                # Feasible face changed; restart CG.
                break

            free = ~active

            r_new = np.zeros_like(z)
            r_new[free] = -w[free]

            scale = 1.0 + np.linalg.norm(z, ord=np.inf) + np.linalg.norm(w, ord=np.inf)

            if np.linalg.norm(r_new, ord=np.inf) <= tol * scale:
                # Current face minimized; outer loop checks active release.
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

    info = {
        "converged": converged,
        "numerical_breakdown": numerical_breakdown,
        "cg_iterations": cg_iterations,
        "active_set_changes": active_set_changes,
        "work_count": work_count,
        "matvec_count": matvec_count,
        "residual": residual,
    }

    return z, w, info


def Sc_n(A, Ic, nrm, tdim=3, f0=1e9, show=True):
    comm = MPI.COMM_WORLD
    f0 = float(f0)
    tdim = int(tdim)
    Ic = np.asarray(Ic, dtype=np.int64)
    nrm = np.asarray(nrm, dtype=float)
    nc = int(Ic.size)

    m = np.linalg.norm(nrm, axis=1)
    n = nrm / m[:, None]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()
    ksp.setUp()

    rhs = A.createVecRight()
    u = A.createVecRight()

    cd = np.stack([Ic * tdim + c for c in range(tdim)], axis=1).astype(np.int32)
    cdf = cd.reshape(-1).astype(np.int32)

    uc = PETSc.Vec().createSeq(tdim * nc, comm=PETSc.COMM_SELF)
    isf = PETSc.IS().createGeneral(cdf, comm=comm)
    ist = PETSc.IS().createStride(tdim * nc, first=0, step=1, comm=PETSc.COMM_SELF)
    sca = PETSc.Scatter().create(u, isf, uc, ist)

    S = np.zeros((nc, nc), dtype=float)

    it = range(nc)
    if show:
        it = tqdm(it, desc="Sampling Snn", unit="col")

    for j in it:
        rhs.set(0.0)
        rhs.setValues(cd[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
        rhs.assemble()

        ksp.solve(rhs, u)

        uc.set(0.0)
        sca.scatter(
            u,
            uc,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

        ux = uc.getArray(readonly=True).copy().reshape(nc, tdim)
        S[:, j] = np.einsum("ij,ij->i", n, ux) / f0

    return S


def uniq_v(mesh, fs):
    fdim = mesh.topology.dim - 1
    fs = np.asarray(fs, dtype=np.int32)
    fg = entities_to_geometry(mesh, fdim, fs)
    v = np.unique(np.asarray(fg, dtype=np.int32).ravel())
    return np.sort(v).astype(np.int32)


def nrm_bnd(mesh, Iv, ds_c, eps=1e-8, save=False):
    gdim = mesh.geometry.dim
    Vn = functionspace(mesh, ("CG", 1, (gdim,)))

    n = FacetNormal(mesh)
    u = TrialFunction(Vn)
    v = TestFunction(Vn)

    a = eps * inner(u, v) * dx + inner(u, v) * ds_c
    L = inner(n, v) * ds_c

    nf = Function(Vn)
    nf.name = "n"

    LinearProblem(
        a=a,
        L=L,
        u=nf,
        bcs=[],
        petsc_options_prefix="nrm_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()

    nf.x.scatter_forward()

    if save:
        with VTKFile(mesh.comm, "n.pvd", "w") as vtk:
            vtk.write_function(nf)

    Iv = np.asarray(Iv, dtype=np.int32)
    nrm = np.zeros((Iv.size, gdim), dtype=float)

    for c in range(gdim):
        Vc, mp = Vn.sub(c).collapse()
        mp = np.asarray(mp, dtype=np.int32)
        ds = locate_dofs_topological(Vc, 0, Iv)
        ds = np.asarray(ds, dtype=np.int32)
        pd = mp[ds]
        nrm[:, c] = nf.x.array[pd]

    m = np.linalg.norm(nrm, axis=1)
    nrm /= m[:, None]

    return nrm, nf


def sys(mesh, E=1.0e9, nu=0.3, gu=None):
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

    f = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))

    a = inner(sig(u), eps(v)) * dx
    L = inner(f, v) * dx

    Gu = locate_entities_boundary(mesh, fdim, gu)
    Gd = locate_dofs_topological(V, fdim, Gu)

    u0 = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gd, V)
    bcs = [bc]

    pb = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="sc_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    pb._A.zeroEntries()
    assemble_matrix(pb._A, pb._a, bcs=pb.bcs)
    pb._A.assemble()

    with pb._b.localForm() as b0:
        b0.set(0.0)

    assemble_vector(pb._b, pb._L)
    apply_lifting(pb._b, [pb._a], bcs=[pb.bcs])
    pb._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    for bc in pb.bcs:
        bc.set(pb._b.array_w)

    return V, pb._A, pb._b, pb


if __name__ == "__main__":
    Nx, Ny, Nz = 50, 50, 5
    L = 1.0
    mesh = create_box(
        MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, L, L]], [Nx, Ny, Nz], CellType.hexahedron
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1

    E, nu = 1.0e9, 0.3

    def Gamma_u_locator(x):
        return np.isclose(x[2], 0, 1.0e-6)

    V, A, b, pb = sys(mesh, E, nu, Gamma_u_locator)

    def Gamma_c_locator(x):
        return np.isclose(x[2], L)

    Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
    Ic = locate_dofs_topological(V, fdim, Gamma_c)
    Gamma_c_x = mesh.geometry.x[Ic]
    Gamma_c_id = 1
    Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)

    facet_indices = np.hstack([Gamma_c]).astype(np.int32)
    facet_values = np.hstack(
        [
            Gamma_c_tags,
        ]
    ).astype(np.int32)

    order = np.argsort(facet_indices)
    facet_indices = facet_indices[order]
    facet_values = facet_values[order]

    mt = meshtags(mesh, fdim, facet_indices, facet_values)

    ds = Measure("ds", domain=mesh, subdomain_data=mt)

    ds_c = ds(Gamma_c_id)
    nrm, _ = nrm_bnd(mesh, Ic, ds_c)

    Rindenter = 0.3
    displ = 0.2
    x_positions = np.linspace(0.0, L, 10)
    contact_center_y = L / 2

    def indenter_gap(contact_center_x):
        return (
            parabolic(
                Gamma_c_x[:, 0],
                Gamma_c_x[:, 1],
                contact_center_x,
                contact_center_y,
                Rindenter,
                np.ones_like(Gamma_c_x[:, 2]) - displ,
            )
            - Gamma_c_x[:, 2]
        )

    # Whole candidate contact boundary Gamma_c
    Ic_gamma = np.asarray(Ic, dtype=np.int64)
    nrm_gamma = np.asarray(nrm, dtype=np.float64)

    assert nrm_gamma.shape[0] == Ic_gamma.size

    # Flexibility: include construction of the dense Sc matrix in the total.
    flex_start = time.process_time()
    sc_start = time.process_time()
    Sc = Sc_n(A, Ic_gamma, nrm_gamma, tdim)
    sc_construction_time = time.process_time() - sc_start

    flex_solve_start = time.process_time()
    fc_flex = np.zeros(Ic_gamma.size, dtype=np.float64)
    flex_forces = []
    flex_results = []
    for contact_center_x in x_positions:
        q_gamma = np.asarray(indenter_gap(contact_center_x), dtype=np.float64).reshape(
            -1
        )
        lcp_result = solve(
            LCP(Sc, q_gamma),
            "ccg_v2",
            z0=fc_flex,
            tol=1e-8,
            max_iter=1000,
        )
        fc_flex = lcp_result.z
        flex_forces.append(fc_flex.copy())
        flex_results.append(lcp_result)

    flex_solve_time = time.process_time() - flex_solve_start
    flex_total_time = time.process_time() - flex_start

    # MatFree: include construction and factorization of its operator in the
    # total, followed by the same ten loading steps.
    matfree_start = time.process_time()
    matfree_operator_start = time.process_time()
    apply_Sc = ScNormalMatvec(
        A=A,
        Ic=Ic_gamma,
        nrm=nrm_gamma,
        tdim=mesh.topology.dim,
    )
    matfree_operator_time = time.process_time() - matfree_operator_start

    matfree_solve_start = time.process_time()
    fc_matfree = np.zeros(Ic_gamma.size, dtype=np.float64)
    matfree_forces = []
    matfree_results = []
    matvecs_per_step = []
    for contact_center_x in x_positions:
        q_gamma = np.asarray(indenter_gap(contact_center_x), dtype=np.float64).reshape(
            -1
        )
        previous_matvec_count = apply_Sc.n_matvec
        fc_matfree, _, info = gpcg_matrix_free(
            apply_M=apply_Sc,
            q=q_gamma,
            z0=fc_matfree,
            tol=1e-8,
            max_iter=1000,
        )
        matfree_forces.append(fc_matfree.copy())
        matfree_results.append(info)
        matvecs_per_step.append(apply_Sc.n_matvec - previous_matvec_count)

    matfree_solve_time = time.process_time() - matfree_solve_start
    matfree_total_time = time.process_time() - matfree_start

    print("\nLoading-step convergence:")
    for step, contact_center_x in enumerate(x_positions):
        print(
            f"  step {step + 1:02d}, x={contact_center_x:.6f}: "
            f"Flexibility={flex_results[step].converged}, "
            f"MatFree={matfree_results[step]['converged']}, "
            f"MatFree matvecs={matvecs_per_step[step]}"
        )

    print("\nCPU-time comparison (visualization excluded):")
    print(
        f"  Flexibility: {flex_total_time:.6f} s total "
        f"(Sc construction {sc_construction_time:.6f} s + "
        f"10 solves {flex_solve_time:.6f} s)"
    )
    print(
        f"  MatFree:     {matfree_total_time:.6f} s total "
        f"(operator construction {matfree_operator_time:.6f} s + "
        f"10 solves {matfree_solve_time:.6f} s)"
    )
    if flex_total_time >= matfree_total_time:
        print(
            f"  MatFree is {flex_total_time / matfree_total_time:.3f}x "
            "faster overall."
        )
    else:
        print(
            f"  Flexibility is {matfree_total_time / flex_total_time:.3f}x "
            "faster overall."
        )
    print(f"  Total MatFree Sc matvecs: {apply_Sc.n_matvec}")

    # Visualization is deliberately performed after the timing measurements.
    solver_petsc = PETSc.KSP().create(mesh.comm)
    solver_petsc.setOperators(A)
    solver_petsc.setType("preonly")
    solver_petsc.getPC().setType("lu")
    solver_petsc.setFromOptions()
    solver_petsc.setUp()

    contact_vector_dofs = np.stack(
        [Ic_gamma * tdim + component for component in range(tdim)],
        axis=1,
    ).astype(np.int32)

    output_root = Path("./results")
    if mesh.comm.rank == 0:
        (output_root / "flexibility").mkdir(parents=True, exist_ok=True)
        (output_root / "matfree").mkdir(parents=True, exist_ok=True)
    mesh.comm.barrier()

    def write_visualization(path, force_history):
        u_fenics = Function(V)
        u_fenics.name = "u"

        f_fenics = Function(V)
        f_fenics.name = "fc"

        b_ = b.copy()
        u_ = A.createVecRight()

        with VTKFile(mesh.comm, str(path), "w") as vtk:
            vtk.write_mesh(mesh)
            for contact_center_x, contact_force in zip(
                x_positions, force_history, strict=True
            ):
                b_.set(0.0)
                force_values = (-contact_force[:, None] * nrm_gamma).reshape(-1)
                b_.setValues(
                    contact_vector_dofs.reshape(-1),
                    np.asarray(force_values, dtype=PETSc.ScalarType),
                    addv=PETSc.InsertMode.INSERT_VALUES,
                )
                b_.assemble()

                solver_petsc.solve(b_, u_)

                u_fenics.x.array[:] = u_.array
                u_fenics.x.scatter_forward()

                f_fenics.x.array[:] = b_.array
                f_fenics.x.scatter_forward()

                vtk.write_function([u_fenics, f_fenics], float(contact_center_x))

    write_visualization(output_root / "flexibility" / "cube.pvd", flex_forces)
    write_visualization(output_root / "matfree" / "cube.pvd", matfree_forces)

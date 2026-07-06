import argparse
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a 3D linear-elasticity problem")

    parser.add_argument(
        "--solver",
        type=str,
        choices=["lu", "cg", "both"],
        default="both",
        help="Solver to use: lu, cg, or both",
    )

    parser.add_argument(
        "--Nr",
        type=int,
        default=10,
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

    args = parser.parse_args()

    solver = args.solver
    Nr, Nt, Np = args.Nr, args.Nt, args.Np
    n_solves = args.nsol

    print(f"Grid size: Nr={Nr}, Nt={Nt}, Np={Np}")
    print(f"Solver: {solver}")
    print(f"Number of solves: {n_solves}")

    # ------------------helpers----------------

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

    # ---- parameters ----
    a0 = 0.20
    b0 = 0.10
    thickness = 0.03
    ox = 0.0
    oz = 0.5
    theta_cut = np.pi / 6

    tyre_hex_mesh(
        a0, b0, thickness, ox, oz, nr=Nr, ntt=Nt, npp=Np, filename="tyre_hex.xdmf"
    )
    with XDMFFile(MPI.COMM_WORLD, "tyre_hex.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    tdim = mesh.topology.dim
    fdim = tdim - 1

    E = 2.5e8
    nu = 0.48

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    xL = ox - a0 * np.cos(theta_cut)
    xR = ox + a0 * np.cos(theta_cut)

    xtol = 1e-3
    ftol = 5e-3
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

        a_ref = a0 + 0.5 * thickness
        b_ref = b0 + 0.5 * thickness

        theta = np.arctan2((r - oz) / b_ref, (X - ox) / a_ref)

        theta1 = -np.pi + theta_cut
        theta2 = -theta_cut

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

        F = ((X - ox) / a0) ** 2 + ((r - oz) / b0) ** 2 - 1.0
        on_inner = np.abs(F) < ftol

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

        aout = a0 + thickness
        bout = b0 + thickness
        F = ((X - ox) / aout) ** 2 + ((r - oz) / bout) ** 2 - 1.0
        on_outer = np.abs(F) < ftol

        return on_outer

    Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
    Gamma_c = np.array([f for f in Gamma_c if f not in Gamma_u_set], dtype=np.int32)
    Gamma_c_id = 2
    Gamma_c_tags = np.full(Gamma_c.shape, Gamma_c_id, dtype=np.int32)
    Gamma_c_dofs = locate_dofs_topological(V, fdim, Gamma_c)

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

    print(f"Total number of nodes: {num_nodes}")
    print(f"Total number of DOFs: {num_dofs}")

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

    n_solves = 10

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

    x = problem.A.createVecRight()

    # ============================================================
    # LU solver
    # ============================================================

    if solver in ["lu", "both"]:
        print("======================LU====================")
        print("Solving with LU...")

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

        for _ in range(n_solves):
            x.set(0.0)

            t0 = time.time()
            lu_solver.solve(problem.b, x)
            solve_time = time.time() - t0

            lu_times.append(solve_time)

        lu_times = np.array(lu_times)
        lu_time = lu_times.mean()

        print(f"LU Factorisation CPU time = {factor_time}")
        print(f"LU average solve CPU time = {lu_time}")
        print(f"LU total average time = {factor_time + lu_time}")
        print()

    # ============================================================
    # CG + GAMG solver
    # ============================================================

    if solver in ["cg", "both"]:
        print("======================CG + GAMG===============")
        print("Solving with CG + GAMG...")

        cg_solver = PETSc.KSP().create(MPI.COMM_WORLD)
        cg_solver.setOperators(problem.A)

        cg_solver.setType("cg")

        pc = cg_solver.getPC()
        pc.setType("gamg")

        cg_solver.setTolerances(
            rtol=1e-8,
            atol=1e-12,
            max_it=10000,
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

        cg_times = np.array(cg_times)
        cg_time = cg_times.mean()

        print(f"CG + GAMG PC setup CPU time = {pc_setup_time}")
        print(f"CG + GAMG average solve CPU time = {cg_time}")
        print(f"CG + GAMG total average time = {pc_setup_time + cg_time}")
        print(f"CG + GAMG average iterations = {np.mean(cg_iterations)}")
        print(f"CG + GAMG last convergence reason = {cg_reasons[-1]}")
        print(f"CG + GAMG last residual norm = {cg_residuals[-1]}")
        print()

    print("All requested solves completed.")


if __name__ == "__main__":
    main()

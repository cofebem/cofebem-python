import time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import locate_entities_boundary, create_box, CellType
from dolfinx.fem import Constant, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import (
    assemble_matrix_mat,
    assemble_vector,
    apply_lifting,
    LinearProblem,
)
from ufl import Identity, TrialFunction, TestFunction, sym, grad, inner, tr, dx


# ---------- IMPORTANT: clear PETSc options database (fixes your error 75) ----------
PETSc.Options().clear()


def epsilon(w):
    return sym(grad(w))


def build_system(mesh, E=1.0e9, nu=0.3):
    tdim = mesh.topology.dim
    fdim = tdim - 1

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    V = functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(tdim) + 2 * mu * epsilon(w)

    f_v = Constant(mesh, np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType))
    a = inner(sigma(u), epsilon(v)) * dx
    Lform = inner(f_v, v) * dx

    # clamp z=0
    def Gamma_u_locator(x):
        return np.isclose(x[2], 0.0)

    Gamma_u = locate_entities_boundary(mesh, fdim, Gamma_u_locator)
    Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)
    bc = dirichletbc(np.array([0, 0, 0], dtype=PETSc.ScalarType), Gamma_u_dofs, V)

    # container
    problem = LinearProblem(a=a, L=Lform, bcs=[bc])

    # assemble A
    problem._A.zeroEntries()
    assemble_matrix_mat(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()

    # assemble b (not used, but keep consistent)
    with problem._b.localForm() as bl:
        bl.set(0.0)
    assemble_vector(problem._b, problem._L)
    apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
    problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for bc in problem.bcs:
        bc.set(problem._b.array_w)

    return V, problem._A, problem._b


def locate_contact_dofs_z_local(V, mesh, Lbox):
    """Local per-rank contact z-component dofs on z=Lbox."""
    tdim = mesh.topology.dim
    fdim = tdim - 1

    def Gamma_c_locator(x):
        return np.isclose(x[2], Lbox)

    Gamma_c = locate_entities_boundary(mesh, fdim, Gamma_c_locator)
    dofs_z = locate_dofs_topological((V.sub(2), V), fdim, Gamma_c)[1]
    return np.asarray(dofs_z, dtype=np.int32)


def make_global_contact_dofs(dofs_local, comm):
    """Union all local dof lists -> sorted unique -> broadcast to all ranks."""
    gathered = comm.gather(dofs_local, root=0)

    if comm.rank == 0:
        concat = (
            np.concatenate(gathered) if len(gathered) else np.array([], dtype=np.int32)
        )
        dofs_global = np.unique(concat).astype(np.int32)
        nc = np.int32(dofs_global.size)
    else:
        dofs_global = None
        nc = np.int32(0)

    nc = comm.bcast(nc, root=0)

    if comm.rank != 0:
        dofs_global = np.empty(nc, dtype=np.int32)

    comm.Bcast(dofs_global, root=0)
    return dofs_global


def setup_solver(comm, A):
    """
    Force preonly+lu and set factor solver explicitly.
    DO NOT call setFromOptions() here (it may re-introduce mpihash/matrix options).
    """
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("lu")

    # Choose your parallel LU backend:
    # Try mumps first (often available with dolfinx builds).
    # If this fails, switch to "superlu_dist".
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pc.setFactorSolverType("superlu_dist")

    ksp.setUp()

    # Debug prints
    if comm.rank == 0:
        print(f"KSP type = {ksp.getType()}  (expect preonly)")
        print(f"PC  type = {pc.getType()}   (expect lu)")
        try:
            print(f"LU solver = {pc.getFactorSolverType()}")
        except Exception:
            pass

    return ksp


def extract_values_on_rank0(sol, idx_global, A, comm):
    """
    Extract sol[idx_global] on rank 0 without Scatter.toZero() (stable).
    """
    lo, hi = A.getOwnershipRange()
    idx_global = np.asarray(idx_global, dtype=np.int64)

    mask = (idx_global >= lo) & (idx_global < hi)
    pos_loc = np.nonzero(mask)[0].astype(np.int32)
    idx_loc = idx_global[mask].astype(np.int64)

    with sol.localForm() as s:
        arr = s.array
        vals_loc = arr[(idx_loc - lo).astype(np.int64)].copy()

    counts = comm.gather(pos_loc.size, root=0)

    if comm.rank == 0:
        counts = np.asarray(counts, dtype=np.int32)
        displs = np.zeros_like(counts)
        displs[1:] = np.cumsum(counts[:-1])
        total = int(np.sum(counts))

        pos_all = np.empty(total, dtype=np.int32)
        vals_all = np.empty(total, dtype=vals_loc.dtype)
    else:
        displs = None
        pos_all = None
        vals_all = None

    comm.Gatherv(pos_loc, (pos_all, counts, displs, MPI.INT), root=0)

    mpi_dtype = MPI.DOUBLE if vals_loc.dtype == np.float64 else MPI.COMPLEX
    comm.Gatherv(vals_loc, (vals_all, counts, displs, mpi_dtype), root=0)

    if comm.rank == 0:
        out = np.empty(idx_global.size, dtype=vals_all.dtype)
        out[pos_all] = vals_all
        return out
    return None


def build_Sc_iterative_parallel(
    mesh, A, contact_dofs_z_global, load=1.0e9, max_cols=50
):
    """
    Build only max_cols columns for testing (otherwise Sc is huge).
    Columns are distributed: rank r computes j=r, r+size, ...
    """
    comm = mesh.comm
    rank = comm.rank
    size = comm.size

    # print matrix layout
    lo, hi = A.getOwnershipRange()
    if rank == 0:
        print("A global size:", A.getSize())
    print(f"[rank {rank}] owns rows [{lo},{hi})", flush=True)
    comm.Barrier()

    ksp = setup_solver(comm, A)

    rhs = A.createVecRight()
    sol = A.createVecLeft()

    idx = contact_dofs_z_global.astype(np.int32)
    nc = int(idx.size)
    ncols = min(int(max_cols), nc)

    Sc0 = np.zeros((nc, ncols), dtype=PETSc.ScalarType) if rank == 0 else None

    t_solve = 0.0
    for j in range(rank, ncols, size):
        rhs.set(0.0)
        rhs.setValue(int(idx[j]), load)
        rhs.assemble()

        t0 = time.time()
        ksp.solve(rhs, sol)
        t_solve += time.time() - t0

        col0 = extract_values_on_rank0(sol, idx, A, comm)
        if rank == 0:
            Sc0[:, j] = col0 / load
            if j % max(1, ncols // 10) == 0:
                print(f"progress: column {j}/{ncols}")

    tmax = comm.allreduce(t_solve, op=MPI.MAX)
    if rank == 0:
        print(
            f"Done: nc={nc}, built {ncols} cols, max solve time across ranks={tmax:.3f}s"
        )

    return Sc0


def main():
    comm = MPI.COMM_WORLD

    # --- mesh ---
    Lbox = 1.0
    ncells = 5
    mesh = create_box(
        comm,
        [[0.0, 0.0, 0.0], [Lbox, Lbox, Lbox]],
        [ncells * 10, ncells * 10, ncells],
        CellType.hexahedron,
    )

    print(f"[rank {comm.rank}/{comm.size}] hello", flush=True)
    comm.Barrier()

    V, A, b = build_system(mesh)

    # --- local contact dofs ---
    dofs_local = locate_contact_dofs_z_local(V, mesh, Lbox)

    counts = comm.gather(int(dofs_local.size), root=0)
    h = comm.gather(int(hash(dofs_local.tobytes())), root=0)
    first10 = comm.gather(dofs_local[:10].copy(), root=0)

    if comm.rank == 0:
        print("\n=== CONTACT DOF LOCALITY CHECK ===")
        print("Number of contact z-dofs per rank:", counts)
        print("Hash of contact z-dofs per rank:", h)
        for r in range(comm.size):
            print(f"  rank {r}: first 10 = {first10[r]}")
        print("=================================\n")

    # --- global union + broadcast ---
    dofs_global = make_global_contact_dofs(dofs_local, comm)
    if comm.rank == 0:
        print("Global nc =", dofs_global.size)

    # --- build Sc (TEST: only 50 columns) ---
    Sc0 = build_Sc_iterative_parallel(mesh, A, dofs_global, load=1.0e9, max_cols=50)

    if comm.rank == 0:
        print("Sc0 shape:", Sc0.shape)


if __name__ == "__main__":
    main()

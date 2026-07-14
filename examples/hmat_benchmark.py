import time
import csv
import numpy as np
import matplotlib.pyplot as plt

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
from dolfinx.mesh import entities_to_geometry

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

from cofebem.hmatrices.hmatrix import HMatrix


def handmade_matvec(A, x):
    m, n = A.shape
    y = np.empty(m, dtype=A.dtype)

    for i in range(m):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x[j]
        y[i] = acc

    return y


def time_average(func, repeats, *args):
    func(*args)
    acc = 0.0

    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        acc += time.perf_counter() - t0

    return acc / repeats


def Sc_n(A, Ic, nrm, tdim=3, f0=1e9, show=True):
    comm = MPI.COMM_WORLD
    f0 = float(f0)
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

    f = Constant(mesh, np.zeros(tdim, dtype=PETSc.ScalarType))

    a = inner(sig(u), eps(v)) * dx
    L = inner(f, v) * dx

    Gu = locate_entities_boundary(mesh, fdim, gu)
    Gd = locate_dofs_topological(V, fdim, Gu)

    u0 = np.zeros(tdim, dtype=PETSc.ScalarType)
    bc = dirichletbc(u0, Gd, V)

    pb = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="sc_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
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


def plot_heatmap(Z, leaf_grid, eta_grid, title, cbar_label, fname):
    plt.figure(figsize=(7.5, 5.5))
    plt.imshow(Z, origin="lower", aspect="auto")
    plt.xticks(np.arange(len(leaf_grid)), leaf_grid)
    plt.yticks(np.arange(len(eta_grid)), eta_grid)
    plt.xlabel("leaf_size")
    plt.ylabel("eta")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.tight_layout()
    # plt.savefig(fname, dpi=200)
    plt.show()


def main():
    comm = MPI.COMM_WORLD

    mesh_grid = [
        (10, 10, 5),
        (20, 20, 10),
        (25, 25, 12),
        # (40, 40, 20),
    ]

    for nx, ny, nz in mesh_grid:
        print("\n================================================")
        print(f"Running case: mesh={nx}x{ny}x{nz}")
        print("================================================")
        mesh = create_box(
            comm,
            [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
            [nx, ny, nz],
            CellType.hexahedron,
        )

        V, A, b, pb = sys(
            mesh,
            E=1.0e9,
            nu=0.3,
            gu=lambda x: np.isclose(x[2], 0.0),
        )

        tdim = mesh.topology.dim
        fdim = tdim - 1

        Gc = locate_entities_boundary(
            mesh,
            fdim,
            lambda x: np.isclose(x[2], 1.0, atol=1e-8),
        )

        gid = 1
        gt = np.full(Gc.shape, gid, dtype=np.int32)

        od = np.argsort(Gc)
        fi = Gc[od].astype(np.int32)
        fv = gt[od].astype(np.int32)

        mt = meshtags(mesh, fdim, fi, fv)
        ds = Measure("ds", domain=mesh, subdomain_data=mt)

        Ic = uniq_v(mesh, Gc)

        nrm, nf = nrm_bnd(
            mesh=mesh,
            Iv=Ic,
            ds_c=ds(gid),
        )

        xc = mesh.geometry.x[Ic]
        Nc = len(xc)
        Sc = Sc_n(A, Ic, nrm, tdim, show=True)
        rng = np.random.default_rng(123)
        v = rng.standard_normal(Nc)
        repeats = 5

        t_dense_handmade = time_average(handmade_matvec, repeats, Sc, v)
        y_dense = handmade_matvec(Sc, v)

        leaf_grid = [8, 16, 32, 64, 128]
        eta_grid = [1.0, 1.5, 2.5, 3.0, 4.0]

        tol = 1e-6
        split = "pca"
        lr_approx = "aca_full"

        shape = (len(leaf_grid), len(eta_grid))
        compression = np.zeros(shape)
        speedup = np.zeros(shape)
        errs = np.zeros(shape)

        for il, leaf in enumerate(leaf_grid):
            for ie, eta in enumerate(eta_grid):
                Sc_hmat = HMatrix(
                    xc,
                    Sc,
                    leaf_size=leaf,
                    eta=eta,
                    tol=tol,
                    split=split,
                    lr_approx=lr_approx,
                )

                dense_memory = Sc.size
                hmat_memory = Sc_hmat.block_tree.memory()
                # print(f"dense_memory = {dense_memory}")
                # print(f"hmat_memory = {hmat_memory}")

                t_hmat = time_average(lambda H, x: H @ x, repeats, Sc_hmat, v)
                y_hmat = Sc_hmat @ v

                err = np.linalg.norm(y_hmat - y_dense) / max(
                    1e-30, np.linalg.norm(y_dense)
                )

                compression[il, ie] = dense_memory / hmat_memory
                speedup[il, ie] = t_dense_handmade / t_hmat
                errs[il, ie] = err

        plot_heatmap(
            compression,
            leaf_grid,
            eta_grid,
            f"Compression ratio dense/$\\mathcal{{H}}$-matrix ",
            "dense entries / $\\mathcal{{H}}$-matrix entries",
            f"compression_ratio.png",
        )

        plot_heatmap(
            speedup,
            leaf_grid,
            eta_grid,
            f"Matvec speedup handmade dense / $\\mathcal{{H}}$-matrix",
            "speedup",
            f"matvec_speedup.png",
        )

        plot_heatmap(
            errs,
            leaf_grid,
            eta_grid,
            f"L2 error dense / $\\mathcal{{H}}$-matrix",
            "Error",
            f"error.png",
        )


if __name__ == "__main__":
    main()

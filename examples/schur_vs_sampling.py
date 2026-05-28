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

from scipy.linalg import solve


def schur_complement(A, B, C, D, assume_a="pos", overwrite_b=False, check_finite=False):
    # 1) Solve A * X = B for X, i.e., X = A^-1 * B
    X = solve(
        A, B, assume_a=assume_a, overwrite_b=overwrite_b, check_finite=check_finite
    )
    # 2) Compute C * X
    CX = np.dot(C, X)
    # 3) Compute S = D - CX
    S = D - CX
    return S



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

def Sc_Schur()
    K  = self.problem.A.convert("dense").getDenseArray()
    print("ending the conversion from sparse to dense")

        # Partition the global matrix into blocks
        all_dofs = np.arange(K.shape[0])
        uv_dofs = np.setdiff1d(all_dofs, boundary_dofs)
        # uv_dofs = np.setdiff1d(uv_dofs, np.array(self.dirichlet_dofs))
        uc_dofs = boundary_dofs

        Kvv = K[np.ix_(uv_dofs, uv_dofs)]
        Kvc = K[np.ix_(uv_dofs, uc_dofs)]
        Kcv = K[np.ix_(uc_dofs, uv_dofs)]
        Kcc = K[np.ix_(uc_dofs, uc_dofs)]

        S = np.linalg.inv(schur_complement(Kvv, Kvc, Kcv, Kcc))

        # import scipy.sparse.linalg as spla

        # solver = spla.gmres
        # T = schur_complement(Kvv, Kvc, Kcv, Kcc)
        # n = T.shape[0]
        # I = np.eye(n)  # Identity matrix
        # S = np.zeros_like(T, dtype=np.float64)  # Storage for inverse

        # # Solve Ax = I column-by-column
        # for i in range(n):
        #     e_i = I[:, i]  # i-th unit vector
        #     S[:, i], _ = solver(T, e_i, rtol=1e-15, maxiter=1000)


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
        Sc_sampling = Sc_n(A, Ic, nrm, tdim, show=True)



    dofs = np.array([25, 100, 324, 400, 676, 961, 1296, 1521, 1681])
    x_lin = np.linspace(100, 1681, 10)
    brut_times = np.array(
        [
            0.013589,
            0.121507,
            1.809009,
            3.103817,
            9.302821,
            20.086458,
            38.231056,
            52.474020,
            65.840583,
            # 1200,
            # 1800,
            # 2400,
        ]
    )
    schur_times_lu = np.array(
        [
            0.014657,
            0.137595,
            1.738470,
            3.097349,
            10.606437,
            25.781998,
            60.205876,
            91.310023,
            138.289542,
        ]
    )

    schur_times_gmres = np.array(
        [
            0.054452,
            0.732104,
            7.769769,
            12.125511,
            30.394951,
            69.287577,
            148.345563,
            230.163766,
            317.156277,
        ]
    )

    # Shifted reference power curves to start at the same point as the first data point
    shift_value = schur_times_gmres[0]
    # power_1 = dofs / dofs[0] * shift_value
    # power_2 = (dofs / dofs[0]) ** 2 * shift_value
    power_1 = x_lin / x_lin[0] * shift_value
    power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(dofs, brut_times, "o-", label="Direct Sampling", markersize=6, linewidth=2)
    ax.plot(
        dofs,
        schur_times_lu,
        "s-",
        label="Schur Compl. Direct (LU)",
        markersize=6,
        linewidth=2,
    )
    ax.plot(
        dofs,
        schur_times_gmres,
        "v-",
        label="Schur Compl. Iterative (GMRES)",
        markersize=6,
        linewidth=2,
    )

    ax.plot(x_lin, power_1, "--", color="black")  # label="O(N)")
    ax.plot(
        x_lin, power_2, "-.", color="black"
    )  # label="O(N²)")  # Annotate power curves

    ax.text(
        dofs[-1],
        power_1[-1],
        "O(N)",
        fontsize=8,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.text(
        dofs[-1],
        power_2[-1] - 10,
        "O(N²)",
        fontsize=8,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=10)
    ax.set_ylabel("CPU Time (s)", fontsize=10)

    # ax.set_title("Comparison of Brute Force and Schur Complement Methods", fontsize=16)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()

    fig.savefig("schur_vs_sampling.pdf", format="pdf")

    plt.show()

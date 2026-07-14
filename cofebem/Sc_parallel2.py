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

from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange
from numba.typed import List as NbList


import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from typing import Tuple
import numpy as np


def aca_full(
    A: np.ndarray, tol: float = 1.0e-6, k_max: int = 50
) -> Tuple[np.ndarray, np.ndarray]:

    m, n = A.shape
    R = A.copy()

    norm_A2 = np.linalg.norm(A, 2)
    max_A = np.abs(A).max()

    U_cols: list[np.ndarray] = []
    V_rows: list[np.ndarray] = []

    k = 0
    while True:
        flat_idx = np.abs(R).argmax()
        i_piv, j_piv = divmod(flat_idx, n)
        delta = R[i_piv, j_piv]

        if abs(delta) <= tol * max_A:
            break

        u_k = R[:, j_piv].copy()
        v_k = R[i_piv, :].copy() / delta

        R -= np.outer(u_k, v_k)

        U_cols.append(u_k)
        V_rows.append(v_k)
        k += 1

        if np.linalg.norm(u_k) * np.linalg.norm(v_k) <= tol * norm_A2:
            break

        if k_max is not None and k >= k_max:
            break

    U = np.column_stack(U_cols)
    V = np.column_stack(V_rows)

    return U, V


def matvec_(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    m, n = A.shape

    if x.ndim == 2:
        if x.shape[1] != 1:
            raise ValueError("x must be (n,) or (n,1).")
        x_flat = x[:, 0]
    elif x.ndim == 1:
        x_flat = x
    else:
        raise ValueError("x must be (n,) or (n,1).")

    if x_flat.shape[0] != n:
        raise ValueError("Dimension mismatch: A is (m, n) but x length is not n.")

    b = np.empty(m, dtype=A.dtype)

    for i in range(m):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x_flat[j]
        b[i] = acc
    return b


@dataclass
class Cluster:
    idx: np.ndarray
    bbox: Tuple[np.ndarray, np.ndarray]
    level: int
    parent: Optional["Cluster"] = None
    left: Optional["Cluster"] = None
    right: Optional["Cluster"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def diam(self) -> float:
        return float((self.bbox[1] - self.bbox[0]).max())


class ClusterTree:
    def __init__(self, pts: np.ndarray, leaf_size: int = 64, split: str = "pca"):
        self.pts = pts
        self.leaf = int(leaf_size)
        self.split = split
        self.root = self._build(np.arange(len(pts)), 0, None)

    def _build(self, idx: np.ndarray, level: int, parent: Optional[Cluster]):
        lo, hi = self.pts[idx].min(0), self.pts[idx].max(0)
        node = Cluster(idx=idx, bbox=(lo, hi), level=level, parent=parent)
        if len(idx) <= self.leaf:
            return node

        if self.split == "kd":
            axis = (hi - lo).argmax()
            mid = (hi[axis] + lo[axis]) * 0.5
            mask = self.pts[idx, axis] <= mid
        else:  # pca
            cen = self.pts[idx] - self.pts[idx].mean(0)
            cov = cen.T @ cen / (len(idx) - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v1 = eigvecs[:, eigvals.argmax()]
            proj = cen @ v1
            mid = np.median(proj)
            mask = proj <= mid
        if mask.all() or (~mask).all():
            return node  # degenerate; stop splitting
        node.left = self._build(idx[mask], level + 1, node)
        node.right = self._build(idx[~mask], level + 1, node)
        return node

    def leaf_order(self) -> List[int]:
        order: List[int] = []

        def _dfs(cl: Cluster):  # depth first search
            if cl.is_leaf:
                order.extend(cl.idx.tolist())
            else:
                _dfs(cl.left)
                _dfs(cl.right)

        _dfs(self.root)
        return order

    def _max_level(self):
        def depth(cl: Cluster):
            return cl.level if cl.is_leaf else max(depth(cl.left), depth(cl.right))

        return depth(self.root)


@dataclass
class Block:
    row: Cluster
    col: Cluster
    kind: str  # "dense" | "lr"
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    dense: Optional[np.ndarray] = None
    rank: Optional[int] = None

    @property
    def shape(self):
        return (len(self.row.idx), len(self.col.idx))

    def memory(self):
        if self.kind == "lr":
            return self.U.size + self.V.size
        return self.dense.size

    def matvec(self, x):
        if self.kind == "lr":
            return self.U @ (self.V.T @ x)
        return self.dense @ x
        # ~ if self.kind == "lr":
        # ~ return matvec_(self.U, (matvec_(self.V.T, x)))
        # ~ return matvec_(self.dense, x)


class BlockClusterTree:
    def __init__(
        self,
        row_tree: ClusterTree,
        col_tree: ClusterTree,
        A: np.ndarray,
        *,
        eta: float = 0.8,
        tol: float = 1e-6,
        lr_approx: str = "aca_partial",
    ):
        self.row_tree = row_tree
        self.col_tree = col_tree
        self.A = A
        self.eta = eta
        self.tol = tol
        self.lr_approx = lr_approx
        self.blocks: List[Block] = []
        self._build(row_tree.root, col_tree.root)

    @staticmethod
    def _dist(a: Cluster, b: Cluster):
        sep = np.maximum(
            0, np.maximum(a.bbox[0], b.bbox[0]) - np.minimum(a.bbox[1], b.bbox[1])
        )
        return float(np.linalg.norm(sep))

    def _admissible(self, a: Cluster, b: Cluster):
        return max(a.diam, b.diam) < self.eta * self._dist(a, b)

    def _build(self, a: Cluster, b: Cluster):
        if self._admissible(a, b):
            self._make_lr(a, b)
            return
        if a.is_leaf or b.is_leaf:
            self._make_dense(a, b)
            return
        for child_a in (a.left, a.right):
            for child_b in (b.left, b.right):
                self._build(child_a, child_b)

    def _make_dense(self, a: Cluster, b: Cluster):
        i, j = a.idx, b.idx
        B = self.A[np.ix_(i, j)]
        self.blocks.append(Block(a, b, "dense", dense=B))

    def _make_lr(self, a: Cluster, b: Cluster):
        i, j = a.idx, b.idx
        B = self.A[np.ix_(i, j)]

        match self.lr_approx:
            # case "truncated_svd":
            #     lr_method = truncated_svd
            case "aca_full":
                lr_method = aca_full
            # case "aca_partial":
            #     lr_method = aca_partial
            # case "aca_plus":
            #     lr_method = aca_plus
            # case "aca_gp":
            #     lr_method = aca_gp
            case _:
                raise ValueError(
                    f"Unknown low rank approximation method: {self.lr_approx}"
                )

        U, V = lr_method(B, self.tol)
        self.blocks.append(Block(a, b, "lr", U=U, V=V, rank=U.shape[1]))

    def memory(self):
        return sum(bl.memory() for bl in self.blocks)


@njit
def _hmatvec_numba(nb_rows, nb_cols, nb_kinds, nb_U, nb_V, nb_D, x, y):
    for k in range(len(nb_kinds)):
        rows = nb_rows[k]
        cols = nb_cols[k]
        if nb_kinds[k] == 1:
            # low-rank: U @ (V.T @ x[cols])
            U = nb_U[k]
            V = nb_V[k]
            tmp = V.T @ x[cols]
            y[rows] += U @ tmp
        else:

            D = nb_D[k]
            y[rows] += D @ x[cols]


class HMatrix:
    def __init__(
        self,
        pts: np.ndarray,
        A: np.ndarray,
        *,
        leaf_size: int = 64,
        eta: float = 0.8,
        tol: float = 1e-6,
        split: str = "pca",
        lr_approx: str = "aca_full",
    ) -> None:
        self.pts = pts
        n = len(pts)
        if A.shape != (n, n):
            raise ValueError("A must be (n,n) and correspond to pts order")

        self.tree = ClusterTree(pts, leaf_size=leaf_size, split=split)
        self.block_tree = BlockClusterTree(
            self.tree, self.tree, A, eta=eta, tol=tol, lr_approx=lr_approx
        )

        self._prepare_numba()

    def _prepare_numba(self):
        self._nb_rows = NbList()
        self._nb_cols = NbList()
        self._nb_U = NbList()
        self._nb_V = NbList()
        self._nb_D = NbList()
        kinds_py = []

        z = np.empty((0, 0))  # placeholder for unused blocks

        for bl in self.block_tree.blocks:
            self._nb_rows.append(np.ascontiguousarray(bl.row.idx.astype(np.int64)))
            self._nb_cols.append(np.ascontiguousarray(bl.col.idx.astype(np.int64)))

            if bl.kind == "lr":
                self._nb_U.append(np.ascontiguousarray(bl.U))
                self._nb_V.append(np.ascontiguousarray(bl.V))
                self._nb_D.append(z)
                kinds_py.append(1)
            else:
                self._nb_U.append(z)
                self._nb_V.append(z)
                self._nb_D.append(np.ascontiguousarray(bl.dense))
                kinds_py.append(0)

        self._nb_kinds = np.array(kinds_py, dtype=np.int8)

    def __matmul__(self, x):
        if x.shape[0] != len(self.pts):
            raise ValueError("size mismatch")
        y = np.zeros_like(x, dtype=float)
        for bl in self.block_tree.blocks:
            y[bl.row.idx] += bl.matvec(x[bl.col.idx])
        return y

    def matvec_overhead(self, x):
        # measure overhead = looping + broadcasting (no arithmetic)
        y = np.zeros_like(x, dtype=float)
        s = 0.0
        for bl in self.block_tree.blocks:
            rows = bl.row.idx
            cols = bl.col.idx
            s += float(np.sum(x[cols])) * 0.0
            y[rows] += 0.0
        return y

    # def __matmul__(self, x):
    #     if x.ndim != 1:
    #         raise ValueError("Use a 1‑D vector on the right‑hand side.")
    #     if x.shape[0] != len(self.pts):
    #         raise ValueError(
    #             "Size mismatch: got |x| = %d, expected %d" % (x.shape[0], len(self.pts))
    #         )

    #     y = np.zeros_like(x, dtype=float)

    #     try:
    #         _hmatvec_numba(
    #             self._nb_rows,
    #             self._nb_cols,
    #             self._nb_kinds,
    #             self._nb_U,
    #             self._nb_V,
    #             self._nb_D,
    #             x,
    #             y,
    #         )
    #     except Exception:
    #         for bl in self.block_tree.blocks:
    #             y[bl.row.idx] += bl.matvec(x[bl.col.idx])

    #     return y

    def stats(self):
        lr = sum(bl.kind == "lr" for bl in self.block_tree.blocks)
        dn = len(self.block_tree.blocks) - lr
        return {
            "blocks": len(self.block_tree.blocks),
            "low_rank": lr,
            "dense": dn,
            "memory": self.block_tree.memory(),
        }

    def visualize(self, save_path=None, dpi=300):
        n = len(self.pts)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("columns")
        ax.set_ylabel("rows")

        perm = np.array(self.tree.leaf_order())
        pos = np.empty_like(perm)
        pos[perm] = np.arange(n)

        for bl in self.block_tree.blocks:
            r0, r1 = pos[bl.row.idx].min(), pos[bl.row.idx].max() + 1
            c0, c1 = pos[bl.col.idx].min(), pos[bl.col.idx].max() + 1
            w, h = c1 - c0, r1 - r0
            face = "#1f77b4" if bl.kind == "lr" else "#ff8c00"
            rect = mpatches.Rectangle(
                (c0, r0), w, h, facecolor=face, edgecolor="black", linewidth=0.3
            )
            ax.add_patch(rect)
        handles = [
            mpatches.Patch(facecolor="#1f77b4", edgecolor="black", label="low‑rank"),
            mpatches.Patch(facecolor="#ff8c00", edgecolor="black", label="dense"),
        ]
        ax.legend(handles=handles, loc="upper right")
        ax.annotate(
            f"η = {self.block_tree.eta}",
            xy=(0.5, -0.10),
            xycoords="axes fraction",
            ha="center",
            va="top",
            color="red",
            fontsize=12,
        )
        ax.set_title("ℋ‑matrix block structure")

        if save_path is not None:
            if save_path.endswith(".png") or save_path.endswith(".jpg"):
                fig.savefig(save_path, dpi=dpi)
            else:
                fig.savefig(save_path)

        plt.show()


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


def Sc_n_parallel(A, Ic, nrm, tdim=3, f0=1e9, show=True):
    """
    Parallel construction of the normal compliance matrix Sc.

    Parallel strategy:
    - each MPI rank owns a full serial copy of the FE problem, built on COMM_SELF;
    - columns of Sc are split between MPI ranks;
    - rank 0 gathers the column blocks and reconstructs the full dense Sc.
    """
    world = MPI.COMM_WORLD
    rank = world.rank
    size = world.size

    comm = MPI.COMM_SELF

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

    cols_local = np.arange(rank, nc, size, dtype=np.int32)
    S_cols = np.zeros((nc, len(cols_local)), dtype=float)

    if show:
        iterator = tqdm(
            enumerate(cols_local),
            total=len(cols_local),
            desc=f"Rank {rank} Sampling Snn",
            unit="col",
            position=rank,
            leave=True,
        )
    else:
        iterator = enumerate(cols_local)

    for jj, j in iterator:
        rhs.set(0.0)
        rhs.setValues(cd[j], f0 * n[j], addv=PETSc.InsertMode.INSERT_VALUES)
        rhs.assemble()

        ksp.solve(rhs, u)

        ux_all = u.getArray(readonly=True).copy().reshape(-1, tdim)
        ux = ux_all[Ic]

        S_cols[:, jj] = np.einsum("ij,ij->i", n, ux) / f0

    gathered = world.gather((cols_local, S_cols), root=0)

    if rank == 0:
        S = np.zeros((nc, nc), dtype=float)
        for cols, block in gathered:
            S[:, cols] = block
        return S

    return None


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
    bad = m < 1e-14
    if np.any(bad):
        print(f"Warning: {np.sum(bad)} boundary normals have near-zero norm")

    nrm[~bad] /= m[~bad, None]
    nrm[bad] = np.array([0.0, 0.0, 1.0])
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
    plt.xticks(np.arange(len(eta_grid)), eta_grid)
    plt.yticks(np.arange(len(leaf_grid)), leaf_grid)
    plt.ylabel(r"$n_{leaf}$", fontsize=20)
    plt.xlabel(r"$\eta$", fontsize=20)
    plt.tick_params(axis="both", labelsize=15)
    # plt.title(title,fontsize=22)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label, fontsize=18)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(fname, dpi=500)
    plt.show()


def main():
    world = MPI.COMM_WORLD
    rank = world.rank
    size = world.size

    comm = MPI.COMM_SELF

    mesh_grid = [
        (50, 50, 5),
        # (20, 20, 10),
        # (25, 25, 12),
        # (40, 40, 20),
    ]

    for nx, ny, nz in mesh_grid:
        if rank == 0:
            print("\n================================================", flush=True)
            print(f"Running case: mesh={nx}x{ny}x{nz}", flush=True)
            print(f"MPI ranks: {size}", flush=True)
            print("================================================", flush=True)

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

        # Keep the top-boundary vertex extraction as in the original code.
        Ic = uniq_v(mesh, Gc)

        # Simple normal vector on the flat top face z = 1:
        # qvec = [0, 0, 1]
        nrm = np.zeros((Ic.size, tdim), dtype=float)
        nrm[:, 2] = 1.0

        xc = mesh.geometry.x[Ic]
        Nc = len(xc)

        if rank == 0:
            print(f"Nc = {Nc}", flush=True)
            print("Start Sc_n_parallel", flush=True)

        Sc = Sc_n_parallel(A, Ic, nrm, tdim, show=True)

        # From here, only rank 0 owns the full Sc matrix.
        # Other ranks stop cleanly.
        if rank != 0:
            return

        rng = np.random.default_rng(123)
        v = rng.standard_normal(Nc)
        repeats = 5

        # ~ t_dense_handmade = time_average(handmade_matvec, repeats, Sc, v)

        t_dense_handmade = time_average(lambda H, x: H @ x, repeats, Sc, v)
        # ~ y_dense = handmade_matvec(Sc, v)
        y_dense = Sc @ v

        leaf_grid = [8, 16, 32, 64, 128, 256]
        eta_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

        tol = 1e-3
        split = "pca"
        lr_approx = "aca_full"

        shape = (len(leaf_grid), len(eta_grid))
        compression = np.zeros(shape)
        speedup = np.zeros(shape)
        errs = np.zeros(shape)

        print("Scn Finished !", flush=True)
        print("Start H_matrix", flush=True)

        for il, leaf in enumerate(leaf_grid):
            for ie, eta in enumerate(eta_grid):
                print(f"HMatrix: leaf={leaf}, eta={eta}", flush=True)

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

                t_hmat = time_average(lambda H, x: H @ x, repeats, Sc_hmat, v)
                y_hmat = Sc_hmat @ v

                err = np.linalg.norm(y_hmat - y_dense) / max(
                    1e-30, np.linalg.norm(y_dense)
                )

                compression[il, ie] = dense_memory / hmat_memory
                speedup[il, ie] = t_dense_handmade / t_hmat
                errs[il, ie] = err

        print("finish H_matrix", flush=True)

        plot_heatmap(
            compression,
            leaf_grid,
            eta_grid,
            rf"Compression ratio dense/$\mathcal{{H}}$-matrix ",
            f"Compression ratio",
            # r"dense entries / $\mathcal{{H}}$-matrix entries",
            f"compression_ratio_parallel.png",
        )

        plot_heatmap(
            speedup,
            leaf_grid,
            eta_grid,
            rf"Matvec speedup dense / $\mathcal{{H}}$-matrix",
            "MatVec speedup",
            f"matvec_speedup_parallel.png",
        )

        plot_heatmap(
            errs * 100,
            leaf_grid,
            eta_grid,
            rf"L2 error dense / $\mathcal{{H}}$-matrix",
            r"$L_2$ Error [$\%$]",
            f"error_parallel.png",
        )


if __name__ == "__main__":
    main()

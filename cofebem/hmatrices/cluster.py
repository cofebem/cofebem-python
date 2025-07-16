import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union
import meshio
from numba import njit, prange
from numba.typed import List as NbList


@njit(parallel=True, fastmath=True, cache=True)
def _hmatvec_numba(
    row_inds,  # numba.typed.List[np.ndarray[int64]]
    col_inds,  # numba.typed.List[np.ndarray[int64]]
    kinds,  # np.ndarray[int8]  – 0 ↔ dense, 1 ↔ low‑rank
    U_buf,  # numba.typed.List[np.ndarray[float64]]
    V_buf,  # numba.typed.List[np.ndarray[float64]]
    D_buf,  # numba.typed.List[np.ndarray[float64]]
    x: np.ndarray,
    y: np.ndarray,
) -> None:

    nblocks = len(kinds)
    for b in prange(nblocks):
        rows = row_inds[b]
        cols = col_inds[b]
        if kinds[b] == 1:  # low‑rank block
            y[rows] += U_buf[b] @ (V_buf[b] @ x[cols])
        else:  # dense block
            y[rows] += D_buf[b] @ x[cols]


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
        # choose direction
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

        def _dfs(n: Cluster):
            if n.is_leaf:
                order.extend(n.idx.tolist())
            else:
                _dfs(n.left)
                _dfs(n.right)

        _dfs(self.root)
        return order

    def _max_level(self):
        def depth(n):
            return n.level if n.is_leaf else max(depth(n.left), depth(n.right))

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
            return self.U @ (self.V @ x)
        return self.dense @ x


class BlockClusterTree:
    def __init__(
        self,
        K: Callable[[np.ndarray, np.ndarray], np.ndarray],
        row_tree: ClusterTree,
        col_tree: ClusterTree,
        eta: float = 0.8,
        tol: float = 1e-6,
    ):
        self.K = K
        self.row_tree = row_tree
        self.col_tree = col_tree
        self.eta = eta
        self.tol = tol
        self.blocks: List[Block] = []
        self._build(row_tree.root, col_tree.root)

    @staticmethod
    def _dist(a: Cluster, b: Cluster):
        sep = np.maximum(
            0, np.maximum(a.bbox[0], b.bbox[0]) - np.minimum(a.bbox[1], b.bbox[1])
        )
        return float(np.linalg.norm(sep))

    def _admits(self, a: Cluster, b: Cluster):
        return max(a.diam, b.diam) < self.eta * self._dist(a, b)

    def _build(self, a: Cluster, b: Cluster):
        if self._admits(a, b):
            self._make_lr(a, b)
            return
        if a.is_leaf or b.is_leaf:
            self._make_dense(a, b)
            return
        for child_a in (a.left, a.right):
            for child_b in (b.left, b.right):
                self._build(child_a, child_b)

    def _make_dense(self, a: Cluster, b: Cluster):
        B = self.K(a.idx, b.idx)
        self.blocks.append(Block(a, b, "dense", dense=B))

    def _make_lr(self, a: Cluster, b: Cluster):
        B = self.K(a.idx, b.idx)
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        k = int((S > self.tol * S[0]).sum()) or 1
        self.blocks.append(
            Block(a, b, "lr", U=U[:, :k], V=np.diag(S[:k]) @ Vt[:k], rank=k)
        )

    def memory(self):
        return sum(bl.memory() for bl in self.blocks)


class HMatrix:
    def __init__(
        self,
        pts: np.ndarray,
        cells,
        *,
        A: Optional[np.ndarray] = None,
        kernel: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        leaf: int = 64,
        eta: float = 0.8,
        tol: float = 1e-6,
        split: str = "pca",
    ) -> None:
        self.pts = pts
        self.cells = cells
        n = len(pts)
        if A is not None:
            if A.shape != (n, n):
                raise ValueError("A must be (n,n) and correspond to pts order")
            self._K = lambda I, J: A[np.ix_(I, J)]
        else:
            if kernel is None:
                kernel = lambda x, y: 1.0 / (np.linalg.norm(x - y) + 1e-8)
            self._K = lambda I, J: kernel(pts[I][:, None, :], pts[J][None, :, :])

        self.tree = ClusterTree(pts, leaf_size=leaf, split=split)
        self.block_tree = BlockClusterTree(
            self._K, self.tree, self.tree, eta=eta, tol=tol
        )

        self._prepare_numba()

    @classmethod
    def from_dense(cls, A: np.ndarray, pts: np.ndarray, cells, **kw):
        return cls(pts, cells=cells, A=A, **kw)

    def _prepare_numba(self):

        self._nb_rows = NbList()
        self._nb_cols = NbList()
        self._nb_U = NbList()
        self._nb_V = NbList()
        self._nb_D = NbList()
        kinds_py = []

        z = np.empty((0, 0))

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

    # def __matmul__(self, x):
    #     if x.shape[0] != len(self.pts):
    #         raise ValueError("size mismatch")
    #     y = np.zeros_like(x, dtype=float)
    #     for bl in self.block_tree.blocks:
    #         y[bl.row.idx] += bl.matvec(x[bl.col.idx])
    #     return y

    def __matmul__(self, x):
        if x.ndim != 1:
            raise ValueError("Use a 1‑D vector on the right‑hand side.")
        if x.shape[0] != len(self.pts):
            raise ValueError(
                "Size mismatch: got |x| = %d, expected %d" % (x.shape[0], len(self.pts))
            )

        y = np.zeros_like(x, dtype=float)

        try:
            # prefer jitted path
            _hmatvec_numba(
                self._nb_rows,
                self._nb_cols,
                self._nb_kinds,
                self._nb_U,
                self._nb_V,
                self._nb_D,
                x,
                y,
            )
        except Exception:
            # fallback – original pure‑Python version
            for bl in self.block_tree.blocks:
                y[bl.row.idx] += bl.matvec(x[bl.col.idx])

        return y

    def stats(self):
        lr = sum(bl.kind == "lr" for bl in self.block_tree.blocks)
        dn = len(self.block_tree.blocks) - lr
        return {
            "blocks": len(self.block_tree.blocks),
            "low_rank": lr,
            "dense": dn,
            "memory": self.block_tree.memory(),
        }

    def show(self, save_path=None):
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
            fig.savefig(save_path)

        plt.show()


if __name__ == "__main__":
    # pts = np.random.rand(200, 3)
    mesh = meshio.read("hollow_cylinder.xdmf")
    pts = mesh.points
    cells = mesh.cells
    # print(mesh.cells)
    # print(mesh.cells_dict)
    X = pts[:, None, :]
    Y = pts[None, :, :]
    A_full = 1.0 / (np.linalg.norm(X - Y, axis=2) + 1e-8)

    hm = HMatrix.from_dense(A_full, pts, cells, leaf=8, eta=0.7)
    print("stats:", hm.stats())

    v = np.random.randn(len(pts))
    ref = A_full @ v
    y = hm @ v
    print("relative error", np.linalg.norm(ref - y) / np.linalg.norm(ref))
    # # hm.show()
    # hm.export_meshio()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numba import njit, prange
from numba.typed import List as NbList


from .cluster_tree import ClusterTree
from .block_cluster_tree import BlockClusterTree


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
        lr_approx: str = "aca_partial",
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

    def visualize(self, save_path=None):
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

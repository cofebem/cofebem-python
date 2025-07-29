import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .cluster_tree import Cluster, ClusterTree
from .low_rank_approx.aca_full import aca_full
from .low_rank_approx.aca_partial import aca_partial
from .low_rank_approx.aca_plus import aca_plus
from .low_rank_approx.aca_gp import aca_gp
from .low_rank_approx.truncated_svd import truncated_svd


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
            case "truncated_svd":
                lr_method = truncated_svd
            case "aca_full":
                lr_method = aca_full
            case "aca_partial":
                lr_method = aca_partial
            case "aca_plus":
                lr_method = aca_plus
            case "aca_gp":
                lr_method = aca_gp
            case _:
                raise ValueError(
                    f"Unknown low rank approximation method: {self.lr_approx}"
                )

        U, V = lr_method(B, self.tol)
        self.blocks.append(Block(a, b, "lr", U=U, V=V, rank=U.shape[1]))

    def memory(self):
        return sum(bl.memory() for bl in self.blocks)

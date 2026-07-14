import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .cluster_tree import Cluster, ClusterTree
from .low_rank_approx import (
    aca_full,
    aca_partial,
    aca_plus,
    truncated_svd,
    aca_partial_entry,
)
from .entry_source import MatrixEntrySource


@dataclass
class Block:
    """A single block in a block-cluster-tree partition.

    Each block covers rows ``row.idx`` and columns ``col.idx`` of the global
    matrix.  It is stored either as a low-rank factorisation ``U @ V.T``
    (``kind="lr"``) or as a full dense submatrix (``kind="dense"``).

    Attributes
    ----------
    row : Cluster
        Row cluster.
    col : Cluster
        Column cluster.
    kind : {"dense", "lr"}
        Storage format.
    U : ndarray of shape (m, r) or None
        Left low-rank factor; only set when ``kind="lr"``.
    V : ndarray of shape (n, r) or None
        Right low-rank factor; only set when ``kind="lr"``.
    dense : ndarray of shape (m, n) or None
        Dense submatrix; only set when ``kind="dense"``.
    rank : int or None
        Numerical rank ``r``; only set when ``kind="lr"``.
    """

    row: Cluster
    col: Cluster
    kind: str  # "dense" | "lr"
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    dense: Optional[np.ndarray] = None
    rank: Optional[int] = None

    @property
    def shape(self):
        """Return ``(m, n)`` — the number of rows and columns in this block."""
        return (len(self.row.idx), len(self.col.idx))

    def memory(self):
        """Return the number of stored floating-point entries."""
        if self.kind == "lr":
            return self.U.size + self.V.size
        return self.dense.size

    def to_dense(self) -> np.ndarray:
        """Expand the block to a full ``(m, n)`` dense matrix."""
        if self.kind == "lr":
            return self.U @ self.V.T
        return self.dense

    def matvec(self, x):
        """Apply this block to a vector *x* of length ``n``, returning length-``m`` result."""
        if self.kind == "lr":
            return self.U @ (self.V.T @ x)
        return self.dense @ x

    def matvec_T(self, x):
        """Apply the transpose of this block to *x* of length ``m``, returning length-``n``."""
        if self.kind == "lr":
            return self.V @ (self.U.T @ x)
        return self.dense.T @ x


def _block_add(bl_a: Block, bl_b: Block, tol: float) -> Block:
    """Add two blocks sharing the same row/col clusters, recompressing LR+LR pairs.

    For two low-rank blocks the combined factor is recompressed via a QR-SVD
    step.  Mixed or dense pairs fall back to dense addition.

    Parameters
    ----------
    bl_a, bl_b : Block
        Blocks to add.  Must cover the same row and column clusters.
    tol : float
        Relative truncation tolerance for the SVD recompression.

    Returns
    -------
    Block
        New block equal to ``bl_a + bl_b``.
    """
    row, col = bl_a.row, bl_a.col
    if bl_a.kind == "lr" and bl_b.kind == "lr":
        U = np.hstack([bl_a.U, bl_b.U])
        V = np.hstack([bl_a.V, bl_b.V])
        if U.shape[1] == 0:
            m, n = bl_a.shape
            return Block(row, col, "lr", U=np.zeros((m, 0)), V=np.zeros((n, 0)), rank=0)
        Q_U, R_U = np.linalg.qr(U)
        Q_V, R_V = np.linalg.qr(V)
        W, S, Zt = np.linalg.svd(R_U @ R_V.T, full_matrices=False)
        r = max(1, int(np.sum(S > tol * S[0]))) if S[0] > 0 else 1
        sqrt_S = np.sqrt(S[:r])
        return Block(row, col, "lr",
                     U=Q_U @ (W[:, :r] * sqrt_S),
                     V=Q_V @ (Zt[:r, :].T * sqrt_S),
                     rank=r)
    elif bl_a.kind == "dense" and bl_b.kind == "dense":
        return Block(row, col, "dense", dense=bl_a.dense + bl_b.dense)
    else:
        return Block(row, col, "dense", dense=bl_a.to_dense() + bl_b.to_dense())


class BlockClusterTree:
    """Block-cluster-tree partition of a matrix into dense and low-rank blocks.

    Recursively partitions the product of two cluster trees into leaves that
    are classified as either *admissible* (stored as a low-rank approximation)
    or *inadmissible* (stored as a dense submatrix).  Admissibility is checked
    via the standard η-criterion: a pair ``(a, b)`` is admissible if
    ``max(diam(a), diam(b)) < η · dist(a, b)``.

    Parameters
    ----------
    row_tree : ClusterTree
        Cluster tree for the row indices.
    col_tree : ClusterTree
        Cluster tree for the column indices.
    A : ndarray or MatrixEntrySource of shape (n, n)
        The matrix to compress, either materialised or available through
        block queries.
    eta : float
        Admissibility parameter.  Larger values admit more low-rank blocks.
    tol : float
        Tolerance forwarded to the low-rank approximation routine.
    lr_approx : {"aca_partial", "aca_full", "aca_plus", "truncated_svd"}
        Low-rank approximation method for admissible blocks.
    symmetric : bool
        If ``True``, only the lower-triangular blocks are stored; the upper
        triangle is recovered by transposition during matrix-vector products.

    Attributes
    ----------
    blocks : list of Block
        Flat list of all leaves in the partition.
    """

    def __init__(
        self,
        row_tree: ClusterTree,
        col_tree: ClusterTree,
        A: np.ndarray,
        *,
        eta: float = 0.8,
        tol: float = 1e-6,
        lr_approx: str = "aca_full",
        symmetric: bool = False,
        max_rank: int = 50,
    ):
        self.row_tree = row_tree
        self.col_tree = col_tree
        self.eta = eta
        self.tol = tol
        self.lr_approx = lr_approx
        self.symmetric = symmetric
        self.max_rank = max_rank
        self.blocks: List[Block] = []
        self._entry_source = not isinstance(A, np.ndarray)
        if self._entry_source:
            if not isinstance(A, MatrixEntrySource):
                raise TypeError("A must be a NumPy array or MatrixEntrySource")
            if lr_approx != "aca_partial":
                raise ValueError(
                    "entry-source H-matrices support lr_approx='aca_partial' only"
                )
        self._A = A
        self._build(row_tree.root, col_tree.root)
        del self._A

    @staticmethod
    def _dist(a: Cluster, b: Cluster):
        """Euclidean distance between bounding boxes of clusters *a* and *b*."""
        sep = np.maximum(
            0, np.maximum(a.bbox[0], b.bbox[0]) - np.minimum(a.bbox[1], b.bbox[1])
        )
        return float(np.linalg.norm(sep))

    def _admissible(self, a: Cluster, b: Cluster):
        """Return ``True`` if the pair ``(a, b)`` satisfies the η-admissibility criterion."""
        return max(a.diam, b.diam) < self.eta * self._dist(a, b)

    def _build(self, a: Cluster, b: Cluster):
        """Recursively partition block ``(a, b)`` into dense or low-rank leaves."""
        if self.symmetric and a.cid < b.cid:
            return
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
        """Append a dense block for the inadmissible pair ``(a, b)``."""
        B = self._get_block(a.idx, b.idx)
        self.blocks.append(Block(a, b, "dense", dense=B))

    def _make_lr(self, a: Cluster, b: Cluster):
        """Append a low-rank block for the admissible pair ``(a, b)``."""
        if self._entry_source:
            U, V = aca_partial_entry(
                self._A, a.idx, b.idx, self.tol, self.max_rank
            )
            self.blocks.append(Block(a, b, "lr", U=U, V=V, rank=U.shape[1]))
            return

        B = self._get_block(a.idx, b.idx)

        match self.lr_approx:
            case "truncated_svd":
                lr_method = truncated_svd
            case "aca_full":
                lr_method = aca_full
            case "aca_partial":
                lr_method = aca_partial
            case "aca_plus":
                lr_method = aca_plus
            case _:
                raise ValueError(
                    f"Unknown low rank approximation method: {self.lr_approx}"
                )

        U, V = lr_method(B, self.tol, self.max_rank)
        self.blocks.append(Block(a, b, "lr", U=U, V=V, rank=U.shape[1]))

    def _get_block(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Read one subblock from a dense array or an entry source."""
        if self._entry_source:
            block = self._A.get_block(rows, columns)
        else:
            block = self._A[np.ix_(rows, columns)]
        block = np.asarray(block, dtype=float)
        expected = (len(rows), len(columns))
        if block.shape != expected:
            raise ValueError(
                f"entry source returned shape {block.shape}, expected {expected}"
            )
        return block

    def _copy_with_blocks(self, new_blocks: List[Block]) -> "BlockClusterTree":
        """Return a shallow copy sharing trees/parameters but using *new_blocks*."""
        obj = BlockClusterTree.__new__(BlockClusterTree)
        obj.row_tree = self.row_tree
        obj.col_tree = self.col_tree
        obj.eta = self.eta
        obj.tol = self.tol
        obj.lr_approx = self.lr_approx
        obj.symmetric = self.symmetric
        obj.max_rank = self.max_rank
        obj._entry_source = self._entry_source
        obj.blocks = new_blocks
        return obj

    def memory(self):
        """Return the total number of stored floating-point entries across all blocks."""
        return sum(bl.memory() for bl in self.blocks)

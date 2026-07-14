import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional
from scipy.linalg import lu_factor, lu_solve

from .cluster_tree import ClusterTree
from .block_cluster_tree import Block, BlockClusterTree, _block_add
from .entry_source import MatrixEntrySource


class HMatrix:
    """Hierarchical matrix built from a point set and matrix entry provider.

    Constructs a :class:`ClusterTree` over *pts* and then a
    :class:`BlockClusterTree` that compresses admissible blocks of *A* with a
    chosen low-rank approximation method. *A* may be dense or queried through
    :class:`MatrixEntrySource`. Provides matrix-vector products, arithmetic
    operators, dense assembly, LU-based solve, diagnostics, and a block-
    structure visualisation.

    Parameters
    ----------
    pts : ndarray of shape (n, d)
        Spatial coordinates of the degrees of freedom.  The i-th row
        corresponds to the i-th row/column of *A*.
    A : ndarray or MatrixEntrySource of shape (n, n)
        Matrix to compress. Entry sources are queried blockwise and do not
        require a global dense matrix.
    leaf_size : int
        Maximum cluster size for the cluster tree.
    eta : float
        Admissibility parameter (see :class:`BlockClusterTree`).
    tol : float
        Low-rank approximation tolerance.
    split : {"pca", "kd"}
        Splitting strategy for the cluster tree.
    lr_approx : {"aca_partial", "aca_full", "aca_plus", "truncated_svd"}
        Low-rank approximation method for admissible blocks.
    symmetric : bool
        If ``True``, only the lower-triangular blocks are stored.

    Attributes
    ----------
    pts : ndarray
        Point coordinates.
    tol : float
        Approximation tolerance.
    symmetric : bool
        Whether the symmetric storage format is used.
    tree : ClusterTree
        Spatial cluster tree.
    block_tree : BlockClusterTree
        Block partition with compressed blocks.
    """

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
        symmetric: bool = False,
        max_rank: int = 50,
    ) -> None:
        self.pts = pts
        self.tol = tol
        self.symmetric = symmetric
        n = len(pts)
        if A.shape != (n, n):
            raise ValueError("A must be (n,n) and correspond to pts order")

        self.tree = ClusterTree(pts, leaf_size=leaf_size, split=split)
        self.block_tree = BlockClusterTree(
            self.tree, self.tree, A,
            eta=eta, tol=tol, lr_approx=lr_approx, symmetric=symmetric,
            max_rank=max_rank,
        )
        self._lu_cache: Optional[tuple] = None

    @classmethod
    def from_entry_source(
        cls,
        pts: np.ndarray,
        source: MatrixEntrySource,
        **options,
    ) -> "HMatrix":
        """Build directly from selected source queries, never a dense matrix."""
        return cls(pts, source, **options)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        n = len(self.pts)
        return n, n

    # ------------------------------------------------------------------
    # Internal constructor used by arithmetic operators
    # ------------------------------------------------------------------
    @classmethod
    def _from_parts(
        cls,
        pts: np.ndarray,
        tree: ClusterTree,
        block_tree: BlockClusterTree,
        tol: float,
        symmetric: bool,
    ) -> "HMatrix":
        """Construct an HMatrix directly from pre-built tree objects (no compression)."""
        obj = cls.__new__(cls)
        obj.pts = pts
        obj.tol = tol
        obj.symmetric = symmetric
        obj.tree = tree
        obj.block_tree = block_tree
        obj._lu_cache = None
        return obj

    # ------------------------------------------------------------------
    # Matrix-vector / matrix-matrix product
    # ------------------------------------------------------------------
    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        """Compute the H-matrix–vector product ``y = H @ x``.

        Parameters
        ----------
        x : ndarray of shape (n,) or (n, k)
            Input vector or matrix.

        Returns
        -------
        y : ndarray of the same shape as *x*.
        """
        n = len(self.pts)
        if x.shape[0] != n:
            raise ValueError("size mismatch")
        y = np.zeros_like(x, dtype=float)
        if self.symmetric:
            for bl in self.block_tree.blocks:
                y[bl.row.idx] += bl.matvec(x[bl.col.idx])
                if bl.row is not bl.col:  # off-diagonal: add transposed contribution
                    y[bl.col.idx] += bl.matvec_T(x[bl.row.idx])
        else:
            for bl in self.block_tree.blocks:
                y[bl.row.idx] += bl.matvec(x[bl.col.idx])
        return y

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other: "HMatrix") -> "HMatrix":
        """Return ``self + other`` as a new HMatrix (requires a shared cluster tree)."""
        if not isinstance(other, HMatrix):
            return NotImplemented
        if self.tree is not other.tree:
            raise ValueError(
                "H-matrix addition requires a shared cluster tree; "
                "build both from the same ClusterTree."
            )
        if len(self.block_tree.blocks) != len(other.block_tree.blocks):
            raise ValueError("H-matrices must have the same block structure for addition.")
        new_blocks = [
            _block_add(bl_a, bl_b, self.tol)
            for bl_a, bl_b in zip(self.block_tree.blocks, other.block_tree.blocks)
        ]
        return HMatrix._from_parts(
            self.pts, self.tree,
            self.block_tree._copy_with_blocks(new_blocks),
            self.tol, self.symmetric,
        )

    def __mul__(self, scalar: float) -> "HMatrix":
        """Return ``scalar * self`` as a new HMatrix."""
        new_blocks = []
        for bl in self.block_tree.blocks:
            if bl.kind == "lr":
                new_blocks.append(
                    Block(bl.row, bl.col, "lr",
                          U=bl.U * scalar, V=bl.V.copy(), rank=bl.rank)
                )
            else:
                new_blocks.append(
                    Block(bl.row, bl.col, "dense", dense=bl.dense * scalar)
                )
        return HMatrix._from_parts(
            self.pts, self.tree,
            self.block_tree._copy_with_blocks(new_blocks),
            self.tol, self.symmetric,
        )

    def __rmul__(self, scalar: float) -> "HMatrix":
        """Support ``scalar * H``."""
        return self.__mul__(scalar)

    def __neg__(self) -> "HMatrix":
        """Return ``-self``."""
        return self.__mul__(-1.0)

    def __sub__(self, other: "HMatrix") -> "HMatrix":
        """Return ``self - other``."""
        return self.__add__((-1.0) * other)

    # ------------------------------------------------------------------
    # Dense assembly
    # ------------------------------------------------------------------
    def to_dense(self) -> np.ndarray:
        """Assemble the full n×n matrix."""
        n = len(self.pts)
        A = np.zeros((n, n))
        for bl in self.block_tree.blocks:
            i, j = bl.row.idx, bl.col.idx
            A[np.ix_(i, j)] += bl.to_dense()
            if self.symmetric and bl.row is not bl.col:
                A[np.ix_(j, i)] += bl.to_dense().T
        return A

    # ------------------------------------------------------------------
    # LU factorisation (dense, in cluster ordering)
    # ------------------------------------------------------------------
    def lu(self) -> "HMatrix":
        """Assemble to dense in cluster order and cache an LU factorisation."""
        perm = np.array(self.tree.leaf_order())
        D = self.to_dense()[np.ix_(perm, perm)]
        self._lu_cache = (lu_factor(D), perm)
        return self

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve H @ x = b using the cached LU factorisation (calls lu() if needed)."""
        if self._lu_cache is None:
            self.lu()
        lu_fac, perm = self._lu_cache
        # permute b to cluster ordering, solve, then unpermute x
        y = lu_solve(lu_fac, b[perm])
        x = np.empty_like(y)
        x[perm] = y
        return x

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return a dictionary of compression statistics.

        Returns
        -------
        dict with keys:
            stored_blocks : int
                Number of leaf blocks in the partition.
            low_rank : int
                Number of admissible (low-rank) blocks.
            dense : int
                Number of inadmissible (dense) blocks.
            memory_entries : int
                Total stored floating-point entries.
            symmetric : bool
                Whether symmetric storage is active.
            effective_blocks : int
                (symmetric only) Counted with mirrored off-diagonal blocks.
        """
        blocks = self.block_tree.blocks
        n_lr = sum(bl.kind == "lr" for bl in blocks)
        n_dn = len(blocks) - n_lr
        d = {
            "stored_blocks": len(blocks),
            "low_rank": n_lr,
            "dense": n_dn,
            "memory_entries": self.block_tree.memory(),
            "symmetric": self.symmetric,
        }
        if self.symmetric:
            n_off = sum(bl.row is not bl.col for bl in blocks)
            d["effective_blocks"] = len(blocks) + n_off  # each off-diag counts twice
        return d

    def visualize(self, save_path=None, dpi=300):
        """Plot the H-matrix block structure.

        Low-rank blocks are shown in blue, dense blocks in orange.  The axes
        use the cluster leaf ordering so spatially close indices appear
        contiguously.

        Parameters
        ----------
        save_path : str or None
            If given, save the figure to this path (PNG/JPG auto-detected).
        dpi : int
            Resolution for raster output.
        """
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
            if self.symmetric and bl.row is not bl.col:
                ax.add_patch(mpatches.Rectangle(
                    (r0, c0), h, w,
                    facecolor=face, edgecolor="black", linewidth=0.3, alpha=0.4,
                ))

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
            if save_path.endswith((".png", ".jpg")):
                fig.savefig(save_path, dpi=dpi)
            else:
                fig.savefig(save_path)

        plt.show()

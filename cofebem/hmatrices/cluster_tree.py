import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Cluster:
    """A node in a binary cluster tree.

    Attributes
    ----------
    idx : ndarray of int
        Indices of the points belonging to this cluster.
    bbox : tuple[ndarray, ndarray]
        Axis-aligned bounding box as ``(lo, hi)`` arrays of shape ``(d,)``.
    level : int
        Depth of this node (root is 0).
    parent : Cluster or None
        Parent node; ``None`` for the root.
    left : Cluster or None
        Left child; ``None`` for leaf nodes.
    right : Cluster or None
        Right child; ``None`` for leaf nodes.
    cid : int
        DFS pre-order identifier assigned by :meth:`ClusterTree._assign_cids`.
    """

    idx: np.ndarray
    bbox: Tuple[np.ndarray, np.ndarray]
    level: int
    parent: Optional["Cluster"] = None
    left: Optional["Cluster"] = None
    right: Optional["Cluster"] = None
    cid: int = 0  # DFS preorder ID, set by ClusterTree._assign_cids()

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` if this cluster has no children."""
        return self.left is None and self.right is None

    @property
    def diam(self) -> float:
        """Return the diameter of the bounding box (maximum side length)."""
        return float((self.bbox[1] - self.bbox[0]).max())


class ClusterTree:
    """Binary space-partitioning tree over a point set.

    Recursively splits the point set into two balanced halves until each
    cluster contains at most ``leaf_size`` points.  Splitting can be done
    either along the longest axis (``"kd"``) or along the principal component
    (``"pca"``).

    Parameters
    ----------
    pts : ndarray of shape (n, d)
        Point coordinates.
    leaf_size : int
        Maximum number of points in a leaf cluster.
    split : {"pca", "kd"}
        Splitting strategy.  ``"pca"`` uses the direction of maximum variance;
        ``"kd"`` splits along the longest bounding-box axis.

    Attributes
    ----------
    pts : ndarray
        The original point array.
    root : Cluster
        Root node of the tree.
    """

    def __init__(self, pts: np.ndarray, leaf_size: int = 64, split: str = "pca"):
        self.pts = pts
        self.leaf = int(leaf_size)
        self.split = split
        self.root = self._build(np.arange(len(pts)), 0, None)
        self._assign_cids()

    def _assign_cids(self):
        """Assign DFS pre-order integer IDs to every cluster node."""
        cid = [0]

        def _dfs(cl: Cluster):
            cl.cid = cid[0]
            cid[0] += 1
            if not cl.is_leaf:
                _dfs(cl.left)
                _dfs(cl.right)

        _dfs(self.root)

    def _build(self, idx: np.ndarray, level: int, parent: Optional[Cluster]):
        """Recursively build the cluster tree starting from *idx*."""
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
        """Return point indices in the order they appear in the leaves (DFS).

        Returns
        -------
        list of int
            Permutation of ``range(n)`` that groups spatially close points.
        """
        order: List[int] = []

        def _dfs(cl: Cluster):
            if cl.is_leaf:
                order.extend(cl.idx.tolist())
            else:
                _dfs(cl.left)
                _dfs(cl.right)

        _dfs(self.root)
        return order

    def _max_level(self):
        """Return the maximum depth (level) of any leaf in the tree."""
        def depth(cl: Cluster):
            return cl.level if cl.is_leaf else max(depth(cl.left), depth(cl.right))

        return depth(self.root)

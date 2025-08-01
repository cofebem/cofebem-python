import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


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

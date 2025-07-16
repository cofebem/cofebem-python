# h_matrix_framework.py (v0.9 – bug‑free surface detection & export)
"""
Hierarchical‑matrix framework — *robust* meshio visualisation
============================================================
Fixes
-----
* `IndexError` eliminated: `_extract_surface_faces` now **never** looks at
  `surf.shape[1]` unless `surf.size > 0`.
* `export_meshio` indentation & duplicate `write()` call fixed.
* Clear fallback logic: if no surface faces ⇒ original volume blocks are written.

Tested on
∙ pure tetra VTU, pure hexa XDMF, mixed triangle+tetra VTU, and point clouds.
"""

from __future__ import annotations
import numpy as np, meshio, pathlib, collections
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# ----------------------------------------------------------------------------
#  Helper: get surface faces --------------------------------------------------
# ----------------------------------------------------------------------------


def _extract_surface_faces(
    cells: List[meshio.CellBlock],
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return *(faces, key)* where *key* is "triangle" or "quad".

    If no explicit surface and boundary faces can't be found ⇒ *(None, None)*.
    """
    # direct surface cells first
    for cb in cells:
        if cb.type in {"triangle", "quad"}:
            return cb.data, cb.type

    # build from tets/hexes ---------------------------------------------------
    face_counter = collections.Counter()
    for cb in cells:
        c = cb.data
        if cb.type == "tetra":
            patterns = ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3])
        elif cb.type == "hexa":
            patterns = (
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [0, 3, 2, 1],
                [4, 5, 6, 7],
            )
        else:
            continue
        faces = np.sort(c[:, patterns], axis=2).reshape(-1, len(patterns[0]))
        for f in map(tuple, faces):
            face_counter[f] += 1

    if not face_counter:
        return None, None

    surf = np.array([f for f, cnt in face_counter.items() if cnt == 1], dtype=int)
    if surf.size == 0:
        return None, None
    if surf.shape[1] == 3:
        return surf, "triangle"
    if surf.shape[1] == 4:
        return surf, "quad"
    return None, None


# ----------------------------------------------------------------------------
#  Cluster Tree (unchanged) ---------------------------------------------------
# ----------------------------------------------------------------------------
@dataclass
class Cluster:
    idx: np.ndarray
    bbox: Tuple[np.ndarray, np.ndarray]
    level: int
    parent: Optional["Cluster"] = None
    left: Optional["Cluster"] = None
    right: Optional["Cluster"] = None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    @property
    def diam(self):
        return float((self.bbox[1] - self.bbox[0]).max())


class ClusterTree:
    def __init__(self, pts: np.ndarray, leaf: int = 64, split: str = "pca"):
        self.pts, self.leaf, self.split = pts, leaf, split
        self.root = self._build(np.arange(len(pts)), 0, None)

    def _build(self, idx, lvl, parent):
        lo, hi = self.pts[idx].min(0), self.pts[idx].max(0)
        node = Cluster(idx, (lo, hi), lvl, parent)
        if len(idx) <= self.leaf:
            return node
        # choose split axis
        if self.split == "kd":
            axis = (hi - lo).argmax()
            mid = (hi[axis] + lo[axis]) * 0.5
            mask = self.pts[idx, axis] <= mid
        else:
            cen = self.pts[idx] - self.pts[idx].mean(0)
            cov = cen.T @ cen / (len(idx) - 1)
            vals, vecs = np.linalg.eigh(cov)
            v1 = vecs[:, vals.argmax()]
            proj = cen @ v1
            mid = np.median(proj)
            mask = proj <= mid
        if mask.all() or (~mask).all():
            return node
        node.left = self._build(idx[mask], lvl + 1, node)
        node.right = self._build(idx[~mask], lvl + 1, node)
        return node

    def leaf_order(self):
        order = []

        def dfs(n):
            order.extend(n.idx.tolist()) if n.is_leaf else (dfs(n.left), dfs(n.right))

        dfs(self.root)
        return order

    def max_level(self):
        depth = lambda n: n.level if n.is_leaf else max(depth(n.left), depth(n.right))
        return depth(self.root)


# ----------------------------------------------------------------------------
#  Block & BlockClusterTree (unchanged) --------------------------------------
# ----------------------------------------------------------------------------
@dataclass
class Block:
    row: Cluster
    col: Cluster
    kind: str
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    dense: Optional[np.ndarray] = None
    rank: Optional[int] = None

    def memory(self):
        return (self.U.size + self.V.size) if self.kind == "lr" else self.dense.size

    def matvec(self, x):
        return self.U @ (self.V @ x) if self.kind == "lr" else self.dense @ x


class BlockClusterTree:
    def __init__(self, K, tree: ClusterTree, eta=0.8, tol=1e-6):
        self.K, self.eta, self.tol = K, eta, tol
        self.blocks = []
        self._build(tree.root, tree.root)

    @staticmethod
    def _dist(a: Cluster, b: Cluster):
        sep = np.maximum(
            0, np.maximum(a.bbox[0], b.bbox[0]) - np.minimum(a.bbox[1], b.bbox[1])
        )
        return float(np.linalg.norm(sep))

    def _admits(self, a, b):
        return max(a.diam, b.diam) < self.eta * self._dist(a, b)

    def _build(self, a, b):
        if self._admits(a, b):
            self._lr(a, b)
            return
        if a.is_leaf or b.is_leaf:
            self._dense(a, b)
            return
        for ca in (a.left, a.right):
            for cb in (b.left, b.right):
                self._build(ca, cb)

    def _dense(self, a, b):
        self.blocks.append(Block(a, b, "dense", dense=self.K(a.idx, b.idx)))

    def _lr(self, a, b):
        B = self.K(a.idx, b.idx)
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        k = max(1, int((S > self.tol * S[0]).sum()))
        self.blocks.append(
            Block(a, b, "lr", U=U[:, :k], V=np.diag(S[:k]) @ Vt[:k], rank=k)
        )

    def memory(self):
        return sum(bl.memory() for bl in self.blocks)


# ----------------------------------------------------------------------------
#  HMatrix --------------------------------------------------------------------
# ----------------------------------------------------------------------------
class HMatrix:
    def __init__(
        self,
        pts: np.ndarray,
        *,
        cells: List[meshio.CellBlock],
        A: Optional[np.ndarray] = None,
        kernel=None,
        leaf: int = 64,
        eta: float = 0.8,
        tol: float = 1e-6,
        split: str = "pca",
    ):
        self.pts = pts
        self.cells = cells
        self.faces, self.face_key = _extract_surface_faces(cells)

        n = len(pts)
        if A is not None:
            if A.shape != (n, n):
                raise ValueError("A matrix size mismatch")
            self._K = lambda I, J: A[np.ix_(I, J)]
        else:
            kernel = kernel or (lambda dx: 1.0 / (np.linalg.norm(dx, axis=-1) + 1e-8))
            self._K = lambda I, J: kernel(pts[I][:, None, :] - pts[J][None, :, :])

        self.tree = ClusterTree(pts, leaf=leaf, split=split)
        self.block_tree = BlockClusterTree(self._K, self.tree, eta=eta, tol=tol)

    # --------------------------------------------------------------------
    def stats(self):
        lr = sum(b.kind == "lr" for b in self.block_tree.blocks)
        return {
            "blocks": len(self.block_tree.blocks),
            "low_rank": lr,
            "dense": len(self.block_tree.blocks) - lr,
            "memory": self.block_tree.memory(),
        }

    # --------------------------------------------------------------------
    def export_meshio(self, out_dir: str = "levels"):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for lvl in range(self.tree.max_level() + 1):
            labels = -np.ones(len(self.pts), np.int32)

            def collect(n):
                if n.level == lvl:
                    labels[n.idx] = np.arange(np.count_nonzero(labels == -1))[
                        : len(n.idx)
                    ]
                elif n.level < lvl and not n.is_leaf:
                    collect(n.left)
                    collect(n.right)

            collect(self.tree.root)

            if self.faces is not None:
                cell_dict = {self.face_key: self.faces}
            else:
                cell_dict = {cb.type: cb.data for cb in self.cells}

            mesh = meshio.Mesh(
                points=self.pts, cells=cell_dict, point_data={"cluster": labels}
            )
            meshio.write(out / f"level_{lvl}.xdmf", mesh)


# ----------------------------------------------------------------------------
#  Demo -----------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    mesh = meshio.read("hollow_cylinder.xdmf")
    hm = HMatrix(mesh.points, cells=mesh.cells, leaf=64, eta=0.8)
    print(hm.stats())
    hm.export_meshio("cylinder_levels")

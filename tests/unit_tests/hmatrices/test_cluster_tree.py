"""Tests for Cluster and ClusterTree.

Invariants verified:
- All n points appear exactly once across the leaves.
- Every cluster's bounding box contains all its points.
- DFS pre-order IDs are unique and start at 0.
- Leaf order is a permutation of range(n).
- Leaf sizes do not exceed leaf_size.
- Both "pca" and "kd" splits work.
"""

from __future__ import annotations

import numpy as np
import pytest

from cofebem.hmatrices.cluster_tree import Cluster, ClusterTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_leaf_indices(root: Cluster):
    """Return the union of all leaf idx arrays (as a sorted list)."""
    result = []

    def _dfs(cl):
        if cl.is_leaf:
            result.extend(cl.idx.tolist())
        else:
            _dfs(cl.left)
            _dfs(cl.right)

    _dfs(root)
    return sorted(result)


def all_clusters(root: Cluster):
    """Yield every cluster node in DFS pre-order."""
    yield root
    if not root.is_leaf:
        yield from all_clusters(root.left)
        yield from all_clusters(root.right)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pts_1d():
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 1))


@pytest.fixture
def pts_2d():
    rng = np.random.default_rng(7)
    return rng.standard_normal((300, 2))


@pytest.fixture
def pts_3d():
    rng = np.random.default_rng(99)
    return rng.standard_normal((400, 3))


@pytest.fixture
def pts_small():
    """Only 5 points — should produce a single-leaf tree."""
    return np.arange(5).reshape(5, 1).astype(float)


# ---------------------------------------------------------------------------
# Cluster dataclass
# ---------------------------------------------------------------------------

class TestCluster:
    def test_is_leaf_true_when_no_children(self):
        lo = np.array([0.0])
        hi = np.array([1.0])
        cl = Cluster(idx=np.array([0, 1]), bbox=(lo, hi), level=0)
        assert cl.is_leaf

    def test_is_leaf_false_with_children(self):
        lo, hi = np.array([0.0]), np.array([1.0])
        left = Cluster(idx=np.array([0]), bbox=(lo, hi), level=1)
        right = Cluster(idx=np.array([1]), bbox=(lo, hi), level=1)
        parent = Cluster(idx=np.array([0, 1]), bbox=(lo, hi), level=0,
                         left=left, right=right)
        assert not parent.is_leaf

    def test_diam_single_dimension(self):
        cl = Cluster(idx=np.array([0]), bbox=(np.array([1.0, 2.0]), np.array([4.0, 3.0])), level=0)
        assert cl.diam == pytest.approx(3.0)

    def test_diam_zero_for_point(self):
        p = np.array([1.0, 2.0])
        cl = Cluster(idx=np.array([0]), bbox=(p, p), level=0)
        assert cl.diam == 0.0


# ---------------------------------------------------------------------------
# ClusterTree — structural invariants
# ---------------------------------------------------------------------------

class TestClusterTreeStructure:
    @pytest.mark.parametrize("split", ["pca", "kd"])
    def test_leaves_cover_all_points(self, pts_2d, split):
        n = len(pts_2d)
        ct = ClusterTree(pts_2d, leaf_size=32, split=split)
        assert collect_leaf_indices(ct.root) == list(range(n))

    @pytest.mark.parametrize("split", ["pca", "kd"])
    def test_leaf_order_is_permutation(self, pts_2d, split):
        n = len(pts_2d)
        ct = ClusterTree(pts_2d, leaf_size=32, split=split)
        order = ct.leaf_order()
        assert sorted(order) == list(range(n))
        assert len(order) == n

    def test_leaf_size_respected(self, pts_2d):
        leaf_size = 20
        ct = ClusterTree(pts_2d, leaf_size=leaf_size)
        for cl in all_clusters(ct.root):
            if cl.is_leaf:
                assert len(cl.idx) <= leaf_size

    def test_single_leaf_when_small(self, pts_small):
        ct = ClusterTree(pts_small, leaf_size=64)
        assert ct.root.is_leaf
        assert collect_leaf_indices(ct.root) == list(range(len(pts_small)))

    def test_root_level_is_zero(self, pts_2d):
        ct = ClusterTree(pts_2d, leaf_size=32)
        assert ct.root.level == 0

    def test_child_level_is_parent_plus_one(self, pts_2d):
        ct = ClusterTree(pts_2d, leaf_size=32)
        for cl in all_clusters(ct.root):
            if not cl.is_leaf:
                assert cl.left.level == cl.level + 1
                assert cl.right.level == cl.level + 1

    def test_bbox_contains_all_points(self, pts_2d):
        ct = ClusterTree(pts_2d, leaf_size=32)
        for cl in all_clusters(ct.root):
            lo, hi = cl.bbox
            pts_in = ct.pts[cl.idx]
            assert np.all(pts_in >= lo - 1e-12)
            assert np.all(pts_in <= hi + 1e-12)

    @pytest.mark.parametrize("split", ["pca", "kd"])
    def test_cids_are_unique_and_start_at_zero(self, pts_2d, split):
        ct = ClusterTree(pts_2d, leaf_size=32, split=split)
        cids = [cl.cid for cl in all_clusters(ct.root)]
        assert min(cids) == 0
        assert len(set(cids)) == len(cids)

    def test_works_on_1d_points(self, pts_1d):
        ct = ClusterTree(pts_1d, leaf_size=16)
        assert collect_leaf_indices(ct.root) == list(range(len(pts_1d)))

    def test_works_on_3d_points(self, pts_3d):
        ct = ClusterTree(pts_3d, leaf_size=32)
        assert collect_leaf_indices(ct.root) == list(range(len(pts_3d)))

    def test_max_level_nonnegative(self, pts_2d):
        ct = ClusterTree(pts_2d, leaf_size=32)
        assert ct._max_level() >= 0

    def test_max_level_zero_for_single_leaf(self, pts_small):
        ct = ClusterTree(pts_small, leaf_size=64)
        assert ct._max_level() == 0

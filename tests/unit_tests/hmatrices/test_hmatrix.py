"""Tests for BlockClusterTree, Block, and HMatrix.

Test strategy
-------------
- Use a 1D exponential-kernel matrix.  With n=128 uniformly spaced points and
  leaf_size=16, clusters that are 3+ leaves apart satisfy the eta=0.8
  admissibility criterion, so admissible (low-rank) blocks are guaranteed.
- Use a small SPD matrix to test solve() accuracy.
- Check arithmetic operators (+, *, neg, sub) return consistent results.
- Check stats() keys and memory() footprint are sensible.
"""

from __future__ import annotations

import numpy as np
import pytest

from cofebem.hmatrices import ClusterTree, HMatrix, IndexedEntrySource
from cofebem.hmatrices.block_cluster_tree import Block, BlockClusterTree, _block_add


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_1d_problem(n=128):
    """1D exponential kernel: guaranteed admissible blocks for leaf_size=16, eta=0.8."""
    pts = np.linspace(0, 1, n).reshape(-1, 1)
    diff = np.abs(pts[:, 0, None] - pts[:, 0][None, :])
    A = np.exp(-5.0 * diff)
    A += n * np.eye(n)
    return pts, A


class _CountingEntrySource:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape
        self.queries = []

    def get_block(self, rows, columns):
        self.queries.append((len(rows), len(columns)))
        return self.matrix[np.ix_(rows, columns)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def problem_1d():
    return _build_1d_problem(n=128)


@pytest.fixture(scope="module")
def hmatrix_default(problem_1d):
    pts, A = problem_1d
    return HMatrix(pts, A, leaf_size=16, tol=1e-6, lr_approx="aca_partial")


@pytest.fixture(scope="module")
def hmatrix_sym(problem_1d):
    pts, A = problem_1d
    return HMatrix(pts, A, leaf_size=16, tol=1e-6, lr_approx="aca_partial", symmetric=True)


@pytest.fixture(scope="module")
def small_spd():
    rng = np.random.default_rng(5)
    n = 32
    pts = np.linspace(0, 1, n).reshape(-1, 1)
    diff = np.abs(pts[:, 0, None] - pts[:, 0][None, :])
    A = np.exp(-3.0 * diff) + n * np.eye(n)
    return pts, A


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------

class TestBlock:
    def test_shape_matches_cluster_sizes(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        for bl in bct.blocks:
            assert bl.shape == (len(bl.row.idx), len(bl.col.idx))

    def test_dense_block_to_dense_roundtrip(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        dense_blocks = [bl for bl in bct.blocks if bl.kind == "dense"]
        assert dense_blocks, "expected at least one dense block"
        bl = dense_blocks[0]
        np.testing.assert_array_equal(bl.to_dense(), bl.dense)

    def test_lr_block_to_dense_accuracy(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        lr_blocks = [bl for bl in bct.blocks if bl.kind == "lr"]
        assert lr_blocks, "expected at least one low-rank block"
        for bl in lr_blocks[:5]:
            exact = A[np.ix_(bl.row.idx, bl.col.idx)]
            norm_exact = np.linalg.norm(exact, "fro")
            if norm_exact < 1e-14:
                continue
            rel = np.linalg.norm(exact - bl.to_dense(), "fro") / norm_exact
            assert rel < 1e-3

    def test_matvec_consistent_with_to_dense(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        rng = np.random.default_rng(42)
        for bl in bct.blocks[:10]:
            n = bl.shape[1]
            x = rng.standard_normal(n)
            np.testing.assert_allclose(bl.matvec(x), bl.to_dense() @ x, atol=1e-12)

    def test_matvec_T_consistent_with_to_dense(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        rng = np.random.default_rng(43)
        for bl in bct.blocks[:10]:
            m = bl.shape[0]
            x = rng.standard_normal(m)
            np.testing.assert_allclose(bl.matvec_T(x), bl.to_dense().T @ x, atol=1e-12)

    def test_block_memory_positive(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        for bl in bct.blocks:
            assert bl.memory() > 0


# ---------------------------------------------------------------------------
# _block_add
# ---------------------------------------------------------------------------

class TestBlockAdd:
    def test_lr_plus_lr_accuracy(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        lr_blocks = [bl for bl in bct.blocks if bl.kind == "lr"]
        assert lr_blocks, "expected at least one low-rank block"
        bl = lr_blocks[0]
        result = _block_add(bl, bl, tol=1e-6)
        expected = 2 * bl.to_dense()
        np.testing.assert_allclose(result.to_dense(), expected, rtol=1e-4)

    def test_dense_plus_dense(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        dense_blocks = [bl for bl in bct.blocks if bl.kind == "dense"]
        assert dense_blocks
        bl = dense_blocks[0]
        result = _block_add(bl, bl, tol=1e-6)
        np.testing.assert_allclose(result.to_dense(), 2 * bl.to_dense())


# ---------------------------------------------------------------------------
# BlockClusterTree
# ---------------------------------------------------------------------------

class TestBlockClusterTree:
    def test_all_blocks_cover_matrix(self, problem_1d):
        pts, A = problem_1d
        n = len(pts)
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        covered = np.zeros((n, n), dtype=bool)
        for bl in bct.blocks:
            covered[np.ix_(bl.row.idx, bl.col.idx)] = True
        assert covered.all(), "not all matrix entries are covered by blocks"

    def test_blocks_do_not_overlap(self, problem_1d):
        pts, A = problem_1d
        n = len(pts)
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        count = np.zeros((n, n), dtype=int)
        for bl in bct.blocks:
            count[np.ix_(bl.row.idx, bl.col.idx)] += 1
        assert count.max() == 1, "some entries covered by more than one block"

    def test_memory_positive(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        assert bct.memory() > 0

    def test_memory_less_than_dense(self, problem_1d):
        pts, A = problem_1d
        n = len(pts)
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        assert bct.memory() < n * n

    @pytest.mark.parametrize("method", ["aca_partial", "aca_full", "aca_plus", "truncated_svd"])
    def test_all_lr_methods_build_successfully(self, problem_1d, method):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-4, lr_approx=method)
        assert len(bct.blocks) > 0

    def test_unknown_lr_method_raises(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        with pytest.raises(ValueError, match="Unknown"):
            BlockClusterTree(ct, ct, A, lr_approx="bogus")

    def test_has_both_lr_and_dense_blocks(self, problem_1d):
        pts, A = problem_1d
        ct = ClusterTree(pts, leaf_size=16)
        bct = BlockClusterTree(ct, ct, A, tol=1e-6)
        kinds = {bl.kind for bl in bct.blocks}
        assert "lr" in kinds, "expected some low-rank blocks"
        assert "dense" in kinds, "expected some dense blocks"


# ---------------------------------------------------------------------------
# HMatrix — construction
# ---------------------------------------------------------------------------

class TestHMatrixConstruction:
    def test_shape_mismatch_raises(self, problem_1d):
        pts, A = problem_1d
        with pytest.raises(ValueError, match="n,n"):
            HMatrix(pts, A[:10, :10])

    def test_stats_keys_present(self, hmatrix_default):
        s = hmatrix_default.stats()
        for key in ("stored_blocks", "low_rank", "dense", "memory_entries", "symmetric"):
            assert key in s

    def test_stats_block_counts_add_up(self, hmatrix_default):
        s = hmatrix_default.stats()
        assert s["low_rank"] + s["dense"] == s["stored_blocks"]

    def test_stats_memory_less_than_dense(self, hmatrix_default, problem_1d):
        _, A = problem_1d
        n = A.shape[0]
        assert hmatrix_default.stats()["memory_entries"] < n * n

    def test_stats_has_lr_blocks(self, hmatrix_default):
        assert hmatrix_default.stats()["low_rank"] > 0

    def test_entry_source_uses_only_block_and_aca_cross_queries(self, problem_1d):
        pts, A = problem_1d
        source = _CountingEntrySource(A)
        H = HMatrix.from_entry_source(
            pts,
            source,
            leaf_size=16,
            tol=1.0e-8,
            lr_approx="aca_partial",
            symmetric=True,
        )

        assert H.stats()["low_rank"] > 0
        assert source.queries
        assert (len(pts), len(pts)) not in source.queries
        assert any(rows == 1 or columns == 1 for rows, columns in source.queries)
        rng = np.random.default_rng(99)
        x = rng.standard_normal(len(pts))
        np.testing.assert_allclose(H @ x, A @ x, rtol=1.0e-5, atol=1.0e-7)

    def test_entry_source_rejects_dense_only_low_rank_method(self, problem_1d):
        pts, A = problem_1d
        source = _CountingEntrySource(A)
        with pytest.raises(ValueError, match="aca_partial"):
            HMatrix.from_entry_source(pts, source, lr_approx="truncated_svd")

    def test_indexed_entry_source_builds_only_selected_principal_matrix(self):
        points = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
        matrix = 2.0 * np.eye(12) + np.exp(
            -np.abs(points[:, 0, None] - points[:, 0][None, :])
        )
        source = _CountingEntrySource(matrix)
        indices = np.array([0, 2, 3, 7, 9, 11])
        restricted = IndexedEntrySource(source, indices)
        block = restricted.get_block(np.array([1, 4]), np.array([0, 3, 5]))

        np.testing.assert_allclose(
            block, matrix[np.ix_(indices[[1, 4]], indices[[0, 3, 5]])]
        )
        assert restricted.shape == (len(indices), len(indices))
        hmatrix = HMatrix.from_entry_source(
            points[indices],
            restricted,
            leaf_size=2,
            tol=1.0e-10,
            lr_approx="aca_partial",
            symmetric=True,
        )
        vector = np.linspace(0.5, 1.5, len(indices))
        np.testing.assert_allclose(
            hmatrix @ vector,
            matrix[np.ix_(indices, indices)] @ vector,
            rtol=1.0e-8,
            atol=1.0e-10,
        )


# ---------------------------------------------------------------------------
# HMatrix — matvec
# ---------------------------------------------------------------------------

class TestHMatrixMatvec:
    def test_matvec_accuracy(self, hmatrix_default, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(10)
        x = rng.standard_normal(len(pts))
        y_exact = A @ x
        y_hmat = hmatrix_default @ x
        rel = np.linalg.norm(y_exact - y_hmat) / np.linalg.norm(y_exact)
        assert rel < 1e-4

    def test_matvec_size_mismatch_raises(self, hmatrix_default, problem_1d):
        pts, _ = problem_1d
        n = len(pts)
        with pytest.raises(ValueError, match="size mismatch"):
            _ = hmatrix_default @ np.ones(n + 1)

    def test_symmetric_matvec_accuracy(self, hmatrix_sym, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(11)
        x = rng.standard_normal(len(pts))
        y_exact = A @ x
        y_sym = hmatrix_sym @ x
        rel = np.linalg.norm(y_exact - y_sym) / np.linalg.norm(y_exact)
        assert rel < 1e-4


# ---------------------------------------------------------------------------
# HMatrix — to_dense
# ---------------------------------------------------------------------------

class TestHMatrixToDense:
    def test_to_dense_accuracy(self, hmatrix_default, problem_1d):
        _, A = problem_1d
        D = hmatrix_default.to_dense()
        rel = np.linalg.norm(A - D, "fro") / np.linalg.norm(A, "fro")
        assert rel < 1e-4

    def test_to_dense_shape(self, hmatrix_default, problem_1d):
        pts, _ = problem_1d
        n = len(pts)
        D = hmatrix_default.to_dense()
        assert D.shape == (n, n)


# ---------------------------------------------------------------------------
# HMatrix — arithmetic
# ---------------------------------------------------------------------------

class TestHMatrixArithmetic:
    def test_scalar_mul(self, hmatrix_default, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(20)
        x = rng.standard_normal(len(pts))
        y_scaled = (3.0 * hmatrix_default) @ x
        y_exact = 3.0 * A @ x
        rel = np.linalg.norm(y_exact - y_scaled) / np.linalg.norm(y_exact)
        assert rel < 1e-4

    def test_rmul_equals_mul(self, hmatrix_default, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(21)
        x = rng.standard_normal(len(pts))
        y1 = (2.0 * hmatrix_default) @ x
        y2 = (hmatrix_default * 2.0) @ x
        np.testing.assert_allclose(y1, y2, atol=1e-14)

    def test_neg(self, hmatrix_default, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(22)
        x = rng.standard_normal(len(pts))
        y_neg = (-hmatrix_default) @ x
        y_exact = -A @ x
        rel = np.linalg.norm(y_exact - y_neg) / np.linalg.norm(y_exact)
        assert rel < 1e-4

    def test_add(self, hmatrix_default, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(23)
        x = rng.standard_normal(len(pts))
        H2 = hmatrix_default + hmatrix_default
        y_sum = H2 @ x
        y_exact = 2 * A @ x
        rel = np.linalg.norm(y_exact - y_sum) / np.linalg.norm(y_exact)
        assert rel < 1e-4

    def test_sub(self, hmatrix_default, problem_1d):
        pts, A = problem_1d
        rng = np.random.default_rng(24)
        x = rng.standard_normal(len(pts))
        H_zero = hmatrix_default - hmatrix_default
        y = H_zero @ x
        np.testing.assert_allclose(y, 0.0, atol=1e-8)

    def test_add_different_tree_raises(self, problem_1d):
        pts, A = problem_1d
        H1 = HMatrix(pts, A, leaf_size=16, tol=1e-6)
        H2 = HMatrix(pts, A, leaf_size=16, tol=1e-6)
        with pytest.raises(ValueError, match="shared cluster tree"):
            _ = H1 + H2

    def test_add_non_hmatrix_returns_notimplemented(self, hmatrix_default):
        result = hmatrix_default.__add__(42)
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# HMatrix — solve
# ---------------------------------------------------------------------------

class TestHMatrixSolve:
    def test_solve_accuracy(self, small_spd):
        pts, A = small_spd
        H = HMatrix(pts, A, leaf_size=8, tol=1e-8, lr_approx="aca_partial")
        rng = np.random.default_rng(30)
        b = rng.standard_normal(len(pts))
        x = H.solve(b)
        np.testing.assert_allclose(A @ x, b, atol=1e-6)

    def test_solve_calls_lu_if_not_cached(self, small_spd):
        pts, A = small_spd
        H = HMatrix(pts, A, leaf_size=8, tol=1e-8)
        assert H._lu_cache is None
        b = np.ones(len(pts))
        _ = H.solve(b)
        assert H._lu_cache is not None

    def test_lu_returns_self(self, small_spd):
        pts, A = small_spd
        H = HMatrix(pts, A, leaf_size=8, tol=1e-8)
        result = H.lu()
        assert result is H

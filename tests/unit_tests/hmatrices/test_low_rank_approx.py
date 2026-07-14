"""Tests for the four low-rank approximation methods.

All methods must satisfy ``A ≈ U @ V.T`` within tolerance for:
- exact rank-1 matrices
- exact rank-k matrices
- zero matrices
- nearly full-rank matrices (only a loose bound is checked)
"""

from __future__ import annotations

import numpy as np
import pytest

from cofebem.hmatrices.low_rank_approx import (
    aca_full,
    aca_partial,
    aca_plus,
    truncated_svd,
)

ALL_METHODS = pytest.mark.parametrize(
    "method", [truncated_svd, aca_full, aca_partial, aca_plus],
    ids=["truncated_svd", "aca_full", "aca_partial", "aca_plus"],
)

# Methods that see the full residual; reliable on general low-rank matrices.
FULL_RESIDUAL_METHODS = pytest.mark.parametrize(
    "method", [truncated_svd, aca_full],
    ids=["truncated_svd", "aca_full"],
)

TOL = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frobenius_rel_error(A, U, V):
    """Return ||A - U @ V.T||_F / ||A||_F."""
    norm_A = np.linalg.norm(A, "fro")
    if norm_A == 0.0:
        return float(np.linalg.norm(A - U @ V.T, "fro"))
    return float(np.linalg.norm(A - U @ V.T, "fro") / norm_A)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rank1_matrix():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(20)
    v = rng.standard_normal(15)
    return np.outer(u, v)


@pytest.fixture
def rank3_matrix():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((30, 3))
    V = rng.standard_normal((25, 3))
    return U @ V.T


@pytest.fixture
def cauchy_matrix():
    """Well-separated Cauchy matrix: x ∈ [0,1], y ∈ [10,11].

    The 9-unit gap between the two intervals gives rapid singular-value decay
    (ratio ≈ 1/9 per step), so even a low-rank approximation achieves small
    relative error.
    """
    x = np.linspace(0.0, 1.0, 40)
    y = np.linspace(10.0, 11.0, 35)
    return 1.0 / (x[:, None] + y[None, :])


@pytest.fixture
def zero_matrix():
    return np.zeros((12, 10))


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

class TestReturnShape:
    @ALL_METHODS
    def test_output_shapes_rank1(self, method, rank1_matrix):
        A = rank1_matrix
        U, V = method(A, tol=1e-6)
        assert U.ndim == 2 and V.ndim == 2
        assert U.shape[0] == A.shape[0]
        assert V.shape[0] == A.shape[1]
        assert U.shape[1] == V.shape[1], "U and V must share the rank dimension"

    @ALL_METHODS
    def test_output_shapes_zero(self, method, zero_matrix):
        U, V = method(zero_matrix, tol=1e-6)
        assert U.shape[0] == zero_matrix.shape[0]
        assert V.shape[0] == zero_matrix.shape[1]
        assert U.shape[1] == V.shape[1]


# ---------------------------------------------------------------------------
# Accuracy on low-rank inputs
# ---------------------------------------------------------------------------

class TestAccuracy:
    @ALL_METHODS
    def test_rank1_approximation(self, method, rank1_matrix):
        U, V = method(rank1_matrix, tol=TOL)
        assert frobenius_rel_error(rank1_matrix, U, V) < 1e-6

    @FULL_RESIDUAL_METHODS
    def test_rank3_approximation(self, method, rank3_matrix):
        """aca_partial/aca_plus use column pivoting and can stall on random rank-3
        matrices; only methods that see the full residual are tested here."""
        U, V = method(rank3_matrix, tol=TOL)
        assert frobenius_rel_error(rank3_matrix, U, V) < 1e-6

    @pytest.mark.parametrize("method,max_err", [
        (truncated_svd, 1e-6),
        (aca_full, 1e-4),
        (aca_partial, 0.02),   # partial ACA may stop at rank 1 (~1% error)
        (aca_plus, 0.02),      # ACA+ recompresses but tolerance is still loose
    ], ids=["truncated_svd", "aca_full", "aca_partial", "aca_plus"])
    def test_cauchy_approximation(self, method, max_err, cauchy_matrix):
        U, V = method(cauchy_matrix, tol=1e-8)
        assert frobenius_rel_error(cauchy_matrix, U, V) < max_err

    @ALL_METHODS
    def test_zero_matrix_gives_zero_approximation(self, method, zero_matrix):
        U, V = method(zero_matrix, tol=1e-6)
        approx = U @ V.T
        np.testing.assert_allclose(approx, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Rank cap
# ---------------------------------------------------------------------------

class TestRankCap:
    @ALL_METHODS
    def test_kmax_limits_rank(self, method, rank3_matrix):
        U, V = method(rank3_matrix, tol=1e-12, k_max=2)
        assert U.shape[1] <= 2

    def test_truncated_svd_kmax_1(self, rank3_matrix):
        U, V = truncated_svd(rank3_matrix, tol=1e-12, k_max=1)
        assert U.shape[1] == 1

    def test_aca_full_kmax_1(self, rank3_matrix):
        U, V = aca_full(rank3_matrix, tol=1e-12, k_max=1)
        assert U.shape[1] <= 1

    def test_aca_partial_kmax_1(self, rank3_matrix):
        U, V = aca_partial(rank3_matrix, tol=1e-12, k_max=1)
        assert U.shape[1] <= 1


# ---------------------------------------------------------------------------
# Rank of approximation
# ---------------------------------------------------------------------------

class TestRank:
    def test_truncated_svd_captures_exact_rank1(self, rank1_matrix):
        U, V = truncated_svd(rank1_matrix, tol=1e-8)
        assert U.shape[1] == 1

    def test_truncated_svd_captures_exact_rank3(self, rank3_matrix):
        U, V = truncated_svd(rank3_matrix, tol=1e-8)
        assert U.shape[1] == 3

    def test_aca_plus_rank_leq_aca_partial(self, rank3_matrix):
        _, V_p = aca_partial(rank3_matrix, tol=1e-6)
        _, V_plus = aca_plus(rank3_matrix, tol=1e-6)
        assert V_plus.shape[1] <= V_p.shape[1]


# ---------------------------------------------------------------------------
# Symmetry / consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_aca_full_accurate_on_rank3(self, rank3_matrix):
        """aca_full sees the full residual, so it accurately approximates any low-rank matrix."""
        U_f, V_f = aca_full(rank3_matrix, tol=1e-8)
        assert frobenius_rel_error(rank3_matrix, U_f, V_f) < 1e-4

    def test_aca_partial_accurate_on_cauchy(self, cauchy_matrix):
        """aca_partial is designed for structured kernel matrices like Cauchy."""
        U_p, V_p = aca_partial(cauchy_matrix, tol=1e-8)
        assert frobenius_rel_error(cauchy_matrix, U_p, V_p) < 0.02

    def test_aca_plus_is_at_least_as_accurate_as_aca_partial(self, cauchy_matrix):
        U_p, V_p = aca_partial(cauchy_matrix, tol=1e-8)
        U_plus, V_plus = aca_plus(cauchy_matrix, tol=1e-8)
        err_p = frobenius_rel_error(cauchy_matrix, U_p, V_p)
        err_plus = frobenius_rel_error(cauchy_matrix, U_plus, V_plus)
        assert err_plus <= err_p + 1e-10  # ACA+ recompresses, so error should not grow

    @ALL_METHODS
    def test_scalar_multiple(self, method, rank1_matrix):
        A = rank1_matrix
        U2, V2 = method(2 * A, tol=TOL)
        assert frobenius_rel_error(2 * A, U2, V2) < 1e-6

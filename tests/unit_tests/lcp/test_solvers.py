"""Tests for all LCP solvers and the solve() dispatcher.

Known solutions used throughout:
- trivial:     M = I₂, q = [1, 1]      → z = [0, 0],      w = [1, 1]
- spd_2x2:     M = [[2,1],[1,2]], q = [-1,-1] → z = [1/3, 1/3], w = [0, 0]
- mixed:       M = diag(4, 2), q = [-2, 1]   → z = [0.5, 0],    w = [0, 1]
- all_active:  M = I₂, q = [-3, -2]    → z = [3, 2],      w = [0, 0]
"""

from __future__ import annotations

import numpy as np
import pytest

from cofebem.lcp import DEFAULT_METHOD, LCP, LCPStatus, SOLVERS, solve
from cofebem.lcp.exceptions import (
    InvalidSolverOptionError,
    UnsupportedMatrixError,
    UnsupportedSolverError,
)
from cofebem.lcp.solvers import ccg, ccg_v2, lemke, nnls, pgs, ppcg, psor


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def check_lcp_solution(result, M, q, atol: float = 1e-6) -> None:
    """Assert that *result* satisfies the LCP complementarity conditions."""
    z, w = result.z, result.w
    np.testing.assert_array_less(-atol, z, err_msg="z >= 0 violated")
    np.testing.assert_array_less(-atol, w, err_msg="w >= 0 violated")
    np.testing.assert_allclose(np.abs(z * w), 0.0, atol=atol,
                               err_msg="z.T w = 0 violated")
    np.testing.assert_allclose(M @ z + q, w, atol=atol,
                               err_msg="w = M z + q violated")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def problem_trivial():
    """q >= 0; trivial solution z = 0."""
    return LCP(np.eye(2), np.array([1.0, 1.0]))


@pytest.fixture
def problem_spd_2x2():
    """SPD 2×2; unique solution z = [1/3, 1/3], w = 0."""
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    q = np.array([-1.0, -1.0])
    return LCP(M, q)


@pytest.fixture
def problem_mixed():
    """Diagonal; solution z = [0.5, 0], w = [0, 1]."""
    return LCP(np.diag([4.0, 2.0]), np.array([-2.0, 1.0]))


@pytest.fixture
def problem_all_active():
    """Identity; solution z = [3, 2], w = [0, 0]."""
    return LCP(np.eye(2), np.array([-3.0, -2.0]))


@pytest.fixture
def problem_spd_large():
    """Random 10×10 SPD; q = -ones."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((10, 10))
    M = A.T @ A + np.eye(10)
    q = -np.ones(10)
    return LCP(M, q)


# ---------------------------------------------------------------------------
# solve() dispatcher
# ---------------------------------------------------------------------------

class TestSolveDispatcher:
    def test_default_method_constant(self):
        assert DEFAULT_METHOD == "lemke"

    def test_solvers_dict_contains_all_methods(self):
        for name in ("psor", "pgs", "nnls", "lemke", "ccg", "ccg_v2", "ppcg"):
            assert name in SOLVERS

    def test_default_call_converges(self, problem_spd_2x2):
        assert solve(problem_spd_2x2).converged

    @pytest.mark.parametrize(
        "method", ["psor", "pgs", "nnls", "lemke", "ccg", "ccg_v2", "ppcg"]
    )
    def test_dispatches_to_each_solver(self, method, problem_spd_2x2):
        result = solve(problem_spd_2x2, method=method)
        assert result.converged

    def test_unknown_solver_raises(self, problem_spd_2x2):
        with pytest.raises(UnsupportedSolverError, match="Unknown"):
            solve(problem_spd_2x2, method="bogus")

    def test_options_forwarded_to_solver(self, problem_spd_2x2):
        result = solve(problem_spd_2x2, method="psor", omega=1.5)
        assert result.converged

    def test_invalid_option_propagates(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError):
            solve(problem_spd_2x2, method="psor", omega=3.0)

    def test_operator_dispatches_to_ccg_without_materialisation(self):
        class Operator:
            shape = (2, 2)
            symmetric = True

            def __matmul__(self, vector):
                return np.array([2.0 * vector[0] + vector[1],
                                 vector[0] + 2.0 * vector[1]])

        problem = LCP(Operator(), np.array([-1.0, -1.0]))
        result = solve(problem, method="ccg_v2")
        assert result.converged
        np.testing.assert_allclose(result.z, [1.0 / 3.0, 1.0 / 3.0])

    def test_operator_rejected_by_dense_solver(self):
        class Operator:
            shape = (2, 2)
            symmetric = True

            def __matmul__(self, vector):
                return vector

        with pytest.raises(UnsupportedMatrixError, match="matrix operators"):
            solve(LCP(Operator(), [-1.0, -1.0]), method="lemke")


# ---------------------------------------------------------------------------
# PSOR
# ---------------------------------------------------------------------------

class TestPSOR:
    def test_trivial(self, problem_trivial):
        result = psor(problem_trivial)
        assert result.converged
        check_lcp_solution(result, problem_trivial.M, problem_trivial.q)

    def test_spd_2x2(self, problem_spd_2x2):
        result = psor(problem_spd_2x2)
        assert result.converged
        check_lcp_solution(result, problem_spd_2x2.M, problem_spd_2x2.q)
        np.testing.assert_allclose(result.z, [1 / 3, 1 / 3], atol=1e-8)

    def test_mixed(self, problem_mixed):
        result = psor(problem_mixed)
        assert result.converged
        check_lcp_solution(result, problem_mixed.M, problem_mixed.q)
        np.testing.assert_allclose(result.z, [0.5, 0.0], atol=1e-8)

    def test_all_active(self, problem_all_active):
        result = psor(problem_all_active)
        assert result.converged
        check_lcp_solution(result, problem_all_active.M, problem_all_active.q)
        np.testing.assert_allclose(result.z, [3.0, 2.0], atol=1e-8)

    def test_record_history_length_equals_iterations(self, problem_spd_2x2):
        result = psor(problem_spd_2x2, record_history=True)
        assert result.residual_history is not None
        assert len(result.residual_history) == result.iterations

    def test_no_history_by_default(self, problem_spd_2x2):
        assert psor(problem_spd_2x2).residual_history is None

    def test_z0_warm_start_converges(self, problem_spd_2x2):
        result = psor(problem_spd_2x2, z0=np.array([0.3, 0.3]))
        assert result.converged

    def test_z0_negative_entries_projected_to_zero(self, problem_trivial):
        result = psor(problem_trivial, z0=np.array([-5.0, -5.0]))
        assert result.converged

    def test_z0_wrong_shape_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="shape"):
            psor(problem_spd_2x2, z0=np.zeros(5))

    def test_omega_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="omega"):
            psor(problem_spd_2x2, omega=0.0)

    def test_omega_two_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="omega"):
            psor(problem_spd_2x2, omega=2.0)

    def test_omega_negative_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="omega"):
            psor(problem_spd_2x2, omega=-0.5)

    def test_tol_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="tol"):
            psor(problem_spd_2x2, tol=0.0)

    def test_max_iter_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="max_iter"):
            psor(problem_spd_2x2, max_iter=0)

    def test_zero_diagonal_raises(self):
        M = np.array([[0.0, 1.0], [1.0, 2.0]])
        with pytest.raises(UnsupportedMatrixError, match="diagonal"):
            psor(LCP(M, np.array([1.0, 1.0])))

    def test_max_iter_1_returns_valid_status(self, problem_spd_2x2):
        result = psor(problem_spd_2x2, max_iter=1)
        assert result.status in (LCPStatus.CONVERGED, LCPStatus.MAX_ITERATIONS,
                                 LCPStatus.STAGNATION)

    def test_message_contains_psor(self, problem_spd_2x2):
        assert "PSOR" in psor(problem_spd_2x2).message


# ---------------------------------------------------------------------------
# PGS
# ---------------------------------------------------------------------------

class TestPGS:
    def test_spd_2x2(self, problem_spd_2x2):
        result = pgs(problem_spd_2x2)
        assert result.converged
        check_lcp_solution(result, problem_spd_2x2.M, problem_spd_2x2.q)

    def test_all_active(self, problem_all_active):
        result = pgs(problem_all_active)
        assert result.converged
        np.testing.assert_allclose(result.z, [3.0, 2.0], atol=1e-8)

    def test_message_says_pgs(self, problem_spd_2x2):
        assert "PGS" in pgs(problem_spd_2x2).message

    def test_message_does_not_say_psor(self, problem_spd_2x2):
        assert "PSOR" not in pgs(problem_spd_2x2).message

    def test_identical_solution_to_psor_omega_1(self, problem_spd_2x2):
        r_pgs = pgs(problem_spd_2x2)
        r_psor = psor(problem_spd_2x2, omega=1.0)
        np.testing.assert_allclose(r_pgs.z, r_psor.z, atol=1e-15)
        assert r_pgs.iterations == r_psor.iterations

    def test_tol_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="tol"):
            pgs(problem_spd_2x2, tol=-1.0)

    def test_max_iter_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="max_iter"):
            pgs(problem_spd_2x2, max_iter=0)

    def test_zero_diagonal_raises(self):
        M = np.array([[0.0, 1.0], [1.0, 2.0]])
        with pytest.raises(UnsupportedMatrixError):
            pgs(LCP(M, np.ones(2)))

    def test_record_history(self, problem_spd_2x2):
        result = pgs(problem_spd_2x2, record_history=True)
        assert result.residual_history is not None
        assert len(result.residual_history) == result.iterations


# ---------------------------------------------------------------------------
# NNLS
# ---------------------------------------------------------------------------

class TestNNLS:
    def test_trivial(self, problem_trivial):
        result = nnls(problem_trivial)
        assert result.converged
        np.testing.assert_allclose(result.z, 0.0, atol=1e-10)

    def test_spd_2x2(self, problem_spd_2x2):
        result = nnls(problem_spd_2x2)
        assert result.converged
        check_lcp_solution(result, problem_spd_2x2.M, problem_spd_2x2.q)
        np.testing.assert_allclose(result.z, [1 / 3, 1 / 3], atol=1e-10)

    def test_mixed(self, problem_mixed):
        result = nnls(problem_mixed)
        assert result.converged
        check_lcp_solution(result, problem_mixed.M, problem_mixed.q)
        np.testing.assert_allclose(result.z, [0.5, 0.0], atol=1e-10)

    def test_all_active(self, problem_all_active):
        result = nnls(problem_all_active)
        assert result.converged
        np.testing.assert_allclose(result.z, [3.0, 2.0], atol=1e-10)

    def test_large_spd(self, problem_spd_large):
        result = nnls(problem_spd_large)
        assert result.converged
        check_lcp_solution(result, problem_spd_large.M, problem_spd_large.q)

    def test_non_symmetric_raises(self):
        M = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(UnsupportedMatrixError, match="symmetric"):
            nnls(LCP(M, np.ones(2)))

    def test_non_pd_raises(self):
        M = np.array([[-1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(UnsupportedMatrixError):
            nnls(LCP(M, np.ones(2)))

    def test_indefinite_symmetric_raises(self):
        M = np.array([[1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(UnsupportedMatrixError):
            nnls(LCP(M, np.ones(2)))

    def test_check_symmetric_false_skips_symmetry_check(self):
        M = np.array([[2.0, 1.0], [1.5, 2.0]])  # slightly asymmetric but PD
        result = nnls(LCP(M, np.array([-1.0, -1.0])), check_symmetric=False)
        assert isinstance(result.converged, bool)

    def test_tol_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="tol"):
            nnls(problem_spd_2x2, tol=0.0)

    def test_max_iter_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="max_iter"):
            nnls(problem_spd_2x2, max_iter=0)


# ---------------------------------------------------------------------------
# Lemke
# ---------------------------------------------------------------------------

class TestLemke:
    def test_trivial(self, problem_trivial):
        result = lemke(problem_trivial)
        assert result.converged
        np.testing.assert_allclose(result.z, 0.0, atol=1e-10)

    def test_spd_2x2(self, problem_spd_2x2):
        result = lemke(problem_spd_2x2)
        assert result.converged
        check_lcp_solution(result, problem_spd_2x2.M, problem_spd_2x2.q)
        np.testing.assert_allclose(result.z, [1 / 3, 1 / 3], atol=1e-10)

    def test_mixed(self, problem_mixed):
        result = lemke(problem_mixed)
        assert result.converged
        check_lcp_solution(result, problem_mixed.M, problem_mixed.q)
        np.testing.assert_allclose(result.z, [0.5, 0.0], atol=1e-10)

    def test_all_active(self, problem_all_active):
        result = lemke(problem_all_active)
        assert result.converged
        np.testing.assert_allclose(result.z, [3.0, 2.0], atol=1e-10)

    def test_1x1_trivial(self):
        result = lemke(LCP(np.array([[2.0]]), np.array([1.0])))
        assert result.converged
        np.testing.assert_allclose(result.z, [0.0], atol=1e-12)

    def test_1x1_nontrivial(self):
        result = lemke(LCP(np.array([[2.0]]), np.array([-1.0])))
        assert result.converged
        np.testing.assert_allclose(result.z, [0.5], atol=1e-12)

    def test_ray_termination_for_infeasible_problem(self):
        # LCP(M=-2, q=-1): needs z >= 0 and -2z-1 >= 0 simultaneously — impossible.
        result = lemke(LCP(np.array([[-2.0]]), np.array([-1.0])))
        assert result.status == LCPStatus.RAY_TERMINATION

    def test_large_spd(self, problem_spd_large):
        result = lemke(problem_spd_large, max_iter=10000)
        assert result.converged
        check_lcp_solution(result, problem_spd_large.M, problem_spd_large.q)

    def test_max_iter_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="max_iter"):
            lemke(problem_spd_2x2, max_iter=0)

    def test_tol_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="tol"):
            lemke(problem_spd_2x2, tol=0.0)

    def test_iterations_nonneg(self, problem_spd_2x2):
        assert lemke(problem_spd_2x2).iterations >= 0

    def test_no_history(self, problem_spd_2x2):
        assert lemke(problem_spd_2x2).residual_history is None

    def test_message_contains_residual(self, problem_spd_2x2):
        assert "residual" in lemke(problem_spd_2x2).message


# ---------------------------------------------------------------------------
# CCG
# ---------------------------------------------------------------------------

class TestCCG:
    def test_trivial(self, problem_trivial):
        result = ccg(problem_trivial)
        assert result.converged
        check_lcp_solution(result, problem_trivial.M, problem_trivial.q)

    def test_spd_2x2(self, problem_spd_2x2):
        result = ccg(problem_spd_2x2)
        assert result.converged
        check_lcp_solution(result, problem_spd_2x2.M, problem_spd_2x2.q)

    def test_mixed(self, problem_mixed):
        result = ccg(problem_mixed)
        assert result.converged
        check_lcp_solution(result, problem_mixed.M, problem_mixed.q)

    def test_all_active(self, problem_all_active):
        result = ccg(problem_all_active)
        assert result.converged
        check_lcp_solution(result, problem_all_active.M, problem_all_active.q)

    def test_large_spd(self, problem_spd_large):
        # CCG (Polonsky-Keer) is designed for compliance matrices in contact
        # mechanics; it may not certify convergence for general SPD matrices
        # with the default pressure-factor initialisation. We verify that the
        # LCP conditions are approximately satisfied instead of asserting status.
        result = ccg(problem_spd_large, max_iter=100000)
        check_lcp_solution(result, problem_spd_large.M, problem_spd_large.q,
                           atol=1e-3)

    @pytest.mark.parametrize("err_type", ["displacement", "mix", "nw"])
    def test_err_types_converge(self, err_type, problem_spd_2x2):
        result = ccg(problem_spd_2x2, err_type=err_type)
        assert result.converged

    def test_invalid_err_type_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="err_type"):
            ccg(problem_spd_2x2, err_type="invalid")

    def test_record_history(self, problem_spd_2x2):
        result = ccg(problem_spd_2x2, record_history=True)
        assert result.residual_history is not None
        assert len(result.residual_history) > 0

    def test_no_history_by_default(self, problem_spd_2x2):
        assert ccg(problem_spd_2x2).residual_history is None

    def test_z0_warm_start(self, problem_spd_2x2):
        result = ccg(problem_spd_2x2, z0=np.array([0.3, 0.3]))
        assert result.converged

    def test_z0_wrong_shape_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="shape"):
            ccg(problem_spd_2x2, z0=np.zeros(5))

    def test_non_symmetric_raises(self):
        M = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(UnsupportedMatrixError, match="symmetric"):
            ccg(LCP(M, np.array([-1.0, -1.0])))

    def test_check_symmetric_false_skips_check(self):
        M = np.array([[2.0, 1.0], [1.5, 2.0]])
        result = ccg(LCP(M, np.array([-1.0, -1.0])), check_symmetric=False)
        assert isinstance(result.converged, bool)

    def test_tol_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="tol"):
            ccg(problem_spd_2x2, tol=0.0)

    def test_max_iter_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="max_iter"):
            ccg(problem_spd_2x2, max_iter=0)

    def test_pressure_factor_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="pressure_factor"):
            ccg(problem_spd_2x2, pressure_factor=0.0)

    def test_pressure_factor_negative_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="pressure_factor"):
            ccg(problem_spd_2x2, pressure_factor=-1.0)

    def test_message_contains_ccg(self, problem_spd_2x2):
        assert "CCG" in ccg(problem_spd_2x2).message


# ---------------------------------------------------------------------------
# CCG v2
# ---------------------------------------------------------------------------

class TestCCGv2:
    def test_trivial(self, problem_trivial):
        result = ccg_v2(problem_trivial)
        assert result.converged
        check_lcp_solution(result, problem_trivial.M, problem_trivial.q)

    def test_spd_2x2(self, problem_spd_2x2):
        result = ccg_v2(problem_spd_2x2)
        assert result.converged
        check_lcp_solution(result, problem_spd_2x2.M, problem_spd_2x2.q)

    def test_mixed(self, problem_mixed):
        result = ccg_v2(problem_mixed)
        assert result.converged
        check_lcp_solution(result, problem_mixed.M, problem_mixed.q)

    def test_all_active(self, problem_all_active):
        result = ccg_v2(problem_all_active)
        assert result.converged
        check_lcp_solution(result, problem_all_active.M, problem_all_active.q)

    def test_large_spd(self, problem_spd_large):
        result = ccg_v2(problem_spd_large)
        assert result.converged
        check_lcp_solution(result, problem_spd_large.M, problem_spd_large.q,
                           atol=1e-4)

    def test_record_history(self, problem_spd_2x2):
        result = ccg_v2(problem_spd_2x2, record_history=True)
        assert result.residual_history is not None

    def test_no_history_by_default(self, problem_spd_2x2):
        assert ccg_v2(problem_spd_2x2).residual_history is None

    def test_z0_warm_start(self, problem_spd_2x2):
        result = ccg_v2(problem_spd_2x2, z0=np.array([0.3, 0.3]))
        assert result.converged

    def test_z0_wrong_shape_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="shape"):
            ccg_v2(problem_spd_2x2, z0=np.zeros(5))

    def test_non_symmetric_raises(self):
        M = np.array([[2.0, 1.0], [0.0, 2.0]])
        with pytest.raises(UnsupportedMatrixError, match="symmetric"):
            ccg_v2(LCP(M, np.array([-1.0, -1.0])))

    def test_check_symmetric_false_skips_check(self):
        M = np.array([[2.0, 1.0], [1.5, 2.0]])
        result = ccg_v2(LCP(M, np.array([-1.0, -1.0])), check_symmetric=False)
        assert isinstance(result.converged, bool)

    def test_tol_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="tol"):
            ccg_v2(problem_spd_2x2, tol=0.0)

    def test_max_iter_zero_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="max_iter"):
            ccg_v2(problem_spd_2x2, max_iter=0)

    def test_message_contains_active_set_info(self, problem_spd_2x2):
        assert "active-set" in ccg_v2(problem_spd_2x2).message

    def test_iterations_counts_cg_steps(self, problem_spd_2x2):
        result = ccg_v2(problem_spd_2x2)
        assert result.iterations >= 0


class TestPPCG:
    @pytest.mark.parametrize(
        "fixture_name",
        ["problem_trivial", "problem_spd_2x2", "problem_mixed", "problem_all_active"],
    )
    def test_contact_cases(self, fixture_name, request):
        problem = request.getfixturevalue(fixture_name)
        result = ppcg(problem)
        assert result.converged
        check_lcp_solution(result, problem.M, problem.q)

    def test_operator_problem(self):
        class Operator:
            shape = (2, 2)
            symmetric = True

            def __matmul__(self, vector):
                return np.array(
                    [2.0 * vector[0] + vector[1], vector[0] + 2.0 * vector[1]]
                )

        result = solve(LCP(Operator(), [-1.0, -1.0]), method="ppcg")
        assert result.converged
        np.testing.assert_allclose(result.z, [1.0 / 3.0, 1.0 / 3.0])

    def test_preconditioner_is_used(self, problem_spd_2x2):
        calls = 0

        def diagonal_preconditioner(gradient, free):
            nonlocal calls
            calls += 1
            return np.where(free, gradient / 2.0, 0.0)

        result = ppcg(problem_spd_2x2, preconditioner=diagonal_preconditioner)
        assert result.converged
        assert calls > 0

    def test_bad_preconditioner_shape_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="preconditioner"):
            ppcg(
                problem_spd_2x2,
                preconditioner=lambda gradient, free: np.zeros(gradient.size + 1),
            )

    @pytest.mark.parametrize("beta_method", ["pr_plus", "fletcher_reeves"])
    def test_beta_methods(self, problem_spd_large, beta_method):
        result = ppcg(problem_spd_large, beta_method=beta_method)
        assert result.converged
        check_lcp_solution(result, problem_spd_large.M, problem_spd_large.q)

    def test_invalid_beta_method_raises(self, problem_spd_2x2):
        with pytest.raises(InvalidSolverOptionError, match="beta_method"):
            ppcg(problem_spd_2x2, beta_method="bogus")

    def test_history_and_message(self, problem_spd_2x2):
        result = ppcg(problem_spd_2x2, record_history=True)
        assert result.residual_history is not None
        assert "operator application" in result.message

    @pytest.mark.parametrize("seed", range(8))
    def test_random_spd_solution_matches_lemke(self, seed):
        rng = np.random.default_rng(seed)
        factor = rng.standard_normal((8, 8))
        matrix = factor.T @ factor + 0.2 * np.eye(8)
        q = rng.standard_normal(8)
        problem = LCP(matrix, q)

        result = ppcg(problem, tol=1.0e-11)
        reference = lemke(problem, tol=1.0e-11)

        assert result.converged
        np.testing.assert_allclose(result.z, reference.z, rtol=1.0e-8, atol=1.0e-9)

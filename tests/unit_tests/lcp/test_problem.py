"""Tests for cofebem.lcp.problem.LCP."""

import numpy as np
import pytest

from cofebem.lcp import LCP, InvalidLCPError


class TestLCPConstruction:
    def test_basic_construction(self):
        M = np.array([[2.0, 1.0], [1.0, 2.0]])
        q = np.array([-1.0, -1.0])
        problem = LCP(M, q)
        assert problem.size == 2

    def test_list_inputs_converted(self):
        problem = LCP([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
        assert isinstance(problem.M, np.ndarray)
        assert isinstance(problem.q, np.ndarray)
        assert problem.M.dtype == np.float64
        assert problem.q.dtype == np.float64

    def test_integer_inputs_promoted_to_float64(self):
        M = np.array([[2, 1], [1, 2]])
        q = np.array([-1, -1])
        problem = LCP(M, q)
        assert problem.M.dtype == np.float64
        assert problem.q.dtype == np.float64

    def test_size_matches_q_length(self):
        M = np.eye(5)
        q = np.ones(5)
        assert LCP(M, q).size == 5

    def test_1x1_accepted(self):
        problem = LCP(np.array([[3.0]]), np.array([1.0]))
        assert problem.size == 1

    def test_frozen_dataclass_rejects_attribute_assignment(self):
        problem = LCP(np.eye(2), np.ones(2))
        with pytest.raises(Exception):
            problem.M = np.eye(2)

    def test_m_stored_as_contiguous_float64(self):
        problem = LCP(np.eye(3), np.zeros(3))
        assert problem.M.dtype == np.float64

    def test_q_stored_as_1d_float64(self):
        problem = LCP(np.eye(3), np.zeros(3))
        assert problem.q.ndim == 1
        assert problem.q.dtype == np.float64

    def test_matrix_operator_is_retained_without_conversion(self):
        class Operator:
            shape = (2, 2)
            symmetric = True

            def __matmul__(self, vector):
                return 2.0 * vector

        operator = Operator()
        problem = LCP(operator, [-1.0, -1.0])
        assert problem.M is operator
        assert problem.uses_operator


class TestLCPValidation:
    def test_complex_M_raises(self):
        M = np.array([[1.0 + 0j, 0.0], [0.0, 1.0]])
        with pytest.raises(InvalidLCPError, match="complex"):
            LCP(M, np.array([1.0, 1.0]))

    def test_M_1d_raises(self):
        with pytest.raises(InvalidLCPError, match="two-dimensional"):
            LCP(np.array([1.0, 2.0]), np.array([1.0]))

    def test_M_3d_raises(self):
        with pytest.raises(InvalidLCPError, match="two-dimensional"):
            LCP(np.ones((2, 2, 2)), np.array([1.0, 1.0]))

    def test_M_non_square_raises(self):
        with pytest.raises(InvalidLCPError, match="square"):
            LCP(np.ones((2, 3)), np.array([1.0, 1.0]))

    def test_q_2d_raises(self):
        with pytest.raises(InvalidLCPError, match="one-dimensional"):
            LCP(np.eye(2), np.ones((2, 1)))

    def test_incompatible_M_q_dimensions_raises(self):
        with pytest.raises(InvalidLCPError, match="[Ii]ncompatible"):
            LCP(np.eye(2), np.array([1.0, 1.0, 1.0]))

    def test_empty_M_raises(self):
        with pytest.raises(InvalidLCPError, match="empty"):
            LCP(np.empty((0, 0)), np.array([]))

    def test_M_with_nan_raises(self):
        M = np.array([[1.0, np.nan], [0.0, 1.0]])
        with pytest.raises(InvalidLCPError, match="NaN"):
            LCP(M, np.array([1.0, 1.0]))

    def test_M_with_inf_raises(self):
        M = np.array([[1.0, np.inf], [0.0, 1.0]])
        with pytest.raises(InvalidLCPError, match="[Ii]nfinite|NaN"):
            LCP(M, np.array([1.0, 1.0]))

    def test_q_with_nan_raises(self):
        with pytest.raises(InvalidLCPError, match="NaN"):
            LCP(np.eye(2), np.array([1.0, np.nan]))

    def test_q_with_inf_raises(self):
        with pytest.raises(InvalidLCPError, match="[Ii]nfinite|NaN"):
            LCP(np.eye(2), np.array([1.0, np.inf]))

    def test_all_errors_are_invalid_lcp_error(self):
        """Every validation failure should be an InvalidLCPError."""
        bad_inputs = [
            (np.array([[1.0 + 0j]]), np.array([1.0])),
            (np.array([1.0]), np.array([1.0])),
            (np.ones((2, 3)), np.ones(2)),
            (np.eye(2), np.ones(3)),
            (np.empty((0, 0)), np.array([])),
        ]
        for M, q in bad_inputs:
            with pytest.raises(InvalidLCPError):
                LCP(M, q)

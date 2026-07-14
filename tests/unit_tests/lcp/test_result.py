"""Tests for cofebem.lcp.result.LCPResult and LCPStatus."""

import numpy as np
import pytest

from cofebem.lcp import InvalidLCPError, LCPResult, LCPStatus


class TestLCPStatus:
    def test_converged_equals_string(self):
        assert LCPStatus.CONVERGED == "converged"

    def test_max_iterations_equals_string(self):
        assert LCPStatus.MAX_ITERATIONS == "max_iterations"

    def test_stagnation_equals_string(self):
        assert LCPStatus.STAGNATION == "stagnation"

    def test_numerical_breakdown_equals_string(self):
        assert LCPStatus.NUMERICAL_BREAKDOWN == "numerical_breakdown"

    def test_ray_termination_equals_string(self):
        assert LCPStatus.RAY_TERMINATION == "ray_termination"

    def test_members_are_distinct(self):
        statuses = list(LCPStatus)
        assert len(statuses) == len(set(statuses))


class TestLCPResultConstruction:
    def _make(self, **kwargs):
        defaults = dict(
            z=np.array([1.0, 0.0]),
            w=np.array([0.0, 1.0]),
            status=LCPStatus.CONVERGED,
            iterations=5,
            residual=1e-12,
        )
        defaults.update(kwargs)
        return LCPResult(**defaults)

    def test_basic_construction(self):
        result = self._make()
        assert result.size == 2

    def test_size_matches_z_length(self):
        result = self._make(z=np.zeros(7), w=np.ones(7))
        assert result.size == 7

    def test_z_stored_as_float64_1d(self):
        result = self._make(z=np.array([1, 0]), w=np.array([0, 1]))
        assert result.z.dtype == np.float64
        assert result.z.ndim == 1

    def test_w_stored_as_float64_1d(self):
        result = self._make(z=np.array([1, 0]), w=np.array([0, 1]))
        assert result.w.dtype == np.float64
        assert result.w.ndim == 1

    def test_residual_stored_as_float(self):
        result = self._make(residual=0)
        assert isinstance(result.residual, float)

    def test_residual_history_none_by_default(self):
        assert self._make().residual_history is None

    def test_residual_history_stored_as_array(self):
        history = [1.0, 0.5, 0.1]
        result = self._make(residual_history=history)
        assert isinstance(result.residual_history, np.ndarray)
        np.testing.assert_array_almost_equal(result.residual_history, history)

    def test_message_defaults_to_empty_string(self):
        assert self._make().message == ""

    def test_frozen_dataclass_rejects_mutation(self):
        result = self._make()
        with pytest.raises(Exception):
            result.z = np.ones(2)


class TestLCPResultProperties:
    def _make(self, z, w, status=LCPStatus.CONVERGED):
        return LCPResult(z=np.array(z), w=np.array(w), status=status,
                         iterations=1, residual=0.0)

    def test_converged_true_when_status_is_converged(self):
        assert self._make([1.0, 0.0], [0.0, 1.0], LCPStatus.CONVERGED).converged

    def test_converged_false_for_max_iterations(self):
        assert not self._make([0.0], [1.0], LCPStatus.MAX_ITERATIONS).converged

    def test_converged_false_for_stagnation(self):
        assert not self._make([0.0], [1.0], LCPStatus.STAGNATION).converged

    def test_converged_false_for_numerical_breakdown(self):
        assert not self._make([0.0], [1.0], LCPStatus.NUMERICAL_BREAKDOWN).converged

    def test_converged_false_for_ray_termination(self):
        assert not self._make([0.0], [1.0], LCPStatus.RAY_TERMINATION).converged

    def test_primal_violation_zero_when_z_nonneg(self):
        result = self._make([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        assert result.primal_violation == 0.0

    def test_primal_violation_is_magnitude_of_most_negative_z(self):
        result = self._make([-0.5, 1.0], [1.0, 0.0])
        assert result.primal_violation == pytest.approx(0.5)

    def test_dual_violation_zero_when_w_nonneg(self):
        result = self._make([1.0, 0.0], [0.0, 1.0])
        assert result.dual_violation == 0.0

    def test_dual_violation_is_magnitude_of_most_negative_w(self):
        result = self._make([0.0, 1.0], [1.0, -0.3])
        assert result.dual_violation == pytest.approx(0.3)

    def test_complementarity_zero_for_complementary_pair(self):
        result = self._make([1.0, 0.0], [0.0, 1.0])
        assert result.complementarity == 0.0

    def test_complementarity_nonzero_when_both_positive(self):
        result = self._make([2.0, 1.0], [3.0, 1.0])
        # max(|z_i * w_i|) = max(6, 1) = 6
        assert result.complementarity == pytest.approx(6.0)

    def test_complementarity_uses_inf_norm(self):
        result = self._make([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert result.complementarity == pytest.approx(3.0)


class TestLCPResultValidation:
    def test_z_w_shape_mismatch_raises(self):
        with pytest.raises(InvalidLCPError, match="shape"):
            LCPResult(z=np.zeros(2), w=np.zeros(3),
                      status=LCPStatus.CONVERGED, iterations=1, residual=0.0)

    def test_negative_iterations_raises(self):
        with pytest.raises(InvalidLCPError, match="iterations"):
            LCPResult(z=np.zeros(2), w=np.zeros(2),
                      status=LCPStatus.CONVERGED, iterations=-1, residual=0.0)

    def test_zero_iterations_is_valid(self):
        result = LCPResult(z=np.zeros(2), w=np.zeros(2),
                           status=LCPStatus.CONVERGED, iterations=0, residual=0.0)
        assert result.iterations == 0

    def test_negative_residual_raises(self):
        with pytest.raises(InvalidLCPError, match="residual"):
            LCPResult(z=np.zeros(2), w=np.zeros(2),
                      status=LCPStatus.CONVERGED, iterations=0, residual=-1.0)

    def test_zero_residual_is_valid(self):
        result = LCPResult(z=np.zeros(2), w=np.zeros(2),
                           status=LCPStatus.CONVERGED, iterations=0, residual=0.0)
        assert result.residual == 0.0

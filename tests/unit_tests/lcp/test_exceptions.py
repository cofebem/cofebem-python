"""Tests for cofebem.lcp.exceptions hierarchy."""

import pytest

from cofebem.lcp import (
    InvalidLCPError,
    InvalidSolverOptionError,
    LCPError,
    LCPNumericalError,
    UnsupportedMatrixError,
    UnsupportedSolverError,
)


class TestExceptionHierarchy:
    def test_lcp_error_is_exception(self):
        assert issubclass(LCPError, Exception)

    def test_invalid_lcp_error_is_lcp_error(self):
        assert issubclass(InvalidLCPError, LCPError)

    def test_invalid_solver_option_error_is_lcp_error(self):
        assert issubclass(InvalidSolverOptionError, LCPError)

    def test_unsupported_solver_error_is_lcp_error(self):
        assert issubclass(UnsupportedSolverError, LCPError)

    def test_unsupported_matrix_error_is_lcp_error(self):
        assert issubclass(UnsupportedMatrixError, LCPError)

    def test_lcp_numerical_error_is_lcp_error(self):
        assert issubclass(LCPNumericalError, LCPError)


class TestExceptionSecondaryBases:
    def test_invalid_lcp_error_is_value_error(self):
        assert issubclass(InvalidLCPError, ValueError)

    def test_invalid_solver_option_error_is_value_error(self):
        assert issubclass(InvalidSolverOptionError, ValueError)

    def test_unsupported_solver_error_is_value_error(self):
        assert issubclass(UnsupportedSolverError, ValueError)

    def test_unsupported_matrix_error_is_type_error(self):
        assert issubclass(UnsupportedMatrixError, TypeError)

    def test_lcp_numerical_error_is_runtime_error(self):
        assert issubclass(LCPNumericalError, RuntimeError)


class TestExceptionCatchability:
    @pytest.mark.parametrize("exc_class", [
        InvalidLCPError,
        InvalidSolverOptionError,
        UnsupportedSolverError,
        UnsupportedMatrixError,
        LCPNumericalError,
    ])
    def test_catchable_as_lcp_error(self, exc_class):
        with pytest.raises(LCPError):
            raise exc_class("test message")

    def test_invalid_lcp_error_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise InvalidLCPError("bad input")

    def test_invalid_solver_option_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise InvalidSolverOptionError("bad option")

    def test_unsupported_solver_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise UnsupportedSolverError("unknown solver")

    def test_unsupported_matrix_catchable_as_type_error(self):
        with pytest.raises(TypeError):
            raise UnsupportedMatrixError("wrong matrix type")

    def test_lcp_numerical_error_catchable_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise LCPNumericalError("numerical failure")

    def test_exceptions_carry_message(self):
        msg = "descriptive error"
        exc = InvalidLCPError(msg)
        assert str(exc) == msg

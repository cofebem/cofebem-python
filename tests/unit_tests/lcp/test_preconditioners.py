"""Tests for matrix-free contact preconditioners."""

import numpy as np
import pytest
from scipy.fft import dct, fft, idct, ifft

from cofebem.lcp import (
    LCP,
    RestrictedProjectedPreconditioner,
    SectorSurfaceSpectralPreconditioner,
)
from cofebem.lcp.solvers import ppcg


def _sector_points(n_sectors=8, n_axial=7):
    x = np.linspace(-0.5, 0.5, n_axial)
    points = np.empty((n_sectors, n_axial, 3))
    for sector in range(n_sectors):
        angle = 2.0 * np.pi * sector / n_sectors
        points[sector, :, 0] = x
        points[sector, :, 1] = 2.0 * np.cos(angle)
        points[sector, :, 2] = 2.0 * np.sin(angle) - 0.3
    return points


def _spectral_compliance(preconditioner, vector):
    values = vector.reshape(
        preconditioner.n_sectors, preconditioner.n_axial
    )
    coefficients = fft(
        dct(values, type=2, axis=1, norm="ortho"), axis=0, norm="ortho"
    )
    transformed = ifft(
        coefficients / preconditioner.symbol, axis=0, norm="ortho"
    ).real
    return idct(transformed, type=2, axis=1, norm="ortho").reshape(-1)


def test_sector_spectral_preconditioner_is_positive_on_masked_subspace():
    preconditioner = SectorSurfaceSpectralPreconditioner(_sector_points())
    rng = np.random.default_rng(4)
    residual = rng.standard_normal(preconditioner.shape[0])
    free = rng.random(preconditioner.shape[0]) > 0.35

    result = preconditioner(residual, free)

    assert np.all(result[~free] == 0.0)
    assert float(residual @ result) > 0.0


def test_constant_mode_is_not_removed():
    preconditioner = SectorSurfaceSpectralPreconditioner(_sector_points())
    ones = np.ones(preconditioner.shape[0])
    result = preconditioner(ones, np.ones_like(ones, dtype=bool))

    assert np.linalg.norm(result) > 0.0
    np.testing.assert_allclose(result, result.mean(), rtol=1.0e-12, atol=1.0e-12)


def test_exact_matching_spectrum_collapses_pcg_iterations():
    preconditioner = SectorSurfaceSpectralPreconditioner(_sector_points())
    n = preconditioner.shape[0]
    basis = np.eye(n)
    matrix = np.column_stack(
        [_spectral_compliance(preconditioner, basis[:, j]) for j in range(n)]
    )
    rng = np.random.default_rng(9)
    expected = 2.0 + 1.0e-3 * rng.standard_normal(n)
    q = -(matrix @ expected)
    assert np.all(q < 0.0)

    result = ppcg(LCP(matrix, q), preconditioner=preconditioner, tol=1.0e-11)

    assert result.converged
    assert result.iterations <= 2
    np.testing.assert_allclose(result.z, expected, rtol=1.0e-9, atol=1.0e-9)


def test_restricted_preconditioner_matches_scatter_apply_gather():
    full = SectorSurfaceSpectralPreconditioner(_sector_points())
    indices = np.array([0, 1, 7, 12, 20, 31, 47, 55])
    restricted = RestrictedProjectedPreconditioner(full, indices)
    rng = np.random.default_rng(13)
    residual = rng.standard_normal(indices.size)
    free = rng.random(indices.size) > 0.25

    result = restricted(residual, free)
    full_residual = np.zeros(full.shape[0])
    full_free = np.zeros(full.shape[0], dtype=bool)
    full_residual[indices] = residual
    full_free[indices] = free
    expected = full(full_residual, full_free)[indices]

    np.testing.assert_allclose(result, expected)
    assert float(residual @ result) > 0.0


@pytest.mark.parametrize("zero_mode_factor", [0.0, -1.0])
def test_zero_mode_factor_must_be_positive(zero_mode_factor):
    with pytest.raises(ValueError, match="zero_mode_factor"):
        SectorSurfaceSpectralPreconditioner(
            _sector_points(), zero_mode_factor=zero_mode_factor
        )

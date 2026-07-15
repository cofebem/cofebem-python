"""Preconditioners for matrix-free contact complementarity solvers."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, fft, fftfreq, idct, ifft


class RestrictedProjectedPreconditioner:
    """Principal-subspace view of a full projected preconditioner.

    Restricted residuals and free masks are scattered into the full surface,
    the wrapped preconditioner is applied there, and its result is gathered at
    ``indices``. If the wrapped linear preconditioner is SPD, this restriction
    remains SPD on the selected subspace.
    """

    def __init__(self, preconditioner, indices: np.ndarray) -> None:
        if not callable(preconditioner):
            raise TypeError("preconditioner must be callable")
        try:
            shape = tuple(preconditioner.shape)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("preconditioner must expose a square shape") from exc
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("preconditioner must expose a square shape")
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise ValueError("indices must not be empty")
        if np.any(selected < 0) or np.any(selected >= shape[0]):
            raise IndexError("selected index outside preconditioner")
        if np.unique(selected).size != selected.size:
            raise ValueError("indices must be unique")
        self.preconditioner = preconditioner
        self.indices = selected.copy()
        self.full_size = int(shape[0])
        self.shape = (selected.size, selected.size)

    def __call__(
        self, residual: np.ndarray, free_mask: np.ndarray
    ) -> np.ndarray:
        """Apply the full preconditioner and gather its principal restriction."""
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        free = np.asarray(free_mask, dtype=bool).reshape(-1)
        n = self.shape[0]
        if residual.shape != (n,) or free.shape != (n,):
            raise ValueError(f"residual and free_mask must have shape ({n},)")
        full_residual = np.zeros(self.full_size, dtype=np.float64)
        full_free = np.zeros(self.full_size, dtype=bool)
        full_residual[self.indices] = residual
        full_free[self.indices] = free
        result = np.asarray(
            self.preconditioner(full_residual, full_free), dtype=np.float64
        ).reshape(-1)
        if result.shape != (self.full_size,):
            raise ValueError(
                "wrapped preconditioner returned shape "
                f"{result.shape}, expected ({self.full_size},)"
            )
        return result[self.indices]


class SectorSurfaceSpectralPreconditioner:
    """Spectral inverse-compliance model for a sector-ordered surface.

    The contact vector must be ordered by circumferential sector and then by
    axial node. A periodic Fourier transform is used in the circumferential
    direction and a cosine transform in the non-periodic axial direction.
    Multiplication by ``sqrt(q_theta**2 + q_x**2 + q_0**2)`` approximates the
    inverse of the elastic compliance's ``1 / |q|`` high-frequency symbol.

    The positive ``q_0`` term is essential for indentation-controlled LCPs:
    unlike a fixed-total-load formulation, their constant-pressure mode is a
    genuine degree of freedom and must not be projected out.

    Parameters
    ----------
    sector_points : ndarray, shape (n_sectors, n_axial, 3)
        Coordinates in the same sector-major order as the LCP unknowns. The
        tyre/revolution axis is assumed to be global x.
    zero_mode_factor : float
        ``q_0`` relative to the smallest nonzero modal wavenumber.
    """

    def __init__(
        self,
        sector_points: np.ndarray,
        *,
        zero_mode_factor: float = 1.0,
    ) -> None:
        points = np.asarray(sector_points, dtype=np.float64)
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError("sector_points must have shape (n_sectors, n_axial, 3)")
        self.n_sectors, self.n_axial, _ = points.shape
        if self.n_sectors < 2 or self.n_axial < 2:
            raise ValueError("at least two sectors and two axial nodes are required")
        if zero_mode_factor <= 0.0:
            raise ValueError("zero_mode_factor must be positive")
        if not np.all(np.isfinite(points)):
            raise ValueError("sector_points contain NaN or infinite values")

        reference_x = points[0, :, 0]
        if not np.allclose(points[:, :, 0], reference_x, rtol=0.0, atol=1.0e-10):
            raise ValueError("axial coordinates must agree between sectors")
        axial_length = float(np.ptp(reference_x))
        if axial_length <= 0.0:
            raise ValueError("axial coordinates must span a positive length")

        # Averaging the transverse coordinates over all regular sectors gives
        # the revolution axis even after the mesh has been shifted vertically.
        transverse_axis = points[:, :, 1:].mean(axis=0)
        radii = np.linalg.norm(
            points[:, :, 1:] - transverse_axis[None, :, :], axis=2
        )
        mean_radius = float(np.mean(radii))
        if mean_radius <= 0.0:
            raise ValueError("sector points must have a positive mean radius")

        q_axial = np.pi * np.arange(self.n_axial) / axial_length
        arc_spacing = 2.0 * np.pi * mean_radius / self.n_sectors
        q_theta = 2.0 * np.pi * fftfreq(self.n_sectors, d=arc_spacing)
        wave_number = np.hypot(q_theta[:, None], q_axial[None, :])
        q_min = float(np.min(wave_number[wave_number > 0.0]))
        q_zero = zero_mode_factor * q_min

        self.symbol = np.sqrt(wave_number**2 + q_zero**2)
        self.shape = (self.n_sectors * self.n_axial,) * 2
        self.axial_length = axial_length
        self.mean_radius = mean_radius
        self.zero_mode_factor = float(zero_mode_factor)

    def __call__(
        self, residual: np.ndarray, free_mask: np.ndarray
    ) -> np.ndarray:
        """Apply the masked SPD spectral preconditioner."""
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        free = np.asarray(free_mask, dtype=bool).reshape(-1)
        n = self.shape[0]
        if residual.shape != (n,) or free.shape != (n,):
            raise ValueError(f"residual and free_mask must have shape ({n},)")
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual contains NaN or infinite values")

        masked = np.where(free, residual, 0.0).reshape(
            self.n_sectors, self.n_axial
        )
        coefficients = fft(
            dct(masked, type=2, axis=1, norm="ortho"),
            axis=0,
            norm="ortho",
        )
        transformed = ifft(
            self.symbol * coefficients, axis=0, norm="ortho"
        ).real
        result = idct(
            transformed, type=2, axis=1, norm="ortho"
        ).reshape(-1)
        result[~free] = 0.0
        return result

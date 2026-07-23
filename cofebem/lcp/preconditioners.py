"""Preconditioners for matrix-free contact complementarity solvers."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, fft, fftfreq, idct, ifft


class SurfaceAreaDiagonalPreconditioner:
    """SPD local inverse-compliance scale for an unstructured surface.

    For a nodal patch of characteristic size ``h ~ sqrt(area)``, elastic
    half-space compliance scales approximately as ``1 / h``. Multiplication by
    ``sqrt(area)`` is therefore a cheap diagonal approximation to its inverse.
    A positive floor prevents small boundary patches from becoming singular.
    """

    def __init__(self, nodal_areas: np.ndarray, *, floor_ratio: float = 1.0e-3):
        areas = np.asarray(nodal_areas, dtype=np.float64).reshape(-1)
        if areas.size == 0 or not np.all(np.isfinite(areas)):
            raise ValueError("nodal_areas must be a nonempty finite vector")
        if np.any(areas <= 0.0):
            raise ValueError("nodal_areas must be positive")
        if not 0.0 < floor_ratio <= 1.0:
            raise ValueError("floor_ratio must lie in (0, 1]")
        scale = np.sqrt(areas)
        floor = floor_ratio * float(np.median(scale))
        self.diagonal = np.maximum(scale, floor)
        self.diagonal /= float(np.mean(self.diagonal))
        self.shape = (areas.size, areas.size)

    def __call__(self, residual: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        free = np.asarray(free_mask, dtype=bool).reshape(-1)
        n = self.shape[0]
        if residual.shape != (n,) or free.shape != (n,):
            raise ValueError(f"residual and free_mask must have shape ({n},)")
        if not np.all(np.isfinite(residual)):
            raise ValueError("residual contains NaN or infinite values")
        return np.where(free, self.diagonal * residual, 0.0)


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

    Nonuniform meridians are accepted for the graded tyre. Their revolution
    axis is obtained by circle fitting; the Fourier transform then acts in the
    periodic meridian index and remains an SPD algebraic preconditioner, though
    its symbol is only an approximation to the nonuniform physical spacing.

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

        # Fit the revolution axis independently on every axial circle. Unlike
        # a coordinate average, this remains exact for nonuniform angular
        # meridians used by the graded tyre mesh.
        transverse_axis = np.empty((self.n_axial, 2), dtype=np.float64)
        for axial in range(self.n_axial):
            yz = points[:, axial, 1:]
            circle_system = np.column_stack(
                [2.0 * yz[:, 0], 2.0 * yz[:, 1], np.ones(self.n_sectors)]
            )
            circle_rhs = np.sum(yz**2, axis=1)
            fit, *_ = np.linalg.lstsq(circle_system, circle_rhs, rcond=None)
            transverse_axis[axial] = fit[:2]
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

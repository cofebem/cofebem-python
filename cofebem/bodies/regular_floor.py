"""Regular flat or self-affine rough rigid floors for vertical contact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegularFloor:
    """A height field on a regular tensor-product grid.

    Array axis 0 follows ``y`` and axis 1 follows ``x``. Contact projection is
    vertical, consistently with the current global-z tyre compliance operator.
    """

    x: np.ndarray
    y: np.ndarray
    height: np.ndarray
    kind: str = "flat"
    level: float = 0.0

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        height = np.asarray(self.height, dtype=np.float64)
        if x.size < 2 or y.size < 2:
            raise ValueError("floor grid needs at least two nodes per direction")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("floor coordinates must be strictly increasing")
        if height.shape != (y.size, x.size):
            raise ValueError(
                f"height must have shape {(y.size, x.size)}, got {height.shape}"
            )
        if not np.all(np.isfinite(height)):
            raise ValueError("floor heights contain NaN or infinite values")
        if self.kind not in {"flat", "rough"}:
            raise ValueError("floor kind must be 'flat' or 'rough'")
        object.__setattr__(self, "x", x.copy())
        object.__setattr__(self, "y", y.copy())
        object.__setattr__(self, "height", height.copy())

    @property
    def cells_x(self) -> int:
        return self.x.size - 1

    @property
    def cells_y(self) -> int:
        return self.y.size - 1

    @classmethod
    def flat(
        cls,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        *,
        cells: int = 256,
        level: float = 0.0,
    ) -> "RegularFloor":
        """Construct a regular flat floor."""
        x, y = _regular_coordinates(x_bounds, y_bounds, cells)
        return cls(
            x=x,
            y=y,
            height=np.full((cells + 1, cells + 1), level, dtype=float),
            kind="flat",
            level=float(level),
        )

    @classmethod
    def self_affine(
        cls,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        *,
        cells: int = 256,
        level: float = 0.0,
        rms: float = 2.0e-4,
        hurst: float = 0.8,
        k_low: float = 0.03,
        k_high: float = 0.3,
        plateau: bool = False,
        noise: bool = True,
        seed: int = 42,
    ) -> "RegularFloor":
        """Construct a periodic rfgen self-affine floor.

        ``level`` is the highest-asperity datum. Subtracting the generated
        maximum follows the rough-contact reference convention and makes a
        rough floor no higher than the corresponding flat floor.
        """
        if rms <= 0.0:
            raise ValueError("roughness rms must be positive")
        if not 0.0 <= hurst <= 1.0:
            raise ValueError("roughness Hurst exponent must lie in [0, 1]")
        if not 0.0 < k_low < k_high <= 0.5:
            raise ValueError("require 0 < k_low < k_high <= 0.5")
        x, y = _regular_coordinates(x_bounds, y_bounds, cells)
        try:
            import rfgen
        except ImportError as exc:  # pragma: no cover - installation diagnostic
            raise RuntimeError(
                "Rough floors require rfgen; install the project dependencies"
            ) from exc

        field = np.asarray(
            rfgen.selfaffine_field(
                dim=2,
                N=cells,
                Hurst=hurst,
                k_low=k_low,
                k_high=k_high,
                plateau=plateau,
                noise=noise,
                rng=np.random.default_rng(seed),
                verbose=False,
            ),
            dtype=np.float64,
        )
        if field.shape != (cells, cells):
            raise RuntimeError(
                f"rfgen returned shape {field.shape}, expected {(cells, cells)}"
            )
        field -= field.mean()
        standard_deviation = float(field.std())
        if standard_deviation <= np.finfo(float).tiny:
            raise RuntimeError("rfgen produced a constant roughness field")
        field *= rms / standard_deviation
        field += float(level) - float(field.max())

        # Close the periodic rfgen field on the inclusive visualization and
        # interpolation grid.
        height = np.empty((cells + 1, cells + 1), dtype=np.float64)
        height[:-1, :-1] = field
        height[-1, :-1] = field[0, :]
        height[:-1, -1] = field[:, 0]
        height[-1, -1] = field[0, 0]
        return cls(x=x, y=y, height=height, kind="rough", level=float(level))

    def height_at(self, projected_xy: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate floor height at projected ``(x, y)`` points."""
        points = np.asarray(projected_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("projected_xy must have shape (n, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("projected coordinates contain NaN or infinity")
        scale = max(
            float(self.x[-1] - self.x[0]),
            float(self.y[-1] - self.y[0]),
            1.0,
        )
        tolerance = 1.0e-12 * scale
        outside = (
            (points[:, 0] < self.x[0] - tolerance)
            | (points[:, 0] > self.x[-1] + tolerance)
            | (points[:, 1] < self.y[0] - tolerance)
            | (points[:, 1] > self.y[-1] + tolerance)
        )
        if np.any(outside):
            first = points[np.flatnonzero(outside)[0]]
            raise ValueError(
                "projected tyre point lies outside the floor grid: "
                f"({first[0]:.6g}, {first[1]:.6g})"
            )

        px = np.clip(points[:, 0], self.x[0], self.x[-1])
        py = np.clip(points[:, 1], self.y[0], self.y[-1])
        ix = np.clip(np.searchsorted(self.x, px, side="right") - 1, 0, self.cells_x - 1)
        iy = np.clip(np.searchsorted(self.y, py, side="right") - 1, 0, self.cells_y - 1)
        tx = (px - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        ty = (py - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        h00 = self.height[iy, ix]
        h10 = self.height[iy, ix + 1]
        h01 = self.height[iy + 1, ix]
        h11 = self.height[iy + 1, ix + 1]
        return (
            (1.0 - tx) * (1.0 - ty) * h00
            + tx * (1.0 - ty) * h10
            + (1.0 - tx) * ty * h01
            + tx * ty * h11
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        """Vertically project three-dimensional points onto the floor."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        projected = values.copy()
        projected[:, 2] = self.height_at(values[:, :2])
        return projected

    def initial_gap(self, points: np.ndarray) -> np.ndarray:
        """Return vertical initial clearance ``z_tyre - z_floor``."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        return values[:, 2] - self.height_at(values[:, :2])

    def point_normals(self) -> np.ndarray:
        """Return upward unit normals at regular floor nodes."""
        if self.kind == "rough":
            core = self.height[:-1, :-1]
            dx = float((self.x[-1] - self.x[0]) / self.cells_x)
            dy = float((self.y[-1] - self.y[0]) / self.cells_y)
            dh_dx = (np.roll(core, -1, axis=1) - np.roll(core, 1, axis=1)) / (
                2.0 * dx
            )
            dh_dy = (np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0)) / (
                2.0 * dy
            )
            core_normals = np.stack(
                (-dh_dx, -dh_dy, np.ones_like(core)), axis=-1
            )
            core_normals /= np.linalg.norm(
                core_normals, axis=-1, keepdims=True
            )
            normals = np.empty((self.y.size, self.x.size, 3), dtype=np.float64)
            normals[:-1, :-1] = core_normals
            normals[-1, :-1] = core_normals[0, :]
            normals[:-1, -1] = core_normals[:, 0]
            normals[-1, -1] = core_normals[0, 0]
            return normals
        dh_dy, dh_dx = np.gradient(self.height, self.y, self.x, edge_order=2)
        normals = np.stack(
            (-dh_dx, -dh_dy, np.ones_like(self.height)), axis=-1
        )
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    def write_vtu(self, path: str | Path) -> Path:
        """Write the regular floor geometry and height/normal fields."""
        try:
            import meshio
        except ImportError as exc:  # pragma: no cover - installation diagnostic
            raise RuntimeError("meshio is required to write the floor mesh") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xx, yy = np.meshgrid(self.x, self.y, indexing="xy")
        points = np.column_stack((xx.ravel(), yy.ravel(), self.height.ravel()))
        node_ids = np.arange(points.shape[0]).reshape(self.y.size, self.x.size)
        quads = np.column_stack(
            (
                node_ids[:-1, :-1].ravel(),
                node_ids[:-1, 1:].ravel(),
                node_ids[1:, 1:].ravel(),
                node_ids[1:, :-1].ravel(),
            )
        )
        meshio.write_points_cells(
            path,
            points,
            [("quad", quads)],
            point_data={
                "floor_height": self.height.ravel(),
                "floor_normal": self.point_normals().reshape(-1, 3),
            },
        )
        return path


def square_floor_bounds(
    projected_xy: np.ndarray, *, margin: float = 0.02
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return centered square bounds containing all projected points."""
    points = np.asarray(projected_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("projected_xy must have nonzero shape (n, 2)")
    if margin < 0.0:
        raise ValueError("floor margin must be non-negative")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    side = max(float((upper - lower).max()) + 2.0 * margin, 1.0e-12)
    half = 0.5 * side
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
    )


def _regular_coordinates(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cells < 2:
        raise ValueError("floor grid needs at least two cells per direction")
    x0, x1 = (float(value) for value in x_bounds)
    y0, y1 = (float(value) for value in y_bounds)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("floor bounds must be increasing")
    return (
        np.linspace(x0, x1, cells + 1, dtype=np.float64),
        np.linspace(y0, y1, cells + 1, dtype=np.float64),
    )

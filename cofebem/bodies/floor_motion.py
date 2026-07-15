"""Rigid motion and time schedules for regular height-field floors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .regular_floor import RegularFloor


@dataclass(frozen=True)
class FloorMotionState:
    """One rigid-floor state in the global coordinate system.

    ``indentation`` is an upward global-z translation. Rotations are expressed
    in degrees and composed as ``Rz(rotation_z) @ Ry(rotation_y)`` about the
    floor pivot. The in-plane translations are applied after the rotations.
    """

    time: float
    indentation: float = 0.0
    rotation_y_deg: float = 0.0
    rotation_z_deg: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.time,
                self.indentation,
                self.rotation_y_deg,
                self.rotation_z_deg,
                self.translation_x,
                self.translation_y,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("floor motion state contains NaN or infinity")
        if self.indentation < 0.0:
            raise ValueError("floor indentation must be non-negative")
        if abs(self.rotation_y_deg) >= 89.0:
            raise ValueError("vertical floor projection requires |rotation_y| < 89 deg")

    @property
    def translation(self) -> np.ndarray:
        return np.array(
            [self.translation_x, self.translation_y, self.indentation],
            dtype=np.float64,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        angle_y, angle_z = np.deg2rad(
            [self.rotation_y_deg, self.rotation_z_deg]
        )
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        rotation_y = np.array(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
        )
        rotation_z = np.array(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
        )
        return rotation_z @ rotation_y


@dataclass(frozen=True)
class FloorMotionSchedule:
    """Expanded, linearly interpolated rigid-floor load history."""

    states: tuple[FloorMotionState, ...]
    key_times: np.ndarray
    interval_steps: np.ndarray

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("floor motion schedule must contain at least one state")
        times = np.array([state.time for state in self.states])
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("expanded floor motion times must be strictly increasing")
        object.__setattr__(self, "key_times", np.asarray(self.key_times, dtype=float))
        object.__setattr__(
            self, "interval_steps", np.asarray(self.interval_steps, dtype=np.int64)
        )

    @classmethod
    def constant(
        cls,
        *,
        indentation: float = 0.0,
        rotation_y_deg: float = 0.0,
        rotation_z_deg: float = 0.0,
        translation_x: float = 0.0,
        translation_y: float = 0.0,
    ) -> "FloorMotionSchedule":
        state = FloorMotionState(
            time=0.0,
            indentation=indentation,
            rotation_y_deg=rotation_y_deg,
            rotation_z_deg=rotation_z_deg,
            translation_x=translation_x,
            translation_y=translation_y,
        )
        return cls((state,), np.array([0.0]), np.empty(0, dtype=np.int64))

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        defaults: FloorMotionState | None = None,
    ) -> "FloorMotionSchedule":
        """Load key frames and expand every interval by linear interpolation."""
        path = Path(path).expanduser()
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("motion JSON root must be an object")
        if "time" not in payload:
            raise ValueError("motion JSON requires a 'time' array")
        key_times = _numeric_vector(payload["time"], "time")
        if key_times.size < 2:
            raise ValueError("motion JSON needs at least two key times")
        if np.any(np.diff(key_times) <= 0.0):
            raise ValueError("motion key times must be strictly increasing")

        if "interval_steps" not in payload:
            raise ValueError("motion JSON requires an 'interval_steps' array")
        raw_steps = np.asarray(payload["interval_steps"])
        if raw_steps.ndim != 1 or raw_steps.size != key_times.size - 1:
            raise ValueError("interval_steps must have len(time) - 1 entries")
        if not np.all(np.isfinite(raw_steps)) or not np.all(
            raw_steps == np.floor(raw_steps)
        ):
            raise ValueError("interval_steps entries must be finite integers")
        interval_steps = raw_steps.astype(np.int64)
        if np.any(interval_steps <= 0):
            raise ValueError("interval_steps entries must be positive")

        default = defaults or FloorMotionState(time=float(key_times[0]))
        fields = {
            "indentation": default.indentation,
            "floor_rotation_y_deg": default.rotation_y_deg,
            "floor_rotation_z_deg": default.rotation_z_deg,
            "floor_translation_x": default.translation_x,
            "floor_translation_y": default.translation_y,
        }
        values = {
            name: _key_values(payload.get(name, fallback), key_times.size, name)
            for name, fallback in fields.items()
        }

        states: list[FloorMotionState] = []
        for interval, steps in enumerate(interval_steps):
            for local_step in range(int(steps)):
                alpha = local_step / float(steps)
                states.append(
                    _interpolated_state(key_times, values, interval, alpha)
                )
        states.append(
            _interpolated_state(key_times, values, key_times.size - 2, 1.0)
        )
        return cls(tuple(states), key_times, interval_steps)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return expanded schedule values for result archives."""
        return {
            "time": np.array([state.time for state in self.states]),
            "indentation": np.array([state.indentation for state in self.states]),
            "floor_rotation_y_deg": np.array(
                [state.rotation_y_deg for state in self.states]
            ),
            "floor_rotation_z_deg": np.array(
                [state.rotation_z_deg for state in self.states]
            ),
            "floor_translation_x": np.array(
                [state.translation_x for state in self.states]
            ),
            "floor_translation_y": np.array(
                [state.translation_y for state in self.states]
            ),
        }


@dataclass(frozen=True)
class MovingRegularFloor:
    """A regular floor viewed through one rigid motion state."""

    floor: RegularFloor
    state: FloorMotionState
    pivot: np.ndarray

    def __post_init__(self) -> None:
        pivot = np.asarray(self.pivot, dtype=np.float64)
        if pivot.shape != (3,) or not np.all(np.isfinite(pivot)):
            raise ValueError("floor pivot must be a finite three-vector")
        object.__setattr__(self, "pivot", pivot.copy())

    def surface_points(self) -> np.ndarray:
        """Return transformed regular-grid nodes in row-major order."""
        xx, yy = np.meshgrid(self.floor.x, self.floor.y, indexing="xy")
        local = np.column_stack((xx.ravel(), yy.ravel(), self.floor.height.ravel()))
        return self._transform(local)

    def point_normals(self) -> np.ndarray:
        normals = self.floor.point_normals().reshape(-1, 3)
        return normals @ self.state.rotation_matrix.T

    def height_at(self, global_xy: np.ndarray) -> np.ndarray:
        """Intersect global vertical rays with the rigidly transformed floor."""
        query = np.asarray(global_xy, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("global_xy must have shape (n, 2)")
        if not np.all(np.isfinite(query)):
            raise ValueError("global floor queries contain NaN or infinity")

        rotation = self.state.rotation_matrix
        translation = self.state.translation
        if self.floor.kind == "flat":
            base_point = np.array(
                [self.pivot[0], self.pivot[1], self.floor.level], dtype=float
            )
            plane_point = self._transform(base_point[None, :])[0]
            normal = rotation[:, 2]
            if abs(normal[2]) < 1.0e-8:
                raise ValueError(
                    "rotated floor is not a vertical-projection height field"
                )
            return plane_point[2] - (
                normal[0] * (query[:, 0] - plane_point[0])
                + normal[1] * (query[:, 1] - plane_point[1])
            ) / normal[2]

        horizontal = rotation[:2, :2]
        rhs = (
            query
            - self.pivot[:2]
            - translation[:2]
            - np.outer(
                np.full(query.shape[0], self.floor.level - self.pivot[2]),
                rotation[:2, 2],
            )
        )
        local_xy = self.pivot[:2] + np.linalg.solve(horizontal, rhs.T).T
        local_xy[:, 0] = np.clip(local_xy[:, 0], self.floor.x[0], self.floor.x[-1])
        local_xy[:, 1] = np.clip(local_xy[:, 1], self.floor.y[0], self.floor.y[-1])
        scale = max(float(np.ptp(query, axis=0).max()), 1.0)
        tolerance = 5.0e-13 * scale
        for _ in range(12):
            height, gradient = self.floor.height_and_gradient_at(local_xy)
            local = np.column_stack((local_xy, height))
            residual = self._transform(local)[:, :2] - query
            if float(np.max(np.linalg.norm(residual, axis=1))) <= tolerance:
                return self._transform(local)[:, 2]
            jacobian = horizontal[None, :, :] + np.einsum(
                "i,nj->nij", rotation[:2, 2], gradient
            )
            determinant = (
                jacobian[:, 0, 0] * jacobian[:, 1, 1]
                - jacobian[:, 0, 1] * jacobian[:, 1, 0]
            )
            if np.any(np.abs(determinant) < 1.0e-10):
                raise RuntimeError("transformed rough floor has a singular projection")
            delta = np.empty_like(residual)
            delta[:, 0] = (
                jacobian[:, 1, 1] * residual[:, 0]
                - jacobian[:, 0, 1] * residual[:, 1]
            ) / determinant
            delta[:, 1] = (
                -jacobian[:, 1, 0] * residual[:, 0]
                + jacobian[:, 0, 0] * residual[:, 1]
            ) / determinant
            local_xy -= delta
            local_xy[:, 0] = np.clip(
                local_xy[:, 0], self.floor.x[0], self.floor.x[-1]
            )
            local_xy[:, 1] = np.clip(
                local_xy[:, 1], self.floor.y[0], self.floor.y[-1]
            )
        raise RuntimeError(
            "vertical projection onto transformed rough floor did not converge"
        )

    def initial_gap(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        return points[:, 2] - self.height_at(points[:, :2])

    def write_vtu(self, path: str | Path) -> Path:
        """Write the transformed floor mesh for this state."""
        try:
            import meshio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("meshio is required to write the floor mesh") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = self.surface_points()
        node_ids = np.arange(points.shape[0]).reshape(
            self.floor.y.size, self.floor.x.size
        )
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
                "floor_height": points[:, 2],
                "floor_normal": self.point_normals(),
            },
        )
        return path

    def _transform(self, local_points: np.ndarray) -> np.ndarray:
        return (
            (local_points - self.pivot) @ self.state.rotation_matrix.T
            + self.pivot
            + self.state.translation
        )


def write_pvd_collection(
    path: str | Path, files_and_times: list[tuple[Path, float]]
) -> Path:
    """Write a ParaView collection referencing moving-floor VTU files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = "\n".join(
        f'    <DataSet timestep="{time:.17g}" group="" part="0" '
        f'file="{file.relative_to(path.parent).as_posix()}"/>'
        for file, time in files_and_times
    )
    content = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n"
        f"{datasets}\n"
        "  </Collection>\n"
        "</VTKFile>\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _numeric_vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite numeric array")
    return array


def _key_values(value: object, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array))
    if array.ndim != 1 or array.size != count or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a scalar or have len(time) entries")
    return array


def _interpolated_state(
    times: np.ndarray,
    values: dict[str, np.ndarray],
    interval: int,
    alpha: float,
) -> FloorMotionState:
    def interpolate(name: str) -> float:
        data = values[name]
        return float((1.0 - alpha) * data[interval] + alpha * data[interval + 1])

    return FloorMotionState(
        time=float((1.0 - alpha) * times[interval] + alpha * times[interval + 1]),
        indentation=interpolate("indentation"),
        rotation_y_deg=interpolate("floor_rotation_y_deg"),
        rotation_z_deg=interpolate("floor_rotation_z_deg"),
        translation_x=interpolate("floor_translation_x"),
        translation_y=interpolate("floor_translation_y"),
    )

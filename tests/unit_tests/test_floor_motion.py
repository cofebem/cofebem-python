import json

import numpy as np
import pytest

from cofebem.bodies.floor_motion import (
    FloorMotionSchedule,
    FloorMotionState,
    MovingRegularFloor,
    write_pvd_collection,
)
from cofebem.bodies.regular_floor import RegularFloor


def test_schedule_interpolates_interval_counts_and_all_components(tmp_path):
    path = tmp_path / "motion.json"
    path.write_text(
        json.dumps(
            {
                "time": [0, 1, 3],
                "interval_steps": [2, 4],
                "indentation": [0.0, 0.01, 0.02],
                "floor_rotation_y_deg": [0.0, 2.0, -2.0],
                "floor_rotation_z_deg": [0.0, 4.0, 8.0],
                "floor_translation_x": [0.0, 1.0, 3.0],
                "floor_translation_y": 0.5,
            }
        )
    )

    schedule = FloorMotionSchedule.from_json(path)

    assert len(schedule.states) == 7
    arrays = schedule.as_arrays()
    np.testing.assert_allclose(arrays["time"], [0, 0.5, 1, 1.5, 2, 2.5, 3])
    np.testing.assert_allclose(arrays["floor_translation_x"], arrays["time"])
    np.testing.assert_allclose(arrays["floor_translation_y"], 0.5)
    assert arrays["indentation"][-1] == pytest.approx(0.02)


def test_schedule_rejects_bad_interval_count(tmp_path):
    path = tmp_path / "motion.json"
    path.write_text(json.dumps({"time": [0, 1, 2], "interval_steps": [10]}))
    with pytest.raises(ValueError, match=r"len\(time\) - 1"):
        FloorMotionSchedule.from_json(path)


def test_embedded_static_motion_accepts_scalars_without_time():
    schedule = FloorMotionSchedule.from_mapping(
        {
            "indentation": 0.01,
            "floor_rotation_y_deg": 3.0,
            "floor_rotation_z_deg": -2.0,
            "floor_translation_x": 0.001,
            "floor_translation_y": -0.002,
        }
    )

    assert len(schedule.states) == 1
    state = schedule.states[0]
    assert state.indentation == pytest.approx(0.01)
    assert state.rotation_y_deg == pytest.approx(3.0)
    assert state.rotation_z_deg == pytest.approx(-2.0)


def test_flat_floor_exact_tilt_translation_and_torsion():
    floor = RegularFloor.flat((-2.0, 2.0), (-2.0, 2.0), cells=4, level=0.0)
    state = FloorMotionState(
        time=1.0,
        indentation=0.2,
        rotation_y_deg=10.0,
        rotation_z_deg=20.0,
        translation_x=0.1,
        translation_y=-0.2,
    )
    moving = MovingRegularFloor(floor, state, pivot=np.zeros(3))
    points = moving.surface_points()
    recovered = moving.height_at(points[:, :2])

    np.testing.assert_allclose(recovered, points[:, 2], atol=2.0e-15)
    np.testing.assert_allclose(
        np.linalg.norm(moving.point_normals(), axis=1), 1.0, atol=1.0e-15
    )


def test_rough_floor_projection_recovers_transformed_nodes():
    x = np.linspace(-1.0, 1.0, 9)
    y = np.linspace(-1.0, 1.0, 9)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    floor = RegularFloor(x, y, 0.02 * np.sin(xx) * np.cos(yy), kind="rough")
    moving = MovingRegularFloor(
        floor,
        FloorMotionState(
            time=0.0,
            indentation=0.1,
            rotation_y_deg=3.0,
            rotation_z_deg=-4.0,
            translation_x=0.02,
            translation_y=-0.03,
        ),
        pivot=np.zeros(3),
    )
    transformed = moving.surface_points()

    np.testing.assert_allclose(
        moving.height_at(transformed[:, :2]), transformed[:, 2], atol=2.0e-12
    )


def test_moving_floor_vtu_and_pvd(tmp_path):
    meshio = pytest.importorskip("meshio")
    floor = RegularFloor.flat((-1.0, 1.0), (-1.0, 1.0), cells=2)
    moving = MovingRegularFloor(
        floor, FloorMotionState(time=0.5, indentation=0.2), np.zeros(3)
    )
    vtu = moving.write_vtu(tmp_path / "floor_0000.vtu")
    pvd = write_pvd_collection(tmp_path / "floor_motion.pvd", [(vtu, 0.5)])

    mesh = meshio.read(vtu)
    np.testing.assert_allclose(mesh.points[:, 2], 0.2)
    assert "timestep=\"0.5\"" in pvd.read_text()

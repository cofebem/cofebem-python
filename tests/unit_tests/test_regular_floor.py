from pathlib import Path

import numpy as np
import pytest

from cofebem.bodies.regular_floor import RegularFloor, square_floor_bounds


def test_flat_floor_projection_and_gap():
    floor = RegularFloor.flat((-1.0, 1.0), (-2.0, 2.0), cells=4, level=0.25)
    points = np.array([[0.0, 0.0, 1.0], [0.5, -1.0, 0.1]])

    projected = floor.project(points)
    np.testing.assert_allclose(projected[:, :2], points[:, :2])
    np.testing.assert_allclose(projected[:, 2], 0.25)
    np.testing.assert_allclose(floor.initial_gap(points), [0.75, -0.15])
    np.testing.assert_allclose(
        floor.point_normals(),
        np.broadcast_to([0.0, 0.0, 1.0], (5, 5, 3)),
    )


def test_bilinear_projection_is_exact_for_a_plane():
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-2.0, 2.0, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    floor = RegularFloor(x, y, 0.2 * xx - 0.1 * yy + 0.3)
    points = np.array([[-0.7, -1.3], [0.25, 0.6], [1.0, 2.0]])

    np.testing.assert_allclose(
        floor.height_at(points),
        0.2 * points[:, 0] - 0.1 * points[:, 1] + 0.3,
        atol=1.0e-14,
    )


def test_projection_rejects_points_outside_floor():
    floor = RegularFloor.flat((0.0, 1.0), (0.0, 1.0), cells=2)
    with pytest.raises(ValueError, match="outside the floor grid"):
        floor.height_at(np.array([[1.1, 0.5]]))


def test_self_affine_floor_is_reproducible_periodic_and_has_requested_rms():
    kwargs = dict(
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        cells=32,
        level=0.1,
        rms=0.02,
        hurst=0.8,
        k_low=0.05,
        k_high=0.3,
        seed=17,
    )
    first = RegularFloor.self_affine(**kwargs)
    second = RegularFloor.self_affine(**kwargs)

    np.testing.assert_array_equal(first.height, second.height)
    np.testing.assert_array_equal(first.height[-1, :-1], first.height[0, :-1])
    np.testing.assert_array_equal(first.height[:-1, -1], first.height[:-1, 0])
    normals = first.point_normals()
    np.testing.assert_array_equal(normals[-1, :-1], normals[0, :-1])
    np.testing.assert_array_equal(normals[:-1, -1], normals[:-1, 0])
    assert first.height.max() == pytest.approx(0.1)
    assert first.height[:-1, :-1].std() == pytest.approx(0.02)


def test_square_bounds_and_floor_mesh_output(tmp_path):
    bounds = square_floor_bounds(
        np.array([[-1.0, -0.5], [1.0, 0.5]]), margin=0.25
    )
    assert bounds == ((-1.25, 1.25), (-1.25, 1.25))
    floor = RegularFloor.flat(*bounds, cells=3)
    output = floor.write_vtu(tmp_path / "floor.vtu")

    assert output == Path(tmp_path / "floor.vtu")
    meshio = pytest.importorskip("meshio")
    mesh = meshio.read(output)
    assert mesh.points.shape == (16, 3)
    assert mesh.cells[0].data.shape == (9, 4)
    assert set(mesh.point_data) == {"floor_height", "floor_normal"}

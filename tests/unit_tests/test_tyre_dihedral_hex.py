from pathlib import Path

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
pytest.importorskip("gmsh")

from cofebem.mesh.tyre_dihedral_hex import (
    CONTACT_TAG,
    FIXED_TAG,
    ROAD_FACING_ANGLE_DEG,
    SYMMETRIC_CONTACT_TAG,
    build_circumferential_layout,
    generate_tyre_mesh,
    load_mesh_manifest,
)


def test_graded_layout_is_symmetric_and_respects_coarsening_factor():
    layout = build_circumferential_layout(
        12, kind="graded", coarsening_factor=4.0
    )

    assert layout.fine_divisions == 12
    assert layout.transition_divisions > 0
    assert layout.coarse_divisions % 2 == 0
    assert layout.total_divisions < 6 * layout.fine_divisions
    assert layout.actual_size_ratio <= 4.0 * (1.0 + 1.0e-12)
    np.testing.assert_allclose(layout.interval_sizes.sum(), 2.0 * np.pi)

    center = np.deg2rad(ROAD_FACING_ANGLE_DEG)
    reflected = np.sort(np.mod(2.0 * center - layout.angles, 2.0 * np.pi))
    np.testing.assert_allclose(reflected, layout.angles, atol=1.0e-12)


def test_fixed_tag_contains_only_short_disk_edge_curves(tmp_path):
    template = Path(__file__).resolve().parents[2] / "geo_files" / "geometry_v2.geo"
    output = tmp_path / "tyre.msh"
    n_sectors = 8
    generate_tyre_mesh(
        template,
        output,
        axial_divisions=6,
        circumferential_divisions=n_sectors,
        scale=1.0e-3,
    )

    mesh = meshio.read(output)
    fixed_quads = []
    for cells, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cells.type == "quad":
            fixed_quads.append(cells.data[np.asarray(tags) == FIXED_TAG])
    fixed_quads = np.vstack(fixed_quads)
    fixed_points = mesh.points[np.unique(fixed_quads)]

    assert fixed_quads.shape == (2 * n_sectors, 4)
    assert fixed_points.shape[0] == 4 * n_sectors
    np.testing.assert_allclose(
        np.unique(np.abs(fixed_points[:, 0])),
        np.array([0.1495, 0.1525]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.hypot(fixed_points[:, 1], fixed_points[:, 2]),
        0.2286,
        atol=1.0e-12,
    )
    assert "disk_edge_clamp" in mesh.field_data
    assert "bead_clamp" not in mesh.field_data


def test_graded_mesh_uses_fine_surface_grid_and_coarse_tetrahedra(tmp_path):
    template = Path(__file__).resolve().parents[2] / "geo_files" / "geometry_v2.geo"
    output = tmp_path / "tyre_graded.msh"
    axial_divisions = 24
    fine_divisions = 12
    generate_tyre_mesh(
        template,
        output,
        axial_divisions=axial_divisions,
        circumferential_divisions=fine_divisions,
        circumferential_layout="graded",
        coarsening_factor=4.0,
        scale=1.0e-3,
    )

    mesh = meshio.read(output)
    contact_triangles = []
    fixed_triangles = []
    for cells, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cells.type != "triangle":
            continue
        contact_triangles.append(
            cells.data[
                np.isin(
                    np.asarray(tags),
                    [CONTACT_TAG, SYMMETRIC_CONTACT_TAG],
                )
            ]
        )
        fixed_triangles.append(cells.data[np.asarray(tags) == FIXED_TAG])
    contact_points = mesh.points[np.unique(np.vstack(contact_triangles))]
    fixed_points = mesh.points[np.unique(np.vstack(fixed_triangles))]

    angles_deg = np.rad2deg(
        np.arctan2(contact_points[:, 2], contact_points[:, 1])
    )
    fine_angles = np.linspace(-120.0, -60.0, fine_divisions + 1)
    for angle in fine_angles:
        assert np.count_nonzero(np.isclose(angles_deg, angle, atol=1.0e-6)) == (
            axial_divisions + 1
        )

    fine_axial_nodes = np.count_nonzero(
        np.isclose(angles_deg, ROAD_FACING_ANGLE_DEG, atol=1.0e-6)
    )
    coarse_axial_nodes = np.count_nonzero(
        np.isclose(angles_deg, 90.0, atol=1.0e-6)
    )
    assert coarse_axial_nodes < fine_axial_nodes

    assert {cells.type for cells in mesh.cells} == {"triangle", "tetra"}
    np.testing.assert_allclose(
        np.hypot(fixed_points[:, 1], fixed_points[:, 2]),
        0.2286,
        atol=1.0e-10,
    )
    assert np.abs(fixed_points[:, 0]).min() >= 0.1495 - 1.0e-10
    assert np.abs(fixed_points[:, 0]).max() <= 0.1525 + 1.0e-10

    manifest = load_mesh_manifest(output)
    assert manifest is not None
    assert manifest["topology"] == "tetrahedron"
    assert manifest["fine_circumferential_divisions"] == fine_divisions
    assert manifest["symmetric_contact_tag"] == SYMMETRIC_CONTACT_TAG
    assert manifest["quality_min_sicn"]["minimum"] > 0.0

    symmetric_triangles = []
    for cells, tags in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
        if cells.type == "triangle":
            symmetric_triangles.append(
                cells.data[np.asarray(tags) == SYMMETRIC_CONTACT_TAG]
            )
    symmetric_points = mesh.points[np.unique(np.vstack(symmetric_triangles))]
    symmetric_angles = np.rad2deg(
        np.arctan2(symmetric_points[:, 2], symmetric_points[:, 1])
    )
    assert symmetric_angles.min() >= -120.0 - 1.0e-6
    assert symmetric_angles.max() <= -60.0 + 1.0e-6

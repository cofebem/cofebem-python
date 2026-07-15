from pathlib import Path

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
pytest.importorskip("gmsh")

from cofebem.mesh.tyre_dihedral_hex import FIXED_TAG, generate_tyre_mesh


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

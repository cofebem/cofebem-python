"""Structured hexahedral tyre mesh generated from ``geometry_v2.geo``.

The Gmsh template contains a blocked half cross-section and its mirrored copy.
This module meshes the eight planar seed surfaces with quadrilaterals, then
revolves those quads through equally spaced angles about the x axis.  The
resulting full tyre has the discrete dihedral symmetry D_n, where ``n`` is the
number of circumferential divisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MATERIAL_OUTER = 1002
MATERIAL_INNER = 1004
CONTACT_TAG = 201
FIXED_TAG = 202
INNER_SURFACE_TAG = 203
BOUNDARY_CONDITION_ID = "disk_edge_short_curves_v1"

_SEED_SURFACES = (17, 19, 21, 23, 24, 29, 144, 149)
_SURFACE_CORNERS = {
    17: (3, 4, 7, 8),
    19: (4, 5, 6, 7),
    21: (6, 14, 15, 7),
    23: (7, 15, 16, 8),
    24: (3, 4, 25, 29),
    29: (25, 48, 52, 29),
    144: (4, 5, 280, 25),
    149: (25, 280, 303, 48),
}
_SURFACE_MATERIAL = {
    17: MATERIAL_INNER,
    23: MATERIAL_INNER,
    24: MATERIAL_INNER,
    29: MATERIAL_INNER,
    19: MATERIAL_OUTER,
    21: MATERIAL_OUTER,
    144: MATERIAL_OUTER,
    149: MATERIAL_OUTER,
}

# Curves following the axial cross-section. Opposite curves use identical
# counts so that each material strip has a conforming transfinite mesh.
_CROWN_CURVES = (3, 4, 5, 26, 28, 146)
_SHOULDER_CURVES = (10, 11, 12, 31, 35, 151)
_SIDEWALL_CURVES = (13, 14, 15, 32, 34, 152)
_THICKNESS_CURVES = (1, 2, 6, 7, 8, 9, 27, 33, 147, 153)

# Boundary curves of the complete mirrored cross-section.
_OUTER_CURVES = (3, 10, 13, 146, 151, 152)
# Only the two shortest (3 mm) bead-boundary curves represent the outer edge
# of the rigid wheel disk. Curves 9 and 33 are the adjacent 5 mm free bead
# surface and must not receive Dirichlet constraints.
_DISK_EDGE_CURVES = (8, 153)
_INNER_CURVES = (5, 12, 15, 28, 35, 34)


def _allocate_half_axial(gmsh, axial_divisions: int) -> tuple[int, int, int]:
    """Distribute half-width elements over crown, shoulder and sidewall."""
    if axial_divisions < 6 or axial_divisions % 2:
        raise ValueError("axial_divisions must be an even integer >= 6")

    total = axial_divisions // 2
    def curve_length(tag: int) -> float:
        lower, upper = gmsh.model.getParametrizationBounds(1, tag)
        parameters = np.linspace(float(lower[0]), float(upper[0]), 257)
        values = np.asarray(
            gmsh.model.getValue(1, tag, parameters.tolist()), dtype=float
        ).reshape(-1, 3)
        return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())

    lengths = np.array([curve_length(tag) for tag in (3, 10, 13)], dtype=float)
    raw = total * lengths / lengths.sum()
    counts = np.maximum(1, np.floor(raw).astype(int))

    while counts.sum() < total:
        score = raw - counts
        counts[int(np.argmax(score))] += 1
    while counts.sum() > total:
        candidates = np.where(counts > 1)[0]
        if candidates.size == 0:
            raise ValueError("Not enough axial divisions for the template blocks")
        score = counts[candidates] - raw[candidates]
        counts[candidates[int(np.argmax(score))]] -= 1

    return tuple(int(v) for v in counts)


def _elements_on_entities(gmsh, dim: int, tags: tuple[int, ...], element_type: int):
    """Return connectivity and owner entity for one Gmsh element type."""
    connectivity: list[np.ndarray] = []
    owners: list[int] = []
    _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(
        element_type
    )

    for tag in tags:
        types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
        found = False
        for kind, elems, nodes in zip(types, element_tags, node_tags):
            if kind != element_type:
                continue
            found = True
            block = np.asarray(nodes, dtype=np.int64).reshape(
                len(elems), nodes_per_element
            )
            connectivity.append(block)
            owners.extend([tag] * len(elems))
        if not found:
            raise RuntimeError(
                f"Entity ({dim}, {tag}) has no element type {element_type}"
            )

    return np.vstack(connectivity), np.asarray(owners, dtype=np.int32)


def _mesh_cross_section(template: Path, axial_divisions: int):
    """Mesh the template seed surfaces and return planar quads and tagged edges."""
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "The Gmsh Python module is required. Install it in fenicsx-env."
        ) from exc

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(template))

        available = {tag for _, tag in gmsh.model.getEntities(2)}
        missing = set(_SEED_SURFACES) - available
        if missing:
            raise RuntimeError(
                "geometry_v2.geo no longer matches the expected seed surfaces; "
                f"missing tags: {sorted(missing)}"
            )

        crown, shoulder, sidewall = _allocate_half_axial(gmsh, axial_divisions)
        for curves, elements in (
            (_CROWN_CURVES, crown),
            (_SHOULDER_CURVES, shoulder),
            (_SIDEWALL_CURVES, sidewall),
            (_THICKNESS_CURVES, 1),
        ):
            for curve in curves:
                gmsh.model.mesh.setTransfiniteCurve(curve, elements + 1)

        for surface in _SEED_SURFACES:
            gmsh.model.mesh.setTransfiniteSurface(
                surface, cornerTags=list(_SURFACE_CORNERS[surface])
            )
            gmsh.model.mesh.setRecombine(2, surface)

        # The template also contains the old rotational extrusion. Mesh only
        # the planar seed surfaces; their boundaries become the 3-D generators.
        gmsh.model.setVisibility(gmsh.model.getEntities(), 0)
        gmsh.model.setVisibility([(2, tag) for tag in _SEED_SURFACES], 1, recursive=True)
        gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        gmsh.model.mesh.generate(2)

        quads, surface_owner = _elements_on_entities(
            gmsh, 2, _SEED_SURFACES, element_type=3
        )
        outer_edges, _ = _elements_on_entities(
            gmsh, 1, _OUTER_CURVES, element_type=1
        )
        disk_edge_edges, _ = _elements_on_entities(
            gmsh, 1, _DISK_EDGE_CURVES, element_type=1
        )
        inner_edges, _ = _elements_on_entities(
            gmsh, 1, _INNER_CURVES, element_type=1
        )

        used_tags = np.unique(
            np.concatenate(
                [
                    quads.ravel(),
                    outer_edges.ravel(),
                    disk_edge_edges.ravel(),
                    inner_edges.ravel(),
                ]
            )
        )
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(all_coords, dtype=float).reshape(-1, 3)
        tag_to_row = {int(tag): i for i, tag in enumerate(all_tags)}
        points = np.array([coordinates[tag_to_row[int(tag)]] for tag in used_tags])
        compact = {int(tag): i for i, tag in enumerate(used_tags)}

        def remap(array: np.ndarray) -> np.ndarray:
            return np.vectorize(compact.__getitem__, otypes=[np.int64])(array)

        material = np.array(
            [_SURFACE_MATERIAL[int(tag)] for tag in surface_owner], dtype=np.int32
        )
        return (
            points,
            remap(quads),
            material,
            remap(outer_edges),
            remap(disk_edge_edges),
            remap(inner_edges),
            (crown, shoulder, sidewall),
        )
    finally:
        gmsh.finalize()


def _positive_hex_order(points: np.ndarray, quad: np.ndarray, angle: float) -> np.ndarray:
    """Orient a cross-section quad so its first revolved hex has positive Jacobian."""
    q = quad.copy()

    def xyz(indices, theta):
        xr = points[indices, :2]
        return np.column_stack(
            [xr[:, 0], xr[:, 1] * np.cos(theta), xr[:, 1] * np.sin(theta)]
        )

    a = xyz(q, 0.0)
    b = xyz(q, angle)
    determinant = np.linalg.det(
        np.column_stack((a[1] - a[0], a[3] - a[0], b[0] - a[0]))
    )
    if determinant < 0.0:
        q = q[[0, 3, 2, 1]]
    return q


def generate_tyre_mesh(
    template: str | Path,
    output: str | Path,
    *,
    axial_divisions: int = 24,
    circumferential_divisions: int = 32,
    scale: float = 1.0e-3,
) -> Path:
    """Generate a full, tagged, D_n-symmetric hexahedral tyre mesh.

    ``axial_divisions`` is the total number of elements along the outer
    cross-section from one bead to the other. ``circumferential_divisions`` is
    the number of equal sectors around the x axis. ``FIXED_TAG`` marks only
    the two mirrored shortest disk-edge curves (template curves 8 and 153).
    """
    if circumferential_divisions < 4 or circumferential_divisions % 2:
        raise ValueError(
            "circumferential_divisions must be an even integer >= 4"
        )
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    template = Path(template).resolve()
    output = Path(output).resolve()
    if not template.is_file():
        raise FileNotFoundError(template)
    if output.suffix.lower() != ".msh":
        raise ValueError("output must have the .msh extension")

    (
        cross_points,
        cross_quads,
        cross_material,
        outer_edges,
        disk_edge_edges,
        inner_edges,
        axial_split,
    ) = _mesh_cross_section(template, axial_divisions)

    # The template is planar at z=0 and uses y as the radius from the x axis.
    cross_points = cross_points.copy()
    cross_points[:, :2] *= scale
    n_cross = len(cross_points)
    n_theta = int(circumferential_divisions)
    dtheta = 2.0 * np.pi / n_theta

    points = np.empty((n_theta * n_cross, 3), dtype=float)
    for sector in range(n_theta):
        theta = sector * dtheta
        start = sector * n_cross
        points[start : start + n_cross, 0] = cross_points[:, 0]
        points[start : start + n_cross, 1] = cross_points[:, 1] * np.cos(theta)
        points[start : start + n_cross, 2] = cross_points[:, 1] * np.sin(theta)

    oriented_quads = np.array(
        [_positive_hex_order(cross_points, q, dtheta) for q in cross_quads]
    )
    hexes: list[list[int]] = []
    cell_tags: list[int] = []
    for sector in range(n_theta):
        next_sector = (sector + 1) % n_theta
        offset = sector * n_cross
        next_offset = next_sector * n_cross
        for quad, tag in zip(oriented_quads, cross_material):
            hexes.append(
                [
                    *(offset + quad).tolist(),
                    *(next_offset + quad).tolist(),
                ]
            )
            cell_tags.append(int(tag))

    facet_quads: list[list[int]] = []
    facet_tags: list[int] = []
    for edges, tag in (
        (outer_edges, CONTACT_TAG),
        (disk_edge_edges, FIXED_TAG),
        (inner_edges, INNER_SURFACE_TAG),
    ):
        for sector in range(n_theta):
            next_sector = (sector + 1) % n_theta
            offset = sector * n_cross
            next_offset = next_sector * n_cross
            for a, b in edges:
                facet_quads.append(
                    [offset + a, offset + b, next_offset + b, next_offset + a]
                )
                facet_tags.append(tag)

    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("meshio is required to write the tyre mesh") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        ("hexahedron", np.asarray(hexes, dtype=np.int64)),
        ("quad", np.asarray(facet_quads, dtype=np.int64)),
    ]
    physical = [
        np.asarray(cell_tags, dtype=np.int32),
        np.asarray(facet_tags, dtype=np.int32),
    ]
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical},
        field_data={
            "rubber_outer": np.array([MATERIAL_OUTER, 3]),
            "rubber_inner": np.array([MATERIAL_INNER, 3]),
            "road_contact": np.array([CONTACT_TAG, 2]),
            "disk_edge_clamp": np.array([FIXED_TAG, 2]),
            "inner_surface": np.array([INNER_SURFACE_TAG, 2]),
        },
    )
    meshio.write(output, mesh, file_format="gmsh22", binary=False)

    crown, shoulder, sidewall = axial_split
    print(
        f"Wrote {output}: {len(points)} points, {len(hexes)} hexes, "
        f"D_{n_theta} sectors; half-width split="
        f"{crown}+{shoulder}+{sidewall}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "geo_files" / "geometry_v2.geo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "tyre_dihedral" / "tyre_dihedral.msh",
    )
    parser.add_argument("--axial-divisions", type=int, default=24)
    parser.add_argument("--circumferential-divisions", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0e-3)
    args = parser.parse_args()
    generate_tyre_mesh(
        args.template,
        args.output,
        axial_divisions=args.axial_divisions,
        circumferential_divisions=args.circumferential_divisions,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()

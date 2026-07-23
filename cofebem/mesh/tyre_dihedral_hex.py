"""Circumferentially uniform or graded tyre mesh from ``geometry_v2.geo``.

The Gmsh template contains a blocked half cross-section and its mirrored copy.
This module meshes the eight planar seed surfaces with quadrilaterals, then
revolves those quads about the x axis. Uniform spacing gives the original
globally D_n-symmetric hexahedral mesh. The graded path constructs a CAD
partition with a 60-degree road-facing fine zone, 30-degree transition zones,
and a coarse rear zone, then fills the complete tyre with tetrahedra. Using one
cell topology is required by the supported DOLFINx 0.9 assembly path and lets
the mesh coarsen in the axial as well as circumferential direction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np


MATERIAL_OUTER = 1002
MATERIAL_INNER = 1004
CONTACT_TAG = 201
FIXED_TAG = 202
INNER_SURFACE_TAG = 203
SYMMETRIC_CONTACT_TAG = 204
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

FINE_SECTOR_ANGLE_DEG = 60.0
TRANSITION_SECTOR_ANGLE_DEG = 30.0
ROAD_FACING_ANGLE_DEG = -90.0
MESH_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class _AngularZone:
    """One CAD extrusion zone in degrees about the tyre x axis."""

    name: str
    width_deg: float
    start_size_factor: float
    end_size_factor: float


@dataclass(frozen=True)
class CircumferentialLayout:
    """Angular meridians and zone counts for a revolved tyre mesh."""

    angles: np.ndarray
    interval_sizes: np.ndarray
    kind: str
    fine_divisions: int
    transition_divisions: int
    coarse_divisions: int
    coarsening_factor: float

    @property
    def total_divisions(self) -> int:
        return int(self.angles.size)

    @property
    def actual_size_ratio(self) -> float:
        return float(self.interval_sizes.max() / self.interval_sizes.min())


def build_circumferential_layout(
    fine_divisions: int,
    *,
    kind: str = "uniform",
    coarsening_factor: float = 6.0,
) -> CircumferentialLayout:
    """Build uniform or road-facing graded angular meridians.

    For ``kind='graded'``, ``fine_divisions`` is the number of equal elements
    in the 60-degree fine zone. The adjacent 30-degree zones use a linear size
    progression and the remaining 240 degrees use equal coarse elements. The
    complete grid is mirror-symmetric about both the road-facing and opposite
    meridians, and no angular element exceeds ``coarsening_factor`` times the
    fine element angle.
    """
    if fine_divisions < 4 or fine_divisions % 2:
        raise ValueError("circumferential divisions must be an even integer >= 4")
    if kind not in {"uniform", "graded"}:
        raise ValueError("circumferential layout must be 'uniform' or 'graded'")
    if not np.isfinite(coarsening_factor) or coarsening_factor < 1.0:
        raise ValueError("coarsening_factor must be finite and >= 1")

    if kind == "uniform":
        interval = 2.0 * np.pi / fine_divisions
        intervals = np.full(fine_divisions, interval)
        angles = np.arange(fine_divisions, dtype=float) * interval
        return CircumferentialLayout(
            angles=angles,
            interval_sizes=intervals,
            kind=kind,
            fine_divisions=fine_divisions,
            transition_divisions=0,
            coarse_divisions=0,
            coarsening_factor=1.0,
        )

    fine_width = np.deg2rad(FINE_SECTOR_ANGLE_DEG)
    transition_width = np.deg2rad(TRANSITION_SECTOR_ANGLE_DEG)
    coarse_width = 2.0 * np.pi - fine_width - 2.0 * transition_width
    fine_size = fine_width / fine_divisions

    coarse_divisions = max(
        2, int(np.ceil(coarse_width / (coarsening_factor * fine_size)))
    )
    if coarse_divisions % 2:
        coarse_divisions += 1
    coarse_size = coarse_width / coarse_divisions

    estimate = 2.0 * transition_width / (fine_size + coarse_size)
    transition_divisions = max(1, int(np.rint(estimate)))
    last_transition_size = (
        2.0 * transition_width / transition_divisions - fine_size
    )
    while last_transition_size > coarsening_factor * fine_size:
        transition_divisions += 1
        last_transition_size = (
            2.0 * transition_width / transition_divisions - fine_size
        )
    while last_transition_size < fine_size and transition_divisions > 1:
        transition_divisions -= 1
        last_transition_size = (
            2.0 * transition_width / transition_divisions - fine_size
        )

    transition_sizes = np.linspace(
        fine_size, last_transition_size, transition_divisions
    )
    fine_sizes = np.full(fine_divisions, fine_size)
    coarse_sizes = np.full(coarse_divisions, coarse_size)
    intervals = np.concatenate(
        [fine_sizes, transition_sizes, coarse_sizes, transition_sizes[::-1]]
    )
    # Accumulation starts at the left boundary of the fine zone. The last
    # boundary is the periodic duplicate and is intentionally omitted.
    start = np.deg2rad(
        ROAD_FACING_ANGLE_DEG - 0.5 * FINE_SECTOR_ANGLE_DEG
    )
    angles = np.mod(
        start + np.concatenate([[0.0], np.cumsum(intervals[:-1])]),
        2.0 * np.pi,
    )
    order = np.argsort(angles)
    angles = angles[order]
    circular_sizes = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
    if not np.isclose(circular_sizes.sum(), 2.0 * np.pi, atol=1.0e-13):
        raise RuntimeError("graded circumferential intervals do not close")
    if circular_sizes.max() > coarsening_factor * fine_size * (1.0 + 1.0e-12):
        raise RuntimeError("graded circumferential layout exceeds coarsening factor")

    return CircumferentialLayout(
        angles=angles,
        interval_sizes=circular_sizes,
        kind=kind,
        fine_divisions=fine_divisions,
        transition_divisions=transition_divisions,
        coarse_divisions=coarse_divisions,
        coarsening_factor=float(coarsening_factor),
    )


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


def _curve_sample(gmsh, tag: int) -> tuple[np.ndarray, float]:
    """Return the midpoint and sampled length of a CAD curve."""
    lower, upper = gmsh.model.getParametrizationBounds(1, int(tag))
    parameters = np.linspace(float(lower[0]), float(upper[0]), 129)
    values = np.asarray(
        gmsh.model.getValue(1, int(tag), parameters.tolist()), dtype=float
    ).reshape(-1, 3)
    return values[len(values) // 2], float(
        np.linalg.norm(np.diff(values, axis=0), axis=1).sum()
    )


def _curve_reference_tag(
    gmsh,
    curve: int,
    reference_midpoints: dict[int, np.ndarray],
) -> int:
    """Identify a revolved cross-section curve from its x/r midpoint."""
    midpoint, _ = _curve_sample(gmsh, curve)
    reduced = np.array([midpoint[0], np.hypot(midpoint[1], midpoint[2])])
    tags = np.fromiter(reference_midpoints, dtype=np.int64)
    distances = np.array(
        [
            np.linalg.norm(
                reduced
                - np.array(
                    [point[0], np.hypot(point[1], point[2])], dtype=float
                )
            )
            for point in reference_midpoints.values()
        ]
    )
    return int(tags[int(np.argmin(distances))])


def _is_meridional_curve(gmsh, curve: int, *, angle_tol: float = 1.0e-8) -> bool:
    """Return whether a curve lies in one constant-angle meridian plane."""
    boundary = gmsh.model.getBoundary([(1, int(curve))], oriented=False)
    if len(boundary) != 2:
        return False
    points = np.array(
        [gmsh.model.getValue(0, tag, []) for _, tag in boundary], dtype=float
    )
    angles = np.arctan2(points[:, 2], points[:, 1])
    difference = np.arctan2(
        np.sin(angles[1] - angles[0]), np.cos(angles[1] - angles[0])
    )
    return bool(abs(difference) <= angle_tol)


def _graded_zone_specification(coarsening_factor: float) -> tuple[_AngularZone, ...]:
    """Return a closed 360-degree sequence starting before the fine zone."""
    coarse = float(coarsening_factor)
    return (
        _AngularZone("transition_left", 30.0, coarse, 1.0),
        _AngularZone("fine", 60.0, 1.0, 1.0),
        _AngularZone("transition_right", 30.0, 1.0, coarse),
        # Keep built-in-kernel revolutions at or below 60 degrees. Larger
        # sweeps of the thin sidewall blocks can produce intersecting PLC
        # surface facets during tetrahedralisation.
        _AngularZone("coarse_1", 60.0, coarse, coarse),
        _AngularZone("coarse_2", 60.0, coarse, coarse),
        _AngularZone("coarse_3", 60.0, coarse, coarse),
        _AngularZone("coarse_4", 60.0, coarse, coarse),
    )


def _write_mesh_manifest(output: Path, payload: dict[str, object]) -> Path:
    """Write lightweight generation metadata next to a generated mesh."""
    path = output.with_suffix(output.suffix + ".json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_mesh_manifest(mesh_path: str | Path) -> dict[str, object] | None:
    """Load generation metadata when it exists next to ``mesh_path``."""
    path = Path(mesh_path).resolve().with_suffix(Path(mesh_path).suffix + ".json")
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format_version") != MESH_MANIFEST_VERSION:
        return None
    return payload


def _merge_coincident_meshio_points(mesh, *, relative_tol: float = 1.0e-10):
    """Merge the two independently meshed node sets on the periodic seam."""
    extent = float(np.ptp(mesh.points, axis=0).max())
    tolerance = max(relative_tol * extent, np.finfo(float).eps)
    keys = np.rint(mesh.points / tolerance).astype(np.int64)
    _, first, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    if first.size == len(mesh.points):
        return mesh
    mesh.points = np.asarray(mesh.points[first], dtype=float)
    for block in mesh.cells:
        block.data[:] = inverse[block.data]
    return mesh


def _generate_graded_tetra_mesh(
    template: Path,
    output: Path,
    *,
    axial_divisions: int,
    circumferential_divisions: int,
    coarsening_factor: float,
    scale: float,
) -> Path:
    """Generate a conforming, zoned, all-tetrahedral tyre mesh.

    The fine exterior surfaces are transfinite so the requested axial and
    circumferential counts are retained at the road-facing boundary. The
    volume and all transition/coarse surfaces are tetrahedral/triangular and
    use Gmsh point-size interpolation, which permits isotropic axial
    coarsening away from the contact sector.
    """
    try:
        import gmsh
        import meshio
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "The graded tetrahedral mesh requires gmsh and meshio in fenicsx-env"
        ) from exc

    zones = _graded_zone_specification(coarsening_factor)
    start_angle_deg = ROAD_FACING_ANGLE_DEG - (
        0.5 * FINE_SECTOR_ANGLE_DEG + TRANSITION_SECTOR_ANGLE_DEG
    )
    marker = "// Duplicate the surfaces"
    template_text = template.read_text()
    if marker not in template_text:
        raise RuntimeError(
            "geometry_v2.geo does not contain the expected cross-section marker"
        )
    clean_geo = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="tyre_cross_section_",
        suffix=".geo",
        delete=False,
    )
    clean_geo.write(template_text.split(marker, 1)[0])
    clean_geo.close()
    clean_geo_path = Path(clean_geo.name)

    gmsh.initialize()
    raw_path: Path | None = None
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(clean_geo_path))
        available_surfaces = {tag for _, tag in gmsh.model.getEntities(2)}
        base_surfaces = (17, 19, 21, 23)
        missing = set(base_surfaces) - available_surfaces
        if missing:
            raise RuntimeError(
                "geometry_v2.geo no longer matches the expected seed surfaces; "
                f"missing tags: {sorted(missing)}"
            )

        crown, shoulder, sidewall = _allocate_half_axial(
            gmsh, axial_divisions
        )
        base_reference_curves = tuple(range(1, 16))
        base_midpoints = {
            curve: _curve_sample(gmsh, curve)[0]
            for curve in base_reference_curves
        }
        mirrored = gmsh.model.geo.copy([(2, tag) for tag in base_surfaces])
        gmsh.model.geo.rotate(
            mirrored,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            np.pi,
        )
        gmsh.model.geo.removeAllDuplicates()
        gmsh.model.geo.synchronize()
        mirrored_surfaces = [int(tag) for dim, tag in mirrored if dim == 2]
        seed_surfaces = tuple(base_surfaces) + tuple(mirrored_surfaces)
        if set(seed_surfaces) != {
            tag for _, tag in gmsh.model.getEntities(2)
        }:
            raise RuntimeError(
                "Gmsh changed cross-section surface tags during coherence"
            )
        surface_material = {
            **{17: MATERIAL_INNER, 19: MATERIAL_OUTER,
               21: MATERIAL_OUTER, 23: MATERIAL_INNER},
            **{
                surface: material
                for surface, material in zip(
                    mirrored_surfaces,
                    (MATERIAL_INNER, MATERIAL_OUTER,
                     MATERIAL_OUTER, MATERIAL_INNER),
                )
            },
        }
        reference_curves = sorted(
            {
                tag
                for dim, tag in gmsh.model.getBoundary(
                    [(2, tag) for tag in seed_surfaces],
                    combined=False,
                    oriented=False,
                )
                if dim == 1
            }
        )
        reference_midpoints: dict[int, np.ndarray] = {}
        reference_lengths: dict[int, float] = {}
        reference_base_tag: dict[int, int] = {}
        for curve in reference_curves:
            midpoint, length = _curve_sample(gmsh, curve)
            reference_midpoints[curve] = midpoint
            reference_lengths[curve] = length
            reduced = np.array([abs(midpoint[0]), np.hypot(midpoint[1], midpoint[2])])
            reference_base_tag[curve] = min(
                base_midpoints,
                key=lambda base: np.linalg.norm(
                    reduced
                    - np.array(
                        [
                            abs(base_midpoints[base][0]),
                            np.hypot(
                                base_midpoints[base][1],
                                base_midpoints[base][2],
                            ),
                        ]
                    )
                ),
            )

        # Rotate the clean, complete cross-section to the first zone boundary.
        current_surfaces = list(seed_surfaces)
        gmsh.model.geo.rotate(
            [(2, tag) for tag in current_surfaces],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            np.deg2rad(start_angle_deg),
        )
        gmsh.model.geo.synchronize()

        material_volumes: dict[int, set[int]] = {
            MATERIAL_OUTER: set(),
            MATERIAL_INNER: set(),
        }
        contact_surfaces: set[int] = set()
        symmetric_contact_surfaces: set[int] = set()
        fixed_surfaces: set[int] = set()
        inner_surfaces: set[int] = set()
        fine_lateral_surfaces: set[int] = set()
        lateral_surfaces_by_zone: dict[str, set[int]] = {
            zone.name: set() for zone in zones
        }
        zone_volumes: dict[str, set[int]] = {
            zone.name: set() for zone in zones
        }

        for zone in zones:
            gmsh.model.geo.synchronize()
            boundary_data: list[tuple[int, list[int], list[int]]] = []
            for seed_surface, surface in zip(seed_surfaces, current_surfaces):
                boundary_curves = [
                    tag
                    for dim, tag in gmsh.model.getBoundary(
                        [(2, surface)], oriented=False
                    )
                    if dim == 1
                ]
                boundary_reference = [
                    _curve_reference_tag(gmsh, curve, reference_midpoints)
                    for curve in boundary_curves
                ]
                boundary_data.append(
                    (seed_surface, boundary_curves, boundary_reference)
                )
            extrusion = gmsh.model.geo.revolve(
                [(2, surface) for surface in current_surfaces],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                np.deg2rad(zone.width_deg),
            )
            next_surfaces: list[int] = []
            cursor = 0
            for seed_surface, boundary_curves, boundary_reference in boundary_data:
                stop = cursor + len(boundary_curves) + 2
                group = extrusion[cursor:stop]
                cursor = stop
                if len(group) != len(boundary_curves) + 2:
                    raise RuntimeError(
                        "Unexpected Gmsh rotational-extrusion topology for "
                        f"seed surface {seed_surface}: got {len(group)} entities for "
                        f"{len(boundary_curves)} boundary curves"
                    )
                top = int(group[0][1])
                volume = int(group[1][1])
                next_surfaces.append(top)
                material_volumes[surface_material[seed_surface]].add(volume)
                zone_volumes[zone.name].add(volume)
                for lateral in group[2:]:
                    lateral_tag = int(lateral[1])
                    lateral_surfaces_by_zone[zone.name].add(lateral_tag)
                    if zone.name == "fine":
                        fine_lateral_surfaces.add(lateral_tag)
            if cursor != len(extrusion):
                raise RuntimeError(
                    "Unexpected trailing entities in Gmsh rotational extrusion"
                )
            current_surfaces = next_surfaces

        # The last top surface is geometrically coincident with the first
        # source surface. Keep both during tetrahedralisation: built-in-kernel
        # coherence across this many thin blocks can create intersecting PLC
        # facets. The two identical seam node sets are merged after export.
        gmsh.model.geo.synchronize()

        for zone_name, lateral_surfaces in lateral_surfaces_by_zone.items():
            for surface in lateral_surfaces:
                meridional = [
                    curve
                    for dim, curve in gmsh.model.getBoundary(
                        [(2, surface)], oriented=False
                    )
                    if dim == 1 and _is_meridional_curve(gmsh, curve)
                ]
                if not meridional:
                    continue
                reference = _curve_reference_tag(
                    gmsh, meridional[0], reference_midpoints
                )
                base_curve = reference_base_tag[reference]
                if base_curve in (3, 10, 13):
                    if zone_name == "fine":
                        symmetric_contact_surfaces.add(surface)
                    else:
                        contact_surfaces.add(surface)
                elif base_curve == 8:
                    fixed_surfaces.add(surface)
                elif base_curve in (5, 12, 15):
                    inner_surfaces.add(surface)

        entities_by_dimension = {
            dim: {tag for _, tag in gmsh.model.getEntities(dim)}
            for dim in (1, 2, 3)
        }
        contact_surfaces &= entities_by_dimension[2]
        symmetric_contact_surfaces &= entities_by_dimension[2]
        fixed_surfaces &= entities_by_dimension[2]
        inner_surfaces &= entities_by_dimension[2]
        fine_lateral_surfaces &= entities_by_dimension[2]
        for tags in material_volumes.values():
            tags.intersection_update(entities_by_dimension[3])
        for tags in zone_volumes.values():
            tags.intersection_update(entities_by_dimension[3])
        if (
            not contact_surfaces
            or not symmetric_contact_surfaces
            or not fixed_surfaces
            or not inner_surfaces
        ):
            raise RuntimeError(
                "Failed to preserve the tyre boundary tags while closing the CAD"
            )
        if any(not tags for tags in material_volumes.values()):
            raise RuntimeError("Failed to preserve tyre material volumes")

        counts_by_curve: dict[int, int] = {}
        for curves, count in (
            ((3, 4, 5), crown),
            ((10, 11, 12), shoulder),
            ((13, 14, 15), sidewall),
            ((1, 2, 6, 7, 8, 9), 1),
        ):
            counts_by_curve.update({curve: count for curve in curves})

        # Each fine lateral surface is the sweep of one cross-section curve.
        # Its two angular edges receive the requested 60-degree count, while
        # meridional edges retain the axial split used by the original mesh.
        configured_curves: set[int] = set()
        for surface in fine_lateral_surfaces:
            curves = [
                tag
                for dim, tag in gmsh.model.getBoundary(
                    [(2, surface)], oriented=False
                )
                if dim == 1
            ]
            for curve in curves:
                if curve in configured_curves:
                    continue
                if _is_meridional_curve(gmsh, curve):
                    reference = _curve_reference_tag(
                        gmsh, curve, reference_midpoints
                    )
                    elements = counts_by_curve.get(
                        reference_base_tag[reference]
                    )
                    if elements is None:
                        continue
                else:
                    elements = circumferential_divisions
                gmsh.model.mesh.setTransfiniteCurve(curve, int(elements) + 1)
                configured_curves.add(curve)
            gmsh.model.mesh.setTransfiniteSurface(surface)

        outer_length = sum(
            reference_lengths[tag]
            for tag in reference_curves
            if reference_base_tag[tag] in (3, 10, 13)
        )
        outer_radius = max(
            np.hypot(point[1], point[2])
            for point in reference_midpoints.values()
        )
        fine_axial_size = outer_length / axial_divisions
        fine_angular_size = (
            outer_radius
            * np.deg2rad(FINE_SECTOR_ANGLE_DEG)
            / circumferential_divisions
        )
        fine_size = float(np.sqrt(fine_axial_size * fine_angular_size))
        # Very coarse chords on the highly curved rear surface can intersect
        # neighbouring thin-layer facets. The curvature cap is normally
        # inactive for production densities and makes the user factor an
        # upper bound, consistent with the previous graded layout.
        coarse_size = float(
            min(coarsening_factor * fine_size, 0.15 * outer_radius)
        )
        actual_coarsening = coarse_size / fine_size

        # CAD points occur at zone interfaces. Linear interpolation of their
        # target sizes supplies the transition grading; all points bounding a
        # rear volume carry the coarse size, so axial resolution coarsens too.
        for _, point in gmsh.model.getEntities(0):
            xyz = np.asarray(gmsh.model.getValue(0, point, []), dtype=float)
            angle = np.rad2deg(np.arctan2(xyz[2], xyz[1]))
            relative = (angle - start_angle_deg) % 360.0
            cursor = 0.0
            factor = float(actual_coarsening)
            for zone in zones:
                end = cursor + zone.width_deg
                if relative <= end + 1.0e-8:
                    fraction = np.clip(
                        (relative - cursor) / zone.width_deg, 0.0, 1.0
                    )
                    factor = min(actual_coarsening, (
                        (1.0 - fraction) * zone.start_size_factor
                        + fraction * zone.end_size_factor
                    ))
                    break
                cursor = end
            gmsh.model.mesh.setSize([(0, point)], fine_size * factor)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * fine_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.05 * coarse_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        # Delaunay is more robust than HXT on the template's very thin,
        # multi-block rubber layers and still feeds the Netgen optimiser.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)

        for tag, name in (
            (MATERIAL_OUTER, "rubber_outer"),
            (MATERIAL_INNER, "rubber_inner"),
        ):
            gmsh.model.addPhysicalGroup(3, sorted(material_volumes[tag]), tag)
            gmsh.model.setPhysicalName(3, tag, name)
        for tag, name, surfaces in (
            (CONTACT_TAG, "road_contact", contact_surfaces),
            (
                SYMMETRIC_CONTACT_TAG,
                "road_contact_symmetric_patch",
                symmetric_contact_surfaces,
            ),
            (FIXED_TAG, "disk_edge_clamp", fixed_surfaces),
            (INNER_SURFACE_TAG, "inner_surface", inner_surfaces),
        ):
            gmsh.model.addPhysicalGroup(2, sorted(surfaces), tag)
            gmsh.model.setPhysicalName(2, tag, name)

        gmsh.model.mesh.generate(3)
        tetra_type = gmsh.model.mesh.getElementType("tetrahedron", 1)
        tetra_tags, _ = gmsh.model.mesh.getElementsByType(tetra_type)
        if len(tetra_tags) == 0:
            raise RuntimeError("Gmsh produced no tetrahedra")
        qualities = np.asarray(
            gmsh.model.mesh.getElementQualities(
                tetra_tags.tolist(), "minSICN"
            ),
            dtype=float,
        )
        if qualities.size and float(qualities.min()) <= 0.0:
            raise RuntimeError(
                "The generated tetrahedral tyre contains inverted elements"
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}_unscaled_",
            suffix=".msh",
            dir=output.parent,
            delete=False,
        )
        temporary.close()
        raw_path = Path(temporary.name)
        gmsh.write(str(raw_path))
        node_count = int(gmsh.model.mesh.getNodes()[0].size)
        tetra_count = int(len(tetra_tags))
        triangle_type = gmsh.model.mesh.getElementType("triangle", 1)
        triangle_tags, _ = gmsh.model.mesh.getElementsByType(triangle_type)
        triangle_count = int(len(triangle_tags))
        quality_summary = {
            "minimum": float(qualities.min(initial=1.0)),
            "mean": float(qualities.mean()) if qualities.size else 1.0,
            "p01": float(np.quantile(qualities, 0.01)) if qualities.size else 1.0,
        }
    finally:
        gmsh.finalize()
        clean_geo_path.unlink(missing_ok=True)

    try:
        mesh = meshio.read(raw_path)
        mesh = _merge_coincident_meshio_points(mesh)
        mesh.points *= scale
        node_count = len(mesh.points)
        meshio.write(output, mesh, file_format="gmsh22", binary=False)
    finally:
        if raw_path is not None:
            raw_path.unlink(missing_ok=True)

    _write_mesh_manifest(
        output,
        {
            "format_version": MESH_MANIFEST_VERSION,
            "topology": "tetrahedron",
            "circumferential_layout": "graded",
            "axial_divisions": axial_divisions,
            "fine_circumferential_divisions": circumferential_divisions,
            "coarsening_factor": coarsening_factor,
            "fine_sector_angle_deg": FINE_SECTOR_ANGLE_DEG,
            "symmetric_contact_tag": SYMMETRIC_CONTACT_TAG,
            "transition_sector_angle_deg": TRANSITION_SECTOR_ANGLE_DEG,
            "scale": scale,
            "nodes": node_count,
            "tetrahedra": tetra_count,
            "boundary_triangles": triangle_count,
            "quality_min_sicn": quality_summary,
            "fine_target_size_unscaled": fine_size,
            "coarse_target_size_unscaled": coarse_size,
        },
    )
    print(
        f"Wrote {output}: {node_count} points, {tetra_count} tetrahedra, "
        f"60 deg fine + 30 deg transitions, target size ratio="
        f"{coarsening_factor:g}; tetra minSICN min="
        f"{quality_summary['minimum']:.3g}, p01={quality_summary['p01']:.3g}, "
        f"mean={quality_summary['mean']:.3g}; half-width split="
        f"{crown}+{shoulder}+{sidewall}"
    )
    return output


def generate_tyre_mesh(
    template: str | Path,
    output: str | Path,
    *,
    axial_divisions: int = 24,
    circumferential_divisions: int = 32,
    circumferential_layout: str = "uniform",
    coarsening_factor: float = 6.0,
    scale: float = 1.0e-3,
) -> Path:
    """Generate a full tagged uniform-hex or graded-tetra tyre mesh.

    ``axial_divisions`` is the total number of elements along the outer
    cross-section from one bead to the other. In the uniform layout,
    ``circumferential_divisions`` is the full-circle count. In the graded
    layout it is the count inside the 60-degree fine zone. The graded volume
    uses tetrahedra so both axial and circumferential sizes can coarsen without
    elongated cells. ``FIXED_TAG`` marks only the two mirrored shortest
    disk-edge curves (template curves 8 and 153).
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    angular_layout = build_circumferential_layout(
        circumferential_divisions,
        kind=circumferential_layout,
        coarsening_factor=coarsening_factor,
    )

    template = Path(template).resolve()
    output = Path(output).resolve()
    if not template.is_file():
        raise FileNotFoundError(template)
    if output.suffix.lower() != ".msh":
        raise ValueError("output must have the .msh extension")

    if angular_layout.kind == "graded":
        return _generate_graded_tetra_mesh(
            template,
            output,
            axial_divisions=axial_divisions,
            circumferential_divisions=circumferential_divisions,
            coarsening_factor=coarsening_factor,
            scale=scale,
        )

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
    angles = angular_layout.angles
    n_theta = angular_layout.total_divisions

    points = np.empty((n_theta * n_cross, 3), dtype=float)
    for sector, theta in enumerate(angles):
        start = sector * n_cross
        points[start : start + n_cross, 0] = cross_points[:, 0]
        points[start : start + n_cross, 1] = cross_points[:, 1] * np.cos(theta)
        points[start : start + n_cross, 2] = cross_points[:, 1] * np.sin(theta)

    oriented_quads = np.array(
        [
            _positive_hex_order(
                cross_points, q, float(angular_layout.interval_sizes.min())
            )
            for q in cross_quads
        ]
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

    _write_mesh_manifest(
        output,
        {
            "format_version": MESH_MANIFEST_VERSION,
            "topology": "hexahedron",
            "circumferential_layout": "uniform",
            "axial_divisions": axial_divisions,
            "fine_circumferential_divisions": circumferential_divisions,
            "total_circumferential_divisions": n_theta,
            "coarsening_factor": 1.0,
            "scale": scale,
            "nodes": len(points),
            "hexahedra": len(hexes),
        },
    )

    crown, shoulder, sidewall = axial_split
    if angular_layout.kind == "uniform":
        layout_description = f"uniform D_{n_theta} sectors"
    else:
        layout_description = (
            f"graded sectors={n_theta} "
            f"(fine={angular_layout.fine_divisions}, "
            f"transition={angular_layout.transition_divisions}+"
            f"{angular_layout.transition_divisions}, "
            f"coarse={angular_layout.coarse_divisions}, "
            f"size ratio={angular_layout.actual_size_ratio:.3g})"
        )
    print(
        f"Wrote {output}: {len(points)} points, {len(hexes)} hexes, "
        f"{layout_description}; half-width split="
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
    parser.add_argument(
        "--circumferential-layout",
        choices=("uniform", "graded"),
        default="uniform",
    )
    parser.add_argument("--coarsening-factor", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=1.0e-3)
    args = parser.parse_args()
    generate_tyre_mesh(
        args.template,
        args.output,
        axial_divisions=args.axial_divisions,
        circumferential_divisions=args.circumferential_divisions,
        circumferential_layout=args.circumferential_layout,
        coarsening_factor=args.coarsening_factor,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()

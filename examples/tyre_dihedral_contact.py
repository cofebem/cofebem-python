"""Tyre-road contact with H-matrix or factorized-FE compliance actions.

The mesh is generated from ``geo_files/geometry_v2.geo`` as a uniform hex tyre
or a road-facing graded tetrahedral tyre. The H-matrix strategy samples compact
scalar normal-compliance data at one reference meridian, then answers ACA
queries using global dihedral symmetry or an explicitly tagged regular local
patch. The flexibility-matrix-free strategy instead applies compliance by
back-solving the factorized FE stiffness for every PPCG request. Neither path
constructs a global dense compliance.

Example
-------
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
    -in examples/input.json

conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
    --axial-divisions 24 --circumferential-divisions 32 --regenerate
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import mmap
from pathlib import Path
import resource
from time import perf_counter, process_time
import warnings

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.io import VTKFile, gmshio
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
    sym,
    tr,
)

from cofebem.fenics.dihedral_compliance import (
    DihedralComplianceEntrySource,
    FactorizedComplianceEntrySource,
    FactorizedComplianceOperator,
    IterativeComplianceOperator,
    LocalDihedralComplianceEntrySource,
    create_lu_solver,
    create_iterative_solver,
    dilate_sector_axial_mask,
    dihedral_reflection_error,
    infer_meridian_shape,
    load_dihedral_compliance_archive,
    memory_map_compliance_samples,
    order_contact_meridians,
    order_contact_sectors,
    order_unstructured_contact,
    potential_contact_indices,
    probe_spd_operator,
    restricted_source_clearance,
    sample_reference_normal_compliance,
    validate_local_dihedral_compliance,
)
from cofebem.fenics.petsc_schur import MumpsSchurComplianceOperator
from cofebem.fenics.contact_postprocess import (
    ContactStressProjector,
    EquilibratedContactStressProjector,
    force_based_contact_pressure,
    project_compressive_normal_stress,
    surface_lumped_nodal_areas,
)
from cofebem.fenics.tyre_contact_input import (
    load_tyre_contact_input,
    validated_argument_defaults,
)
from cofebem.hmatrices import HMatrix, IndexedEntrySource
from cofebem.lcp import (
    LCP,
    RestrictedProjectedPreconditioner,
    SectorSurfaceSpectralPreconditioner,
    SurfaceAreaDiagonalPreconditioner,
    solve,
)
from cofebem.mesh.tyre_dihedral_hex import (
    BOUNDARY_CONDITION_ID,
    CONTACT_TAG,
    FINE_SECTOR_ANGLE_DEG,
    FIXED_TAG,
    INNER_SURFACE_TAG,
    ROAD_FACING_ANGLE_DEG,
    SYMMETRIC_CONTACT_TAG,
    build_circumferential_layout,
    generate_tyre_mesh,
    load_mesh_manifest,
)
from cofebem.bodies.regular_floor import RegularFloor
from cofebem.bodies.floor_motion import (
    FloorMotionSchedule,
    FloorMotionState,
    MovingRegularFloor,
    write_pvd_collection,
)


def _peak_rss_bytes() -> int:
    """Return this process's Linux maximum resident set size in bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


HMATRIX_STRATEGIES = frozenset({"hmatrix", "hmatrix_full"})


def _uses_hmatrix(strategy: str) -> bool:
    return strategy in HMATRIX_STRATEGIES


def _surface_vertices(mesh, facets: np.ndarray) -> np.ndarray:
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    return np.unique(
        np.concatenate([facet_to_vertex.links(int(facet)) for facet in facets])
    )


def _validate_disk_edge_facets(
    mesh, facets: np.ndarray, n_sectors: int | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the two mirrored, one-element-wide disk-edge strips."""
    vertices = _surface_vertices(mesh, facets)
    points = mesh.geometry.x[vertices]
    length_scale = max(float(np.ptp(mesh.geometry.x, axis=0).max()), 1.0e-15)
    radius = np.hypot(points[:, 1], points[:, 2])
    if float(np.ptp(radius)) > 1.0e-10 * length_scale:
        raise ValueError("Disk-edge vertices do not lie on one cylindrical radius")

    if n_sectors is None:
        if not (np.any(points[:, 0] < 0.0) and np.any(points[:, 0] > 0.0)):
            raise ValueError("Disk-edge tag must include both mirrored axial edges")
        absolute_x = np.abs(points[:, 0])
        x_span = np.array([absolute_x.min(), absolute_x.max()])
        if x_span[1] <= x_span[0]:
            raise ValueError("Disk-edge tag must span a nonzero axial strip")
        return vertices, x_span, float(radius.mean())

    expected_facets = 2 * n_sectors
    if facets.size != expected_facets:
        raise ValueError(
            "Disk-edge tag must contain exactly the two shortest boundary "
            f"curves ({expected_facets} facets), found {facets.size}"
        )
    expected_vertices = 4 * n_sectors
    if vertices.size != expected_vertices:
        raise ValueError(
            "Disk-edge tag must span four circumferential node rings "
            f"({expected_vertices} vertices), found {vertices.size}"
        )

    x_levels = np.unique(np.round(points[:, 0] / length_scale, decimals=10))
    absolute_x_levels = np.unique(np.round(np.abs(x_levels), decimals=10))
    if x_levels.size != 4 or absolute_x_levels.size != 2:
        raise ValueError(
            "Disk-edge tag must contain two axial rings on each mirrored edge"
        )
    return vertices, x_levels * length_scale, float(radius.mean())


def _match_component_dofs(
    reference_points: np.ndarray,
    component_points: np.ndarray,
    component_dofs: np.ndarray,
) -> np.ndarray:
    """Match collapsed scalar-space DOFs by coordinate without layout assumptions."""
    def key(point):
        return tuple(np.round(point, decimals=12))

    lookup = {key(component_points[dof]): int(dof) for dof in component_dofs}
    try:
        return np.array([lookup[key(point)] for point in reference_points], dtype=np.int32)
    except KeyError as exc:
        raise RuntimeError("Could not align collapsed y/z component DOFs") from exc


def _contact_scalar_field(space, dofs, values, name):
    field = fem.Function(space)
    field.name = name
    field.x.array[:] = 0.0
    field.x.array[np.asarray(dofs, dtype=np.int32)] = np.asarray(values)
    field.x.scatter_forward()
    return field


def _set_contact_scalar_field(field, dofs, values) -> None:
    """Update a reusable scalar output field without allocating a new function."""
    field.x.array[:] = 0.0
    field.x.array[np.asarray(dofs, dtype=np.int32)] = np.asarray(values)
    field.x.scatter_forward()


def _dilate_unstructured_point_mask(
    mask: np.ndarray,
    points: np.ndarray,
    *,
    halo: int,
    neighbours: int = 8,
) -> np.ndarray:
    """Expand selected unstructured surface nodes through nearest neighbours."""
    selected = np.asarray(mask, dtype=bool).reshape(-1).copy()
    points = np.asarray(points, dtype=float)
    if selected.shape != (len(points),):
        raise ValueError("mask and point count must agree")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if halo == 0 or not np.any(selected):
        return selected
    from scipy.spatial import cKDTree

    count = min(int(neighbours) + 1, len(points))
    neighbours_by_point = cKDTree(points).query(points, k=count)[1]
    if neighbours_by_point.ndim == 1:
        neighbours_by_point = neighbours_by_point[:, None]
    for _ in range(halo):
        expanded = selected.copy()
        expanded[neighbours_by_point[selected].reshape(-1)] = True
        selected = expanded
    return selected


def _compliance_sample_archive_fields(samples, output_dir: Path) -> dict:
    """Store a memmap reference instead of embedding multi-gigabyte samples."""
    filename = getattr(samples, "filename", None)
    if filename is None:
        return {"samples": samples}
    sample_path = Path(filename).expanduser().resolve()
    try:
        stored_path = sample_path.relative_to(output_dir.resolve())
    except ValueError:
        stored_path = sample_path
    return {"samples_file": str(stored_path)}


def _discard_memmap_pages(samples) -> bool:
    """Advise the kernel to reclaim file-backed compliance pages if needed."""
    mapping = getattr(samples, "_mmap", None)
    advice = getattr(mmap, "MADV_DONTNEED", None)
    if mapping is None or not hasattr(mapping, "madvise") or advice is None:
        return False
    mapping.madvise(advice)
    return True


def _assemble_elasticity(
    mesh,
    disk_edge_facets,
    facet_tags,
    young_modulus,
    poisson_ratio,
    inflation_pressure,
):
    tdim = mesh.topology.dim
    fdim = tdim - 1
    V = fem.functionspace(mesh, ("Lagrange", 1, (tdim,)))
    u = TrialFunction(V)
    v = TestFunction(V)

    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))

    def epsilon(w):
        return sym(grad(w))

    def sigma(w):
        return lmbda * tr(epsilon(w)) * Identity(tdim) + 2.0 * mu * epsilon(w)

    a = inner(sigma(u), epsilon(v)) * dx
    fixed_dofs = fem.locate_dofs_topological(V, fdim, disk_edge_facets)
    bc = fem.dirichletbc(
        np.zeros(tdim, dtype=PETSc.ScalarType), fixed_dofs, V
    )
    a_form = fem.form(a)
    A = assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    normal = FacetNormal(mesh)
    pressure = fem.Constant(mesh, PETSc.ScalarType(inflation_pressure))
    L_pressure = inner(-pressure * normal, v) * ds(INNER_SURFACE_TAG)
    pressure_rhs = assemble_vector(fem.form(L_pressure))
    apply_lifting(pressure_rhs, [a_form], bcs=[[bc]])
    pressure_rhs.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    set_bc(pressure_rhs, [bc])
    return V, A, pressure_rhs


def _contact_component_data(V, mesh, contact_facets):
    fdim = mesh.topology.dim - 1
    Wy, Wy_to_V = V.sub(1).collapse()
    Wz, Wz_to_V = V.sub(2).collapse()
    dofs_y = np.asarray(
        fem.locate_dofs_topological(Wy, fdim, contact_facets), dtype=np.int32
    )
    dofs_z = np.asarray(
        fem.locate_dofs_topological(Wz, fdim, contact_facets), dtype=np.int32
    )
    points_y = Wy.tabulate_dof_coordinates().reshape(-1, 3)
    points_z = Wz.tabulate_dof_coordinates().reshape(-1, 3)
    matched_y = _match_component_dofs(points_z[dofs_z], points_y, dofs_y)
    return (
        Wz,
        dofs_z,
        np.asarray(Wy_to_V, dtype=np.int32)[matched_y],
        np.asarray(Wz_to_V, dtype=np.int32)[dofs_z],
        points_z[dofs_z],
    )


def _parse_args(argv=None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    input_parser = argparse.ArgumentParser(add_help=False)
    input_parser.add_argument("-in", "--input", dest="input_file", type=Path)
    input_args, _ = input_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-in",
        "--input",
        dest="input_file",
        type=Path,
        help="complete structured JSON input; explicit CLI options override it",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "geo_files" / "geometry_v2.geo",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=root / "results" / "tyre_dihedral" / "tyre_dihedral.msh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory (default: directory containing the mesh)",
    )
    parser.add_argument("--axial-divisions", type=int, default=24)
    parser.add_argument("--circumferential-divisions", type=int, default=32)
    parser.add_argument(
        "--circumferential-layout",
        choices=("uniform", "graded"),
        default="uniform",
        help=(
            "uniform full-circle sectors or a 60-degree road-facing fine "
            "zone with 30-degree transitions"
        ),
    )
    parser.add_argument(
        "--coarsening-factor",
        type=float,
        default=6.0,
        help="maximum coarse-to-fine tetrahedral target-size ratio",
    )
    parser.add_argument("--scale", type=float, default=1.0e-3)
    parser.add_argument(
        "--regenerate", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--indentation", type=float, default=5.0e-3)
    parser.add_argument(
        "--rotate-floor",
        type=float,
        default=0.0,
        metavar="DEGREES",
        help="rotate the floor about global OY through the point below the tyre centre",
    )
    parser.add_argument(
        "--torsion-floor",
        type=float,
        default=0.0,
        metavar="DEGREES",
        help="rotate the floor about global OZ after the OY slope rotation",
    )
    parser.add_argument("--floor-translation-x", type=float, default=0.0)
    parser.add_argument("--floor-translation-y", type=float, default=0.0)
    parser.add_argument(
        "--motion-file",
        type=Path,
        metavar="JSON",
        help="linearly interpolated floor-motion schedule",
    )
    parser.add_argument(
        "--floor",
        dest="floor_kind",
        choices=("flat", "rough"),
        default="flat",
        help="regular rigid floor type (vertical-contact height field)",
    )
    parser.add_argument("--floor-level", type=float, default=0.0)
    parser.add_argument("--floor-grid-size", type=int, default=256)
    parser.add_argument("--floor-margin", type=float, default=2.0e-2)
    parser.add_argument("--roughness-rms", type=float, default=2.0e-4)
    parser.add_argument("--roughness-hurst", type=float, default=0.8)
    parser.add_argument("--roughness-k-low", type=float, default=0.03)
    parser.add_argument("--roughness-k-high", type=float, default=0.3)
    parser.add_argument("--roughness-seed", type=int, default=42)
    parser.add_argument(
        "--roughness-plateau",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--roughness-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--young-modulus", type=float, default=2.5e8)
    parser.add_argument("--poisson-ratio", type=float, default=0.48)
    parser.add_argument("--inflation-pressure", type=float, default=1.5e5)
    parser.add_argument(
        "--compliance-strategy",
        choices=(
            "hmatrix",
            "hmatrix_full",
            "fe_matrix_free",
            "fe_iterative",
            "mumps_schur",
        ),
        default="hmatrix",
        help=(
            "use a sampled dihedral H-matrix, a direct or iterative full FE "
            "action, or a MUMPS selected-Schur factor"
        ),
    )
    parser.add_argument(
        "--contact-solver",
        choices=("ppcg", "ccg_v2", "ccg"),
        default="ppcg",
    )
    parser.add_argument(
        "--pcg-preconditioner",
        choices=("spectral", "none"),
        default="spectral",
    )
    parser.add_argument("--pcg-zero-mode-factor", type=float, default=1.0)
    parser.add_argument(
        "--pcg-beta-method",
        choices=("pr_plus", "fletcher_reeves"),
        default="pr_plus",
    )
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--h-leaf-size", type=int, default=32)
    parser.add_argument("--h-eta", type=float, default=1.0)
    parser.add_argument("--h-tol", type=float, default=1.0e-7)
    parser.add_argument("--h-max-rank", type=int, default=50)
    parser.add_argument("--h-split", choices=("pca", "kd"), default="pca")
    parser.add_argument(
        "--local-symmetry-tag",
        type=int,
        default=None,
        help=(
            "restrict the sampled symmetry H-matrix and contact solve to this "
            "regular open-patch facet tag (graded meshes use tag 204)"
        ),
    )
    parser.add_argument(
        "--local-symmetry-validation-columns", type=int, default=4
    )
    parser.add_argument("--local-symmetry-tolerance", type=float, default=5.0e-2)
    parser.add_argument(
        "--local-symmetry-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="fail instead of warn when direct-FE local-symmetry validation fails",
    )
    parser.add_argument(
        "--warning-distance",
        type=float,
        default=2.0e-2,
        help=(
            "maximum free gap admitted to the potential contact zone "
            "(default: 0.02; use inf for the full surface)"
        ),
    )
    parser.add_argument(
        "--warning-halo",
        type=int,
        default=1,
        help="surface-neighbour halo added around verification violations",
    )
    parser.add_argument(
        "--warning-max-rounds",
        type=int,
        default=5,
        help="maximum restricted solve/verification rounds",
    )
    parser.add_argument(
        "--warning-verification-tol",
        type=float,
        default=1.0e-7,
        help="allowed negative clearance outside the potential zone",
    )
    parser.add_argument("--factor-solver-type", default=None)
    parser.add_argument("--fe-ksp-type", default="cg")
    parser.add_argument("--fe-pc-type", default="gamg")
    parser.add_argument("--fe-rtol", type=float, default=1.0e-10)
    parser.add_argument("--fe-atol", type=float, default=1.0e-14)
    parser.add_argument("--fe-max-iter", type=int, default=2000)
    parser.add_argument("--fe-options-prefix", default="cofebem_fe_")
    parser.add_argument(
        "--schur-factor-type", choices=("lu", "cholesky"), default="lu"
    )
    parser.add_argument("--schur-max-memory-gib", type=float, default=4.0)
    parser.add_argument(
        "--mmap-compliance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="store sampled compliance in an on-disk NPY memory map",
    )
    compliance_load_group = parser.add_mutually_exclusive_group()
    compliance_load_group.add_argument(
        "--load-compliance",
        type=Path,
        metavar="PATH",
        help=(
            "load reference-meridian samples from a previous compliance.npz "
            "and skip the compliance sampling solves"
        ),
    )
    compliance_load_group.add_argument(
        "--no-load-compliance",
        dest="load_compliance",
        action="store_const",
        const=None,
        help="ignore a compliance archive configured in the JSON input",
    )
    parser.add_argument(
        "--sampling-only", action=argparse.BooleanOptionalAction, default=False
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--no-progress", dest="no_progress", action="store_true"
    )
    progress_group.add_argument(
        "--progress", dest="no_progress", action="store_false"
    )
    parser.set_defaults(no_progress=False)
    parser.add_argument(
        "--stress-projection",
        choices=("lumped", "consistent"),
        default="lumped",
        help="low-memory lumped or factorized consistent stress recovery",
    )
    parser.add_argument(
        "--stress-recovery",
        choices=("equilibrated", "nodal_average", "raw"),
        default="equilibrated",
        help=(
            "weak-equilibrium boundary traction (recommended), volume "
            "nodal-patch recovery, or the oscillatory raw CG1 stress trace"
        ),
    )
    parser.add_argument(
        "--write-vtk", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-volume-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store full volume displacement arrays in every step NPZ",
    )

    input_data = None
    if input_args.input_file is not None:
        try:
            input_data = load_tyre_contact_input(input_args.input_file)
            parser.set_defaults(
                **validated_argument_defaults(
                    parser, input_data.argument_defaults
                )
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    args.embedded_motion = None if input_data is None else input_data.motion
    args.input_file = None if input_data is None else input_data.path
    return args


def _run_motion_history(
    *,
    args,
    comm,
    mesh,
    facet_tags,
    contact_surface_tags,
    V,
    A,
    ksp,
    pressure_rhs,
    Wz,
    ordering,
    full_points,
    verification_points,
    verification_parent_z,
    moving_floors,
    output_dir,
    motion,
    floor_pivot,
    full_gaps,
    inflation_values,
    compliance_source,
    samples,
    floor_build_seconds,
    floor_build_cpu_seconds,
    factorization_seconds,
    factorization_cpu_seconds,
    inflation_solve_seconds,
    inflation_solve_cpu_seconds,
    compliance_load_seconds,
    compliance_load_cpu_seconds,
    compliance_sampling_seconds,
    compliance_sampling_cpu_seconds,
    sampling_solves,
    strategy_start,
    strategy_cpu_start,
    reference_vertical_shift,
) -> None:
    """Solve an interpolated floor history with one assembled/factorized FE model."""
    contact_parent_dofs = ordering.parent_z_dofs.ravel()
    contact_scalar_dofs = ordering.scalar_dofs.ravel()
    full_unknowns = full_points.shape[0]
    candidate_indices = potential_contact_indices(
        np.min(full_gaps, axis=0), args.warning_distance
    )
    initial_candidate_count = candidate_indices.size
    print(
        f"union potential contact zone={candidate_indices.size}/{full_unknowns} "
        f"unknowns ({candidate_indices.size / full_unknowns:.3%}), "
        f"states={len(motion.states)}"
    )

    h_stats = None
    query_stats = None
    operator_build_seconds = 0.0
    operator_build_cpu_seconds = 0.0
    factorization_count = 1
    schur_operator = None
    if args.compliance_strategy == "mumps_schur":
        start_cpu = process_time()
        ksp.destroy()
        gc.collect()
        schur_operator = MumpsSchurComplianceOperator(
            A,
            contact_parent_dofs[candidate_indices],
            factor_type=args.schur_factor_type,
            max_memory_gib=args.schur_max_memory_gib,
        )
        factorization_seconds += schur_operator.factorization_seconds
        factorization_cpu_seconds += process_time() - start_cpu
        factorization_count += 1
        ksp = schur_operator
        print(
            "MUMPS selected-Schur factor: "
            f"{candidate_indices.size} unknowns, estimated dense/factor storage="
            f"{schur_operator.estimated_memory_bytes / 2**30:.3f} GiB"
        )

    def build_operator(indices):
        nonlocal h_stats, query_stats, operator_build_seconds
        nonlocal operator_build_cpu_seconds, compliance_sampling_seconds
        nonlocal compliance_sampling_cpu_seconds
        start = perf_counter()
        start_cpu = process_time()
        if _uses_hmatrix(args.compliance_strategy):
            points = full_points[indices]
            if points.shape[0] > 1 and args.h_leaf_size >= points.shape[0]:
                raise ValueError(
                    "h-leaf-size must be smaller than the union potential zone"
                )
            restricted_source = IndexedEntrySource(compliance_source, indices)
            compliance_source.reset_stats()
            operator = HMatrix.from_entry_source(
                points,
                restricted_source,
                leaf_size=args.h_leaf_size,
                eta=args.h_eta,
                tol=args.h_tol,
                split=args.h_split,
                lr_approx="aca_partial",
                symmetric=True,
                max_rank=args.h_max_rank,
            )
            h_stats = operator.stats()
            query_stats = compliance_source.stats()
            print(
                f"motion normal H-matrix stored entries={h_stats['memory_entries']}/"
                f"{points.shape[0] ** 2}, low-rank blocks={h_stats['low_rank']}, "
                f"near-field blocks={h_stats['dense']}"
            )
            if "linear_solves" in query_stats:
                print(
                    "motion full-FE ACA source: "
                    f"linear solves={query_stats['linear_solves']}, "
                    f"cached columns={query_stats['cached_columns']}, "
                    f"solve time={query_stats['solve_seconds']:.3f}s"
                )
        elif args.compliance_strategy == "fe_matrix_free":
            operator = FactorizedComplianceOperator(
                A, ksp, contact_parent_dofs[indices]
            )
            print("motion FE matrix-free operator: 0 stored compliance entries")
        elif args.compliance_strategy == "fe_iterative":
            operator = IterativeComplianceOperator(
                A, ksp, contact_parent_dofs[indices]
            )
            diagnostics = probe_spd_operator(operator)
            tolerance = max(1.0e-8, 100.0 * args.fe_rtol)
            if (
                diagnostics["reciprocity_error"] > tolerance
                or diagnostics["minimum_probed_rayleigh"] <= 0.0
            ):
                raise RuntimeError(
                    "Iterative compliance action failed its SPD probe: "
                    f"{diagnostics}"
                )
            print(
                "motion iterative FE operator: 0 stored compliance entries, "
                f"reciprocity probe={diagnostics['reciprocity_error']:.3e}, "
                "minimum Rayleigh="
                f"{diagnostics['minimum_probed_rayleigh']:.3e}"
            )
        else:
            if not np.array_equal(indices, candidate_indices):
                raise RuntimeError(
                    "The selected-Schur contact set is fixed for the motion "
                    "history; increase warning_distance and restart"
                )
            operator = schur_operator
        build_seconds = perf_counter() - start
        build_cpu_seconds = process_time() - start_cpu
        if args.compliance_strategy == "hmatrix_full":
            sampling_wall = float(query_stats["solve_seconds"])
            sampling_cpu = float(query_stats["solve_cpu_seconds"])
            compliance_sampling_seconds += sampling_wall
            compliance_sampling_cpu_seconds += sampling_cpu
            build_seconds = max(0.0, build_seconds - sampling_wall)
            build_cpu_seconds = max(0.0, build_cpu_seconds - sampling_cpu)
        operator_build_seconds += build_seconds
        operator_build_cpu_seconds += build_cpu_seconds
        return operator

    contact_operator = build_operator(candidate_indices)
    if args.compliance_strategy == "hmatrix":
        np.savez(
            output_dir / "compliance.npz",
            points=full_points,
            gap=np.min(full_gaps, axis=0),
            candidate_indices=candidate_indices,
            warning_distance=args.warning_distance,
            potential_contact_unknowns=candidate_indices.size,
            global_contact_unknowns=full_unknowns,
            inflation_displacement=inflation_values,
            inflation_pressure=args.inflation_pressure,
            floor_kind=args.floor_kind,
            floor_level=args.floor_level,
            floor_grid_size=args.floor_grid_size,
            archive_format_version=8,
            local_symmetry_tag=(
                -1 if args.local_symmetry_tag is None else args.local_symmetry_tag
            ),
            symmetry_mode=(
                "global_dihedral"
                if args.local_symmetry_tag is None
                else "local_open_patch"
            ),
            boundary_condition_id=BOUNDARY_CONDITION_ID,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            axial_divisions=args.axial_divisions,
            circumferential_divisions=args.circumferential_divisions,
            circumferential_layout=args.circumferential_layout,
            total_circumferential_divisions=(
                ordering.n_sectors if ordering.structured else 0
            ),
            coarsening_factor=args.coarsening_factor,
            h_leaf_size=args.h_leaf_size,
            h_eta=args.h_eta,
            h_tolerance=args.h_tol,
            h_max_rank=args.h_max_rank,
            h_stored_entries=h_stats["memory_entries"],
            h_low_rank_blocks=h_stats["low_rank"],
            h_dense_blocks=h_stats["dense"],
            source_queried_entries=query_stats["queried_entries"],
            reference_vertical_shift=reference_vertical_shift,
            **_compliance_sample_archive_fields(samples, output_dir),
        )

    contact_areas = surface_lumped_nodal_areas(
        Wz, facet_tags, contact_surface_tags, contact_scalar_dofs
    )
    spectral_preconditioner = None
    preconditioner_name = "none"
    if args.contact_solver == "ppcg" and args.pcg_preconditioner == "spectral":
        if ordering.structured and args.local_symmetry_tag is None:
            spectral_preconditioner = SectorSurfaceSpectralPreconditioner(
                ordering.points, zero_mode_factor=args.pcg_zero_mode_factor
            )
            preconditioner_name = "restricted_sector_spectral"
        else:
            spectral_preconditioner = SurfaceAreaDiagonalPreconditioner(
                contact_areas
            )
            preconditioner_name = "restricted_surface_area_diagonal"
    associated_area_field = (
        _contact_scalar_field(
            Wz, contact_scalar_dofs, contact_areas, "contact_associated_area"
        )
        if args.write_vtk
        else None
    )
    previous_force = np.zeros(full_unknowns, dtype=float)
    step_dir = output_dir / "motion_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    vtk_path = output_dir / f"tyre_dihedral_contact_{args.compliance_strategy}.pvd"
    history = {
        "iterations": [],
        "potential_rounds": [],
        "potential_contact_unknowns": [],
        "force_resultant": [],
        "primal_violation": [],
        "dual_violation": [],
        "complementarity": [],
        "minimum_clearance": [],
    }
    contact_solve_seconds = 0.0
    contact_solve_cpu_seconds = 0.0
    verification_seconds = 0.0
    verification_cpu_seconds = 0.0
    final_solve_seconds = 0.0
    final_solve_cpu_seconds = 0.0
    pressure_seconds = 0.0
    pressure_cpu_seconds = 0.0
    total_iterations = 0
    rhs = pressure_rhs.copy()
    displacement = A.createVecLeft()
    u_result = fem.Function(V)
    u_result.name = "displacement"
    contact_displacement = fem.Function(V)
    contact_displacement.name = "contact_displacement"
    pressure_force = fem.Function(Wz)
    pressure_force.name = "contact_pressure_force_based"
    if args.stress_recovery == "equilibrated":
        stress_projector = EquilibratedContactStressProjector(
            contact_displacement,
            A,
            Wz,
            facet_tags,
            contact_surface_tags,
            contact_scalar_dofs,
            contact_parent_dofs,
            projection=args.stress_projection,
            nodal_areas=contact_areas,
            name="contact_pressure_stress",
        )
    else:
        stress_projector = ContactStressProjector(
            contact_displacement,
            Wz,
            facet_tags,
            contact_surface_tags,
            contact_scalar_dofs,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            projection=args.stress_projection,
            recovery=args.stress_recovery,
            nodal_areas=contact_areas,
            name="contact_pressure_stress",
        )
    if args.write_vtk:
        nodal_force = fem.Function(Wz)
        nodal_force.name = "nodal_contact_force"
        potential_zone = fem.Function(Wz)
        potential_zone.name = "potential_contact_zone"
        initial_gap_field = fem.Function(Wz)
        initial_gap_field.name = "initial_gap"
        floor_height_field = fem.Function(Wz)
        floor_height_field.name = "floor_height_projection"

    with ExitStack() as stack:
        vtk = (
            stack.enter_context(VTKFile(comm, str(vtk_path), "w"))
            if args.write_vtk
            else None
        )
        if vtk is not None:
            vtk.write_mesh(mesh, motion.states[0].time)
        for step, state in enumerate(motion.states):
            full_gap = full_gaps[step]
            geometric_gap = full_gap - inflation_values[contact_parent_dofs]
            floor_height = full_points[:, 2] - geometric_gap
            full_force = previous_force.copy()
            full_clearance = full_gap.copy()
            print(
                f"load step {step + 1}/{len(motion.states)}: t={state.time:g}, "
                f"indentation={state.indentation:g}, Ry={state.rotation_y_deg:g} deg, "
                f"Rz={state.rotation_z_deg:g} deg, "
                f"translation=({state.translation_x:g}, {state.translation_y:g})"
            )
            solve_rounds = 0
            while True:
                solve_rounds += 1
                options: dict[str, object] = {
                    "tol": args.tol,
                    "max_iter": args.max_iter,
                    "record_history": True,
                    "z0": full_force[candidate_indices],
                }
                if args.contact_solver == "ppcg":
                    options["beta_method"] = args.pcg_beta_method
                    if spectral_preconditioner is not None:
                        options["preconditioner"] = RestrictedProjectedPreconditioner(
                            spectral_preconditioner, candidate_indices
                        )
                start = perf_counter()
                start_cpu = process_time()
                result = solve(
                    LCP(contact_operator, full_gap[candidate_indices]),
                    method=args.contact_solver,
                    **options,
                )
                contact_solve_seconds += perf_counter() - start
                contact_solve_cpu_seconds += process_time() - start_cpu
                total_iterations += result.iterations
                print(result.message)
                if not result.converged:
                    raise RuntimeError(
                        f"Contact solve failed at motion step {step}: "
                        f"{result.status.value}"
                    )
                full_force.fill(0.0)
                full_force[candidate_indices] = result.z
                excluded = np.ones(full_unknowns, dtype=bool)
                excluded[candidate_indices] = False
                if not np.any(excluded):
                    full_clearance = result.w.copy()
                    break

                start = perf_counter()
                start_cpu = process_time()
                if _uses_hmatrix(args.compliance_strategy):
                    full_clearance = restricted_source_clearance(
                        compliance_source,
                        full_gap,
                        candidate_indices,
                        result.z,
                    )
                else:
                    response = contact_operator.apply(
                        result.z, response_dofs=contact_parent_dofs
                    )
                    full_clearance = full_gap + response
                verification_seconds += perf_counter() - start
                verification_cpu_seconds += process_time() - start_cpu
                violations = excluded & (
                    full_clearance < -args.warning_verification_tol
                )
                print(
                    (
                        "tagged-patch verification: "
                        if args.local_symmetry_tag is not None
                        else "full-surface verification: "
                    )
                    +
                    f"minimum excluded clearance={full_clearance[excluded].min():.3e}, "
                    f"violations={np.count_nonzero(violations)}"
                )
                if not np.any(violations):
                    break
                if solve_rounds >= args.warning_max_rounds:
                    raise RuntimeError(
                        f"Potential zone failed verification at motion step {step}"
                    )
                if args.compliance_strategy == "mumps_schur":
                    raise RuntimeError(
                        "Contact escaped the fixed MUMPS Schur union. Increase "
                        "warning_distance and restart so one factor can be reused "
                        "for the complete motion history."
                    )
                if ordering.structured and args.local_symmetry_tag is None:
                    violation_mask = violations.reshape(
                        ordering.n_sectors, ordering.n_axial
                    )
                    addition = dilate_sector_axial_mask(
                        violation_mask, halo=args.warning_halo
                    ).reshape(-1)
                else:
                    addition = _dilate_unstructured_point_mask(
                        violations,
                        full_points,
                        halo=args.warning_halo,
                    )
                candidate_mask = np.zeros(full_unknowns, dtype=bool)
                candidate_mask[candidate_indices] = True
                candidate_mask |= addition
                candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
                contact_operator = build_operator(candidate_indices)
                print(
                    "expanded union potential zone to "
                    f"{candidate_indices.size} unknowns"
                )

            previous_force = full_force.copy()
            primal = max(0.0, -float(full_force.min()))
            dual = max(0.0, -float(full_clearance.min()))
            complementarity = float(
                np.linalg.norm(full_force * full_clearance, ord=np.inf)
            )
            print(
                f"global primal={primal:.3e}, dual={dual:.3e}, "
                f"complementarity={complementarity:.3e}"
            )
            if _discard_memmap_pages(samples) and step == 0:
                print(
                    "released file-backed compliance pages before FE "
                    "recovery/postprocessing"
                )

            rhs.set(0.0)
            rhs.axpy(1.0, pressure_rhs)
            rhs.setValues(
                contact_parent_dofs[candidate_indices],
                full_force[candidate_indices],
                addv=PETSc.InsertMode.ADD_VALUES,
            )
            rhs.assemble()
            start = perf_counter()
            start_cpu = process_time()
            ksp.solve(rhs, displacement)
            step_final_seconds = perf_counter() - start
            step_final_cpu_seconds = process_time() - start_cpu
            final_solve_seconds += step_final_seconds
            final_solve_cpu_seconds += step_final_cpu_seconds
            displacement_values = displacement.getArray(readonly=True)
            u_result.x.array[:] = displacement_values
            u_result.x.scatter_forward()
            contact_displacement.x.array[:] = displacement_values - inflation_values
            contact_displacement.x.scatter_forward()
            if args.local_symmetry_tag is not None:
                outside_patch = ~np.isin(
                    verification_parent_z, contact_parent_dofs
                )
                outside_clearance = (
                    verification_points[outside_patch, 2]
                    - moving_floors[step].height_at(
                        verification_points[outside_patch, :2]
                    )
                    + displacement_values[verification_parent_z[outside_patch]]
                )
                minimum_outside = float(outside_clearance.min(initial=np.inf))
                print(
                    "exact FE verification outside local symmetry tag: "
                    f"minimum clearance={minimum_outside:.3e}"
                )
                if minimum_outside < -args.warning_verification_tol:
                    raise RuntimeError(
                        "Contact escaped local symmetry tag "
                        f"{args.local_symmetry_tag} at motion step {step + 1}; "
                        "enlarge the regular patch or use FE matrix-free contact"
                    )

            start = perf_counter()
            start_cpu = process_time()
            _set_contact_scalar_field(
                pressure_force,
                contact_scalar_dofs,
                full_force / contact_areas,
            )
            pressure_stress = stress_projector.project()
            pressure_force_values = pressure_force.x.array[contact_scalar_dofs]
            pressure_stress_values = pressure_stress.x.array[contact_scalar_dofs]
            if not np.isclose(
                float(pressure_force_values @ contact_areas),
                float(full_force.sum()),
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "force-based contact pressure does not preserve force"
                )
            step_pressure_seconds = perf_counter() - start
            step_pressure_cpu_seconds = process_time() - start_cpu
            pressure_seconds += step_pressure_seconds
            pressure_cpu_seconds += step_pressure_cpu_seconds
            operator_stats = (
                {}
                if _uses_hmatrix(args.compliance_strategy)
                else contact_operator.stats()
            )

            if vtk is not None:
                _set_contact_scalar_field(
                    nodal_force, contact_scalar_dofs, full_force
                )
                potential_zone.x.array[:] = 0.0
                potential_zone.x.array[
                    contact_scalar_dofs[candidate_indices]
                ] = 1.0
                potential_zone.x.scatter_forward()
                _set_contact_scalar_field(
                    initial_gap_field, contact_scalar_dofs, geometric_gap
                )
                _set_contact_scalar_field(
                    floor_height_field, contact_scalar_dofs, floor_height
                )
                vtk.write_function(
                    [
                        u_result,
                        contact_displacement,
                        nodal_force,
                        potential_zone,
                        initial_gap_field,
                        floor_height_field,
                        associated_area_field,
                        pressure_stress,
                        pressure_force,
                    ],
                    state.time,
                )

            payload = {
                "time": state.time,
                "step": step,
                "force": full_force,
                "gap": full_gap,
                "clearance": full_clearance,
                "initial_gap": geometric_gap,
                "floor_height": floor_height,
                "contact_associated_area": contact_areas,
                "contact_pressure_stress": pressure_stress_values,
                "contact_pressure_force_based": pressure_force_values,
                "candidate_indices": candidate_indices,
                "initial_potential_contact_unknowns": initial_candidate_count,
                "potential_contact_unknowns": candidate_indices.size,
                "global_contact_unknowns": full_unknowns,
                "fe_unknowns": A.getSize()[0],
                "volume_cells": mesh.topology.index_map(
                    mesh.topology.dim
                ).size_global,
                "road_surface_unknowns": verification_points.shape[0],
                "local_symmetry_tag": (
                    -1
                    if args.local_symmetry_tag is None
                    else args.local_symmetry_tag
                ),
                "warning_distance": args.warning_distance,
                "warning_verification_tol": args.warning_verification_tol,
                "axial_divisions": args.axial_divisions,
                "circumferential_divisions": args.circumferential_divisions,
                "circumferential_layout": args.circumferential_layout,
                "total_circumferential_divisions": (
                    ordering.n_sectors if ordering.structured else 0
                ),
                "coarsening_factor": args.coarsening_factor,
                "scale": args.scale,
                "indentation": state.indentation,
                "floor_rotation_y_deg": state.rotation_y_deg,
                "floor_rotation_z_deg": state.rotation_z_deg,
                "floor_translation_x": state.translation_x,
                "floor_translation_y": state.translation_y,
                "floor_pivot": floor_pivot,
                "floor_kind": args.floor_kind,
                "floor_level": args.floor_level,
                "floor_grid_size": args.floor_grid_size,
                "young_modulus": args.young_modulus,
                "poisson_ratio": args.poisson_ratio,
                "inflation_pressure": args.inflation_pressure,
                "status": result.status.value,
                "iterations": result.iterations,
                "residual": result.residual,
                "potential_rounds": solve_rounds,
                "primal_violation": primal,
                "dual_violation": dual,
                "complementarity": complementarity,
                "compliance_strategy": args.compliance_strategy,
                "contact_solver": args.contact_solver,
                "preconditioner": preconditioner_name,
                "factorization_seconds": factorization_seconds,
                "factorization_cpu_seconds": factorization_cpu_seconds,
                "final_solve_seconds": step_final_seconds,
                "final_solve_cpu_seconds": step_final_cpu_seconds,
                "pressure_postprocess_seconds": step_pressure_seconds,
                "pressure_postprocess_cpu_seconds": step_pressure_cpu_seconds,
                "factorization_count": factorization_count,
                "peak_rss_bytes": _peak_rss_bytes(),
                "fe_operator_applications": int(
                    operator_stats.get("operator_applications", 0)
                ),
                "fe_linear_solves": int(
                    operator_stats.get("linear_solves", 0)
                ),
                "fe_linear_iterations": int(
                    operator_stats.get("linear_iterations", 0)
                ),
                "fe_maximum_linear_iterations": int(
                    operator_stats.get("maximum_linear_iterations", 0)
                ),
                "fe_linear_solve_seconds": float(
                    operator_stats.get("solve_seconds", 0.0)
                ),
                "schur_solves": int(operator_stats.get("schur_solves", 0)),
                "schur_full_solves": int(
                    operator_stats.get("full_solves", 0)
                ),
                "schur_estimated_memory_bytes": int(
                    operator_stats.get("estimated_memory_bytes", 0)
                ),
                "reference_vertical_shift": reference_vertical_shift,
                "input_file": str(args.input_file or ""),
                "stress_projection": args.stress_projection,
                "stress_recovery": args.stress_recovery,
            }
            if args.save_volume_fields:
                payload.update(
                    {
                        "displacement": u_result.x.array,
                        "contact_displacement": contact_displacement.x.array,
                        "inflation_displacement": inflation_values,
                    }
                )
            np.savez(step_dir / f"contact_result_{step:05d}.npz", **payload)
            if step == len(motion.states) - 1:
                np.savez(output_dir / "contact_result.npz", **payload)
                np.savez(
                    output_dir
                    / f"contact_result_{args.compliance_strategy}.npz",
                    **payload,
                )
            history["iterations"].append(result.iterations)
            history["potential_rounds"].append(solve_rounds)
            history["potential_contact_unknowns"].append(candidate_indices.size)
            history["force_resultant"].append(float(full_force.sum()))
            history["primal_violation"].append(primal)
            history["dual_violation"].append(dual)
            history["complementarity"].append(complementarity)
            history["minimum_clearance"].append(float(full_clearance.min()))

    total_seconds = (
        operator_build_seconds
        + contact_solve_seconds
        + verification_seconds
        + final_solve_seconds
        + pressure_seconds
    )
    total_cpu_seconds = (
        operator_build_cpu_seconds
        + contact_solve_cpu_seconds
        + verification_cpu_seconds
        + final_solve_cpu_seconds
        + pressure_cpu_seconds
    )
    strategy_total_seconds = perf_counter() - strategy_start
    strategy_total_cpu_seconds = process_time() - strategy_cpu_start
    history_payload = motion.as_arrays()
    history_payload.update({key: np.asarray(value) for key, value in history.items()})
    history_payload.update(
        {
            "factorization_count": factorization_count,
            "peak_rss_bytes": _peak_rss_bytes(),
            "factorization_seconds": factorization_seconds,
            "factorization_cpu_seconds": factorization_cpu_seconds,
            "floor_build_seconds": floor_build_seconds,
            "floor_build_cpu_seconds": floor_build_cpu_seconds,
            "inflation_solve_seconds": inflation_solve_seconds,
            "inflation_solve_cpu_seconds": inflation_solve_cpu_seconds,
            "compliance_load_seconds": compliance_load_seconds,
            "compliance_load_cpu_seconds": compliance_load_cpu_seconds,
            "compliance_sampling_seconds": compliance_sampling_seconds,
            "compliance_sampling_cpu_seconds": compliance_sampling_cpu_seconds,
            "total_contact_iterations": total_iterations,
            "load_states": len(motion.states),
            "fe_unknowns": A.getSize()[0],
            "volume_cells": mesh.topology.index_map(
                mesh.topology.dim
            ).size_global,
            "motion_solve_seconds": total_seconds,
            "motion_solve_cpu_seconds": total_cpu_seconds,
            "strategy_total_seconds": strategy_total_seconds,
            "strategy_total_cpu_seconds": strategy_total_cpu_seconds,
            "operator_build_seconds": operator_build_seconds,
            "operator_build_cpu_seconds": operator_build_cpu_seconds,
            "contact_solve_seconds": contact_solve_seconds,
            "contact_solve_cpu_seconds": contact_solve_cpu_seconds,
            "verification_seconds": verification_seconds,
            "verification_cpu_seconds": verification_cpu_seconds,
            "final_solve_seconds": final_solve_seconds,
            "final_solve_cpu_seconds": final_solve_cpu_seconds,
            "pressure_postprocess_seconds": pressure_seconds,
            "pressure_postprocess_cpu_seconds": pressure_cpu_seconds,
            "compliance_stored_entries": (
                0 if h_stats is None else h_stats["memory_entries"]
            ),
            "compliance_sampling_solves": (
                sampling_solves
                if query_stats is None
                else int(query_stats.get("linear_solves", sampling_solves))
            ),
            "compliance_strategy": args.compliance_strategy,
            "contact_solver": args.contact_solver,
            "floor_pivot": floor_pivot,
            "input_file": str(args.input_file or ""),
        }
    )
    np.savez(output_dir / "motion_history.npz", **history_payload)
    print(
        f"motion solve reused one {args.compliance_strategy} operator for "
        f"{len(motion.states)} states; "
        f"wrote results under {output_dir}"
    )


def main() -> None:
    args = _parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("This example is currently serial; run without mpiexec.")
    if not (0.0 <= args.poisson_ratio < 0.5):
        raise ValueError("poisson-ratio must lie in [0, 0.5)")
    if args.indentation < 0.0:
        raise ValueError("indentation must be non-negative")
    static_motion = FloorMotionState(
        time=0.0,
        indentation=args.indentation,
        rotation_y_deg=args.rotate_floor,
        rotation_z_deg=args.torsion_floor,
        translation_x=args.floor_translation_x,
        translation_y=args.floor_translation_y,
    )
    if args.motion_file is not None:
        motion = FloorMotionSchedule.from_json(
            args.motion_file, defaults=static_motion
        )
    elif args.embedded_motion is not None:
        motion = FloorMotionSchedule.from_mapping(
            args.embedded_motion, defaults=static_motion
        )
    else:
        motion = FloorMotionSchedule.constant(
            indentation=args.indentation,
            rotation_y_deg=args.rotate_floor,
            rotation_z_deg=args.torsion_floor,
            translation_x=args.floor_translation_x,
            translation_y=args.floor_translation_y,
        )
    if not np.isfinite(args.floor_level):
        raise ValueError("floor-level must be finite")
    if args.floor_grid_size < 2:
        raise ValueError("floor-grid-size must be at least 2")
    if args.floor_margin < 0.0:
        raise ValueError("floor-margin must be non-negative")
    if args.floor_kind == "rough":
        if args.roughness_rms <= 0.0:
            raise ValueError("roughness-rms must be positive")
        if not 0.0 <= args.roughness_hurst <= 1.0:
            raise ValueError("roughness-hurst must lie in [0, 1]")
        if not 0.0 < args.roughness_k_low < args.roughness_k_high <= 0.5:
            raise ValueError(
                "roughness cutoffs must satisfy 0 < k-low < k-high <= 0.5"
            )
    if args.inflation_pressure < 0.0:
        raise ValueError("inflation-pressure must be non-negative")
    if args.h_leaf_size <= 0 or args.h_max_rank <= 0:
        raise ValueError("H-matrix leaf size and maximum rank must be positive")
    if args.h_eta <= 0.0 or args.h_tol <= 0.0:
        raise ValueError("H-matrix eta and tolerance must be positive")
    if args.local_symmetry_validation_columns <= 0:
        raise ValueError("local-symmetry-validation-columns must be positive")
    if args.local_symmetry_tolerance <= 0.0:
        raise ValueError("local-symmetry-tolerance must be positive")
    if args.pcg_zero_mode_factor <= 0.0:
        raise ValueError("pcg-zero-mode-factor must be positive")
    if args.axial_divisions < 6 or args.axial_divisions % 2:
        raise ValueError("axial-divisions must be an even integer >= 6")
    angular_layout = build_circumferential_layout(
        args.circumferential_divisions,
        kind=args.circumferential_layout,
        coarsening_factor=args.coarsening_factor,
    )
    if (
        args.circumferential_layout == "graded"
        and args.compliance_strategy == "hmatrix"
        and args.local_symmetry_tag is None
    ):
        raise ValueError(
            "The sampled dihedral H-matrix requires equal sectors around the "
            "full tyre unless a regular patch is selected. Set "
            "hmatrix.local_symmetry_tag=204, use "
            "an FE operator strategy, or use "
            "mesh.circumferential_layout='uniform'."
        )
    if (
        args.local_symmetry_tag is not None
        and (
            args.circumferential_layout != "graded"
            or args.compliance_strategy != "hmatrix"
        )
    ):
        raise ValueError(
            "local-symmetry-tag is only valid for the H-matrix strategy on a "
            "graded mesh"
        )
    if args.scale <= 0.0:
        raise ValueError("scale must be positive")
    if np.isnan(args.warning_distance) or args.warning_distance < 0.0:
        raise ValueError("warning-distance must be non-negative")
    if args.warning_halo < 0:
        raise ValueError("warning-halo must be non-negative")
    if args.warning_max_rounds <= 0:
        raise ValueError("warning-max-rounds must be positive")
    if args.warning_verification_tol <= 0.0:
        raise ValueError("warning-verification-tol must be positive")
    if args.compliance_strategy != "hmatrix" and args.load_compliance:
        raise ValueError(
            "--load-compliance is only valid with --compliance-strategy hmatrix"
        )
    if not _uses_hmatrix(args.compliance_strategy) and args.sampling_only:
        raise ValueError(
            "--sampling-only is only valid with --compliance-strategy hmatrix"
        )
    if args.fe_rtol <= 0.0 or not np.isfinite(args.fe_rtol):
        raise ValueError("fe-rtol must be finite and positive")
    if args.fe_atol < 0.0 or not np.isfinite(args.fe_atol):
        raise ValueError("fe-atol must be finite and non-negative")
    if args.fe_max_iter <= 0:
        raise ValueError("fe-max-iter must be positive")
    if not args.fe_options_prefix.endswith("_"):
        raise ValueError("fe-options-prefix must end in '_'")
    if (
        args.schur_max_memory_gib <= 0.0
        or not np.isfinite(args.schur_max_memory_gib)
    ):
        raise ValueError("schur-max-memory-gib must be finite and positive")

    mesh_path = args.mesh.resolve()
    default_mesh_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "tyre_dihedral"
        / "tyre_dihedral.msh"
    ).resolve()
    generate_mesh = args.regenerate or not mesh_path.exists()
    while True:
        if generate_mesh:
            generate_tyre_mesh(
                args.template,
                mesh_path,
                axial_divisions=args.axial_divisions,
                circumferential_divisions=args.circumferential_divisions,
                circumferential_layout=args.circumferential_layout,
                coarsening_factor=args.coarsening_factor,
                scale=args.scale,
            )

        mesh, _, facet_tags = gmshio.read_from_msh(
            mesh_path, comm, rank=0, gdim=3
        )
        mesh_manifest = load_mesh_manifest(mesh_path)
        regular_contact_facets = facet_tags.find(CONTACT_TAG)
        symmetric_patch_facets = facet_tags.find(SYMMETRIC_CONTACT_TAG)
        local_symmetry_facets = (
            np.empty(0, dtype=np.int32)
            if args.local_symmetry_tag is None
            else facet_tags.find(args.local_symmetry_tag)
        )
        contact_facets = np.unique(
            np.concatenate(
                [
                    regular_contact_facets,
                    symmetric_patch_facets,
                    local_symmetry_facets,
                ]
            )
        ).astype(np.int32)
        fixed_facets = facet_tags.find(FIXED_TAG)
        if contact_facets.size == 0 or fixed_facets.size == 0:
            raise RuntimeError(
                "Generated mesh is missing contact or disk-edge facet tags"
            )

        contact_vertices = _surface_vertices(mesh, contact_facets)
        if args.circumferential_layout == "graded":
            actual_sectors = None
            actual_axial_divisions = None
            density_matches = bool(
                mesh_manifest is not None
                and mesh_manifest.get("topology") == "tetrahedron"
                and mesh_manifest.get("circumferential_layout") == "graded"
                and mesh_manifest.get("axial_divisions")
                == args.axial_divisions
                and mesh_manifest.get("fine_circumferential_divisions")
                == args.circumferential_divisions
                and np.isclose(
                    float(mesh_manifest.get("coarsening_factor", np.nan)),
                    args.coarsening_factor,
                )
                and np.isclose(
                    float(mesh_manifest.get("scale", np.nan)), args.scale
                )
                and (
                    args.local_symmetry_tag is None
                    or local_symmetry_facets.size > 0
                )
            )
            boundary_matches = fixed_facets.size > 0
        else:
            actual_angles, actual_axial_nodes = infer_meridian_shape(
                mesh.geometry.x[contact_vertices]
            )
            actual_sectors = actual_angles.size
            actual_axial_divisions = actual_axial_nodes - 1
            expected_angles = angular_layout.angles
            angle_layout_matches = (
                actual_angles.shape == expected_angles.shape
                and np.allclose(
                    actual_angles,
                    expected_angles,
                    rtol=0.0,
                    atol=2.0e-8,
                )
            )
            density_matches = (
                angle_layout_matches
                and actual_axial_divisions == args.axial_divisions
            )
            boundary_matches = fixed_facets.size == 2 * actual_sectors
        if density_matches and boundary_matches:
            break
        mismatch_parts = []
        if not density_matches:
            if args.circumferential_layout == "graded":
                mismatch_parts.append(
                    "existing mesh has no matching graded-tetra manifest; "
                    f"requested axial={args.axial_divisions}, fine divisions="
                    f"{args.circumferential_divisions}, coarsening factor="
                    f"{args.coarsening_factor:g}"
                )
            else:
                mismatch_parts.append(
                    f"existing mesh density is axial={actual_axial_divisions}, "
                    f"circumferential meridians={actual_sectors}; requested axial="
                    f"{args.axial_divisions}, layout={args.circumferential_layout}, "
                    f"fine divisions={args.circumferential_divisions}, expected "
                    f"meridians={angular_layout.total_divisions}"
                )
        if not boundary_matches:
            if actual_sectors is None:
                mismatch_parts.append("disk-edge tag is empty")
            else:
                mismatch_parts.append(
                    f"disk-edge tag contains {fixed_facets.size} facets; expected "
                    f"{2 * actual_sectors} for {BOUNDARY_CONDITION_ID}"
                )
        mismatch = "; ".join(mismatch_parts)
        if generate_mesh:
            raise RuntimeError(
                f"Generated tyre mesh has an unexpected density: {mismatch}"
            )
        if mesh_path != default_mesh_path:
            raise ValueError(
                f"{mismatch}. Pass --regenerate to replace the custom mesh."
            )
        print(f"{mismatch}; regenerating the default tyre mesh")
        generate_mesh = True

    contact_surface_tags = tuple(
        tag
        for tag in dict.fromkeys(
            [CONTACT_TAG, SYMMETRIC_CONTACT_TAG, args.local_symmetry_tag]
        )
        if tag is not None and facet_tags.find(int(tag)).size > 0
    )
    fixed_vertices, fixed_x_levels, fixed_radius = _validate_disk_edge_facets(
        mesh, fixed_facets, actual_sectors
    )
    fixed_location_label = (
        "axial |x| span="
        if actual_sectors is None
        else "axial rings="
    )
    print(
        f"disk-edge constraint={fixed_facets.size} facets, "
        f"{fixed_vertices.size} vertices, {fixed_location_label}"
        f"{np.array2string(fixed_x_levels, precision=6)}, "
        f"radius={fixed_radius:.6g}"
    )

    # Align the tyre once at zero indentation. Every load step then moves only
    # the rigid floor, so the assembled stiffness and its factorization remain
    # unchanged throughout the history.
    reference_vertical_shift = (
        args.floor_level - float(mesh.geometry.x[contact_vertices, 2].min())
    )
    mesh.geometry.x[:, 2] += reference_vertical_shift

    V, A, pressure_rhs = _assemble_elasticity(
        mesh,
        fixed_facets,
        facet_tags,
        args.young_modulus,
        args.poisson_ratio,
        args.inflation_pressure,
    )
    (
        all_Wz,
        all_scalar_dofs,
        all_parent_y,
        all_parent_z,
        all_contact_points,
    ) = _contact_component_data(
        V, mesh, contact_facets
    )
    if args.local_symmetry_tag is not None:
        Wz, scalar_dofs, parent_y, parent_z, contact_points = (
            _contact_component_data(V, mesh, local_symmetry_facets)
        )
        half_angle = 0.5 * np.deg2rad(FINE_SECTOR_ANGLE_DEG)
        centre_angle = np.deg2rad(ROAD_FACING_ANGLE_DEG)
        local_angles = np.linspace(
            centre_angle - half_angle,
            centre_angle + half_angle,
            args.circumferential_divisions + 1,
        )
        ordering = order_contact_meridians(
            contact_points,
            scalar_dofs,
            parent_y,
            parent_z,
            local_angles,
            axis_yz=(0.0, reference_vertical_shift),
            geometry_tol=None,
        )
        print(
            f"local symmetry contact tag={args.local_symmetry_tag}: "
            f"{ordering.n_sectors} meridians x {ordering.n_axial} axial nodes"
        )
    elif args.circumferential_layout == "uniform":
        Wz = all_Wz
        ordering = order_contact_sectors(
            all_contact_points,
            all_scalar_dofs,
            all_parent_y,
            all_parent_z,
            angular_layout.total_divisions,
            axis_yz=(0.0, reference_vertical_shift),
        )
    else:
        Wz = all_Wz
        ordering = order_unstructured_contact(
            all_contact_points,
            all_scalar_dofs,
            all_parent_y,
            all_parent_z,
            axis_yz=(0.0, reference_vertical_shift),
        )

    full_points = ordering.points.reshape(-1, 3)
    verification_points = all_contact_points
    output_dir = (
        mesh_path.parent
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    start_cpu = process_time()
    floor_center = 0.5 * (
        verification_points[:, :2].min(axis=0)
        + verification_points[:, :2].max(axis=0)
    )
    floor_pivot = np.array(
        [floor_center[0], floor_center[1], args.floor_level], dtype=float
    )
    maximum_radius = float(
        np.linalg.norm(verification_points[:, :2] - floor_center, axis=1).max()
    )
    maximum_translation = max(
        np.hypot(state.translation_x, state.translation_y)
        for state in motion.states
    )
    minimum_cosine = min(
        np.cos(np.deg2rad(abs(state.rotation_y_deg)))
        for state in motion.states
    )
    half_width = (
        (maximum_radius + maximum_translation) / minimum_cosine
        + args.floor_margin
        + 10.0 * (args.roughness_rms if args.floor_kind == "rough" else 0.0)
    )
    x_bounds = (floor_center[0] - half_width, floor_center[0] + half_width)
    y_bounds = (floor_center[1] - half_width, floor_center[1] + half_width)
    if args.floor_kind == "flat":
        floor = RegularFloor.flat(
            x_bounds,
            y_bounds,
            cells=args.floor_grid_size,
            level=args.floor_level,
        )
    else:
        floor = RegularFloor.self_affine(
            x_bounds,
            y_bounds,
            cells=args.floor_grid_size,
            level=args.floor_level,
            rms=args.roughness_rms,
            hurst=args.roughness_hurst,
            k_low=args.roughness_k_low,
            k_high=args.roughness_k_high,
            plateau=args.roughness_plateau,
            noise=args.roughness_noise,
            seed=args.roughness_seed,
        )
    moving_floors = [
        MovingRegularFloor(floor, state, floor_pivot) for state in motion.states
    ]
    floor_heights = np.empty((len(moving_floors), full_points.shape[0]))
    for step, moving_floor in enumerate(moving_floors):
        floor_heights[step] = moving_floor.height_at(full_points[:, :2])
    floor_files = []
    for step, moving_floor in enumerate(moving_floors):
        floor_path = output_dir / "floor_motion" / f"floor_{step:05d}.vtu"
        floor_files.append(
            (moving_floor.write_vtu(floor_path), moving_floor.state.time)
        )
    write_pvd_collection(output_dir / "floor_motion.pvd", floor_files)
    if len(moving_floors) == 1:
        moving_floors[0].write_vtu(output_dir / f"floor_{args.floor_kind}.vtu")
    np.savez(
        output_dir / "floor.npz",
        x=floor.x,
        y=floor.y,
        height=floor.height,
        floor_kind=args.floor_kind,
        floor_level=args.floor_level,
        floor_grid_size=args.floor_grid_size,
        floor_margin=args.floor_margin,
        circumferential_layout=args.circumferential_layout,
        fine_circumferential_divisions=args.circumferential_divisions,
        total_circumferential_divisions=(
            angular_layout.total_divisions
            if args.circumferential_layout == "uniform"
            else 0
        ),
        coarsening_factor=args.coarsening_factor,
        roughness_rms=args.roughness_rms,
        roughness_hurst=args.roughness_hurst,
        roughness_k_low=args.roughness_k_low,
        roughness_k_high=args.roughness_k_high,
        roughness_seed=args.roughness_seed,
        roughness_plateau=args.roughness_plateau,
        roughness_noise=args.roughness_noise,
        floor_pivot=floor_pivot,
        input_file=str(args.input_file or ""),
        **motion.as_arrays(),
    )
    floor_build_seconds = perf_counter() - start
    floor_build_cpu_seconds = process_time() - start_cpu
    print(
        f"{args.floor_kind} floor={args.floor_grid_size} x "
        f"{args.floor_grid_size} regular cells, bounds="
        f"x[{floor.x[0]:.6g}, {floor.x[-1]:.6g}], "
        f"y[{floor.y[0]:.6g}, {floor.y[-1]:.6g}], "
        f"height=[{floor.height.min():.3e}, {floor.height.max():.3e}], "
        f"motion states={len(motion.states)}"
    )
    strategy_start = perf_counter()
    strategy_cpu_start = process_time()
    samples = None
    compliance_source = None
    compliance_load_seconds = 0.0
    compliance_load_cpu_seconds = 0.0
    compliance_sampling_seconds = 0.0
    compliance_sampling_cpu_seconds = 0.0
    sampling_solves = 0
    if args.compliance_strategy == "hmatrix":
        if args.load_compliance is not None:
            start = perf_counter()
            start_cpu = process_time()
            samples = load_dihedral_compliance_archive(
                args.load_compliance,
                full_points,
                n_axial=ordering.n_axial,
                n_sectors=ordering.n_sectors,
                circumferential_divisions=args.circumferential_divisions,
                local_symmetry_tag=args.local_symmetry_tag,
                young_modulus=args.young_modulus,
                poisson_ratio=args.poisson_ratio,
                boundary_condition_id=BOUNDARY_CONDITION_ID,
            )
            if args.mmap_compliance:
                samples = memory_map_compliance_samples(
                    samples, output_dir / "compliance_samples.npy"
                )
                gc.collect()
            compliance_load_seconds = perf_counter() - start
            compliance_load_cpu_seconds = process_time() - start_cpu
            print(
                "loaded reference compliance data from "
                f"{args.load_compliance.expanduser().resolve()}"
            )
        else:
            sampling_solves = 2 * ordering.n_axial

    start = perf_counter()
    start_cpu = process_time()
    if args.compliance_strategy == "fe_iterative":
        ksp = create_iterative_solver(
            A,
            comm,
            coordinates=V.tabulate_dof_coordinates(),
            ksp_type=args.fe_ksp_type,
            pc_type=args.fe_pc_type,
            relative_tolerance=args.fe_rtol,
            absolute_tolerance=args.fe_atol,
            max_iterations=args.fe_max_iter,
            options_prefix=args.fe_options_prefix,
        )
    else:
        ksp = create_lu_solver(
            A, comm, factor_solver_type=args.factor_solver_type
        )
    factorization_seconds = perf_counter() - start
    factorization_cpu_seconds = process_time() - start_cpu
    factorization_count = 1
    inflation_displacement = A.createVecLeft()
    start = perf_counter()
    start_cpu = process_time()
    ksp.solve(pressure_rhs, inflation_displacement)
    inflation_solve_seconds = perf_counter() - start
    inflation_solve_cpu_seconds = process_time() - start_cpu
    if hasattr(ksp, "getConvergedReason") and int(ksp.getConvergedReason()) <= 0:
        raise RuntimeError(
            "Inflation FE solve failed with PETSc reason "
            f"{ksp.getConvergedReason()}"
        )

    if args.compliance_strategy == "hmatrix":
        if samples is None:
            start = perf_counter()
            start_cpu = process_time()
            samples = sample_reference_normal_compliance(
                A,
                ksp,
                ordering,
                show_progress=not args.no_progress,
            )
            compliance_sampling_seconds = perf_counter() - start
            compliance_sampling_cpu_seconds = process_time() - start_cpu
        if args.local_symmetry_tag is None:
            compliance_source = DihedralComplianceEntrySource(samples)
        else:
            sector_step = float(np.diff(ordering.sector_angles).mean())
            compliance_source = LocalDihedralComplianceEntrySource(
                samples, sector_step=sector_step
            )
        reflection_error = (
            dihedral_reflection_error(samples)
            if args.local_symmetry_tag is None
            else None
        )
        reciprocity_error = compliance_source.reciprocity_error(
            sample_size=min(4096, compliance_source.shape[0] ** 2)
        )
        print(
            f"contact unknowns={compliance_source.shape[0]}, "
            f"reference axial nodes={ordering.n_axial}, "
            f"normal H-matrices=1, PETSc LU auxiliary y/z solves={sampling_solves}"
        )
        print(f"maximum sector angle error={ordering.sector_angle_error:.3e} rad")
        if reflection_error is not None:
            print(f"dihedral reflection error={reflection_error:.3e}")
        else:
            validation = validate_local_dihedral_compliance(
                A,
                ksp,
                ordering,
                compliance_source,
                sample_columns=args.local_symmetry_validation_columns,
            )
            print(
                "local symmetry validation against direct FE: "
                f"columns={validation['sample_columns']}, relative estimate="
                f"{validation['relative_frobenius_estimate']:.3e}, maximum "
                f"column error={validation['maximum_relative_column_error']:.3e}"
            )
            if (
                validation["maximum_relative_column_error"]
                > args.local_symmetry_tolerance
            ):
                message = (
                    "Local rotational-symmetry compliance error exceeds "
                    f"{args.local_symmetry_tolerance:.3e}; the non-symmetric "
                    "remainder of the FE structure materially affects this patch"
                )
                if args.local_symmetry_strict:
                    raise RuntimeError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=1)
        print(f"sampled compliance reciprocity error={reciprocity_error:.3e}")
        if reciprocity_error > 1.0e-6:
            raise RuntimeError(
                "Compliance reciprocity error is too large; check DOF ordering "
                "and mesh symmetry"
            )
        compliance_source.reset_stats()
    elif args.compliance_strategy == "hmatrix_full":
        compliance_source = FactorizedComplianceEntrySource(
            A,
            ksp,
            ordering.parent_z_dofs.ravel(),
        )
        print(
            f"contact unknowns={compliance_source.shape[0]}, normal H-matrices=1, "
            "geometric symmetry=none; ACA entries use cached exact FE solves"
        )
        if ordering.structured:
            print(
                f"maximum sector angle error="
                f"{ordering.sector_angle_error:.3e} rad"
            )
    else:
        if ordering.structured and args.local_symmetry_tag is None:
            ordering_detail = f"reference axial nodes={ordering.n_axial}"
        else:
            ordering_detail = "unstructured triangular contact surface"
        print(
            f"contact unknowns={full_points.shape[0]}, {ordering_detail}, "
            "compliance storage=none"
        )
        if ordering.structured and args.local_symmetry_tag is None:
            print(
                f"maximum sector angle error="
                f"{ordering.sector_angle_error:.3e} rad"
            )
        if args.compliance_strategy == "fe_iterative":
            print(
                "iterative flexibility strategy: every compliance action "
                f"uses {args.fe_ksp_type}+{args.fe_pc_type} with rtol="
                f"{args.fe_rtol:.1e}"
            )
        elif args.compliance_strategy == "mumps_schur":
            print(
                "selected-Schur strategy: the inflation-adjusted potential "
                "zone will be condensed and factorized with MUMPS"
            )
        else:
            print(
                "flexibility-matrix-free strategy: every compliance action "
                "reuses the factorized FE stiffness"
            )

    if args.compliance_strategy == "hmatrix" and args.mmap_compliance:
        old_source = compliance_source
        samples = memory_map_compliance_samples(
            samples, output_dir / "compliance_samples.npy"
        )
        if args.local_symmetry_tag is None:
            compliance_source = DihedralComplianceEntrySource(samples)
        else:
            compliance_source = LocalDihedralComplianceEntrySource(
                samples,
                sector_step=float(np.diff(ordering.sector_angles).mean()),
            )
        compliance_source.reset_stats()
        del old_source
        gc.collect()
        print(
            "compliance samples memory-mapped from "
            f"{Path(samples.filename).resolve()} "
            f"({samples.nbytes / 2**30:.3f} GiB virtual data)"
        )

    inflation_values = np.array(
        inflation_displacement.getArray(readonly=True), copy=True
    )
    if args.local_symmetry_tag is not None:
        local_parent_dofs = ordering.parent_z_dofs.ravel()
        outside_patch = ~np.isin(all_parent_z, local_parent_dofs)
        minimum_outside_gap = np.inf
        limiting_step = -1
        for step, moving_floor in enumerate(moving_floors):
            outside_gap = (
                all_contact_points[outside_patch, 2]
                - moving_floor.height_at(all_contact_points[outside_patch, :2])
                + inflation_values[all_parent_z[outside_patch]]
            )
            step_minimum = float(outside_gap.min(initial=np.inf))
            if step_minimum < minimum_outside_gap:
                minimum_outside_gap = step_minimum
                limiting_step = step
        print(
            "local symmetry patch geometric certificate: minimum free gap "
            f"outside tag={minimum_outside_gap:.3e} at motion step "
            f"{limiting_step + 1}"
        )
        if minimum_outside_gap <= args.warning_distance:
            raise RuntimeError(
                "The potential contact zone reaches outside local symmetry tag "
                f"{args.local_symmetry_tag}: minimum outside gap="
                f"{minimum_outside_gap:.3e}, warning distance="
                f"{args.warning_distance:.3e}. Enlarge the tagged regular patch, "
                "reduce the motion/indentation or warning distance, or use the "
                "FE matrix-free strategy."
            )
    if len(motion.states) > 1 and not args.sampling_only:
        motion_full_gaps = floor_heights
        np.subtract(
            full_points[None, :, 2],
            motion_full_gaps,
            out=motion_full_gaps,
        )
        motion_full_gaps += inflation_values[
            ordering.parent_z_dofs.ravel()
        ][None, :]
        del floor_heights
        _run_motion_history(
            args=args,
            comm=comm,
            mesh=mesh,
            facet_tags=facet_tags,
            contact_surface_tags=contact_surface_tags,
            V=V,
            A=A,
            ksp=ksp,
            pressure_rhs=pressure_rhs,
            Wz=Wz,
            ordering=ordering,
            full_points=full_points,
            verification_points=verification_points,
            verification_parent_z=all_parent_z,
            moving_floors=moving_floors,
            output_dir=output_dir,
            motion=motion,
            floor_pivot=floor_pivot,
            full_gaps=motion_full_gaps,
            inflation_values=inflation_values,
            compliance_source=compliance_source,
            samples=samples,
            floor_build_seconds=floor_build_seconds,
            floor_build_cpu_seconds=floor_build_cpu_seconds,
            factorization_seconds=factorization_seconds,
            factorization_cpu_seconds=factorization_cpu_seconds,
            inflation_solve_seconds=inflation_solve_seconds,
            inflation_solve_cpu_seconds=inflation_solve_cpu_seconds,
            compliance_load_seconds=compliance_load_seconds,
            compliance_load_cpu_seconds=compliance_load_cpu_seconds,
            compliance_sampling_seconds=compliance_sampling_seconds,
            compliance_sampling_cpu_seconds=compliance_sampling_cpu_seconds,
            sampling_solves=sampling_solves,
            strategy_start=strategy_start,
            strategy_cpu_start=strategy_cpu_start,
            reference_vertical_shift=reference_vertical_shift,
        )
        return

    floor_height = floor_heights[0]
    geometric_gap = full_points[:, 2] - floor_height
    full_gap = (
        geometric_gap
        + inflation_values[ordering.parent_z_dofs.ravel()]
    )
    candidate_indices = potential_contact_indices(
        full_gap, args.warning_distance
    )
    full_unknowns = full_points.shape[0]
    print(
        f"potential contact zone={candidate_indices.size}/{full_unknowns} "
        f"unknowns ({candidate_indices.size / full_unknowns:.3%}), "
        f"warning distance={args.warning_distance:g}"
    )
    schur_operator = None
    schur_candidate_indices = candidate_indices.copy()
    if args.compliance_strategy == "mumps_schur":
        start_cpu = process_time()
        ksp.destroy()
        gc.collect()
        schur_operator = MumpsSchurComplianceOperator(
            A,
            ordering.parent_z_dofs.ravel()[candidate_indices],
            factor_type=args.schur_factor_type,
            max_memory_gib=args.schur_max_memory_gib,
        )
        factorization_seconds += schur_operator.factorization_seconds
        factorization_cpu_seconds += process_time() - start_cpu
        factorization_count += 1
        ksp = schur_operator
        print(
            "MUMPS selected-Schur factor: "
            f"{candidate_indices.size} unknowns, estimated dense/factor storage="
            f"{schur_operator.estimated_memory_bytes / 2**30:.3f} GiB"
        )

    def build_potential_hmatrix(indices):
        points = full_points[indices]
        if points.shape[0] > 1 and args.h_leaf_size >= points.shape[0]:
            raise ValueError(
                "h-leaf-size must be smaller than the number of potential "
                "contact unknowns to avoid a single global dense block"
            )
        restricted_source = IndexedEntrySource(compliance_source, indices)
        compliance_source.reset_stats()
        matrix = HMatrix.from_entry_source(
            points,
            restricted_source,
            leaf_size=args.h_leaf_size,
            eta=args.h_eta,
            tol=args.h_tol,
            split=args.h_split,
            lr_approx="aca_partial",
            symmetric=True,
            max_rank=args.h_max_rank,
        )
        stats = matrix.stats()
        queries = compliance_source.stats()
        dense_entries = points.shape[0] ** 2
        print(
            "H-matrix blocks="
            f"{stats['stored_blocks']} ({stats['low_rank']} low-rank, "
            f"{stats['dense']} near-field), stored entries="
            f"{stats['memory_entries']}/{dense_entries} "
            f"({stats['memory_entries'] / dense_entries:.3%} of potential dense), "
            f"{stats['memory_entries'] / full_unknowns ** 2:.3%} of global dense"
        )
        print(
            f"entry-source calls={queries['query_calls']}, "
            f"queried scalar entries={queries['queried_entries']}, "
            f"largest query={queries['largest_query']}"
        )
        if "linear_solves" in queries:
            print(
                f"full-FE ACA source solves={queries['linear_solves']}, "
                f"cached columns={queries['cached_columns']}, "
                f"solve time={queries['solve_seconds']:.3f}s"
            )
        return matrix, stats, queries

    preconditioner_contact_areas = surface_lumped_nodal_areas(
        Wz,
        facet_tags,
        contact_surface_tags,
        ordering.scalar_dofs.ravel(),
    )
    full_spectral_preconditioner = None
    preconditioner_name = "none"
    if args.contact_solver == "ppcg" and args.pcg_preconditioner == "spectral":
        if ordering.structured and args.local_symmetry_tag is None:
            full_spectral_preconditioner = SectorSurfaceSpectralPreconditioner(
                ordering.points,
                zero_mode_factor=args.pcg_zero_mode_factor,
            )
            preconditioner_name = (
                "restricted_sector_spectral"
                if candidate_indices.size < full_unknowns
                else "sector_spectral"
            )
        else:
            full_spectral_preconditioner = SurfaceAreaDiagonalPreconditioner(
                preconditioner_contact_areas
            )
            preconditioner_name = (
                "restricted_surface_area_diagonal"
                if candidate_indices.size < full_unknowns
                else "surface_area_diagonal"
            )
        preconditioner_label = preconditioner_name.replace("_", " ")
        print(
            f"PPCG preconditioner={preconditioner_label} "
            f"(zero-mode factor={args.pcg_zero_mode_factor:g})"
        )

    result = None
    full_force = np.zeros(full_unknowns, dtype=float)
    full_clearance = full_gap.copy()
    initial_candidate_count = candidate_indices.size
    solve_rounds = 0
    total_iterations = 0
    operator_build_seconds = 0.0
    operator_build_cpu_seconds = 0.0
    contact_solve_seconds = 0.0
    contact_solve_cpu_seconds = 0.0
    verification_seconds = 0.0
    verification_cpu_seconds = 0.0
    h_stats = None
    query_stats = None
    fe_totals = {
        "operator_applications": 0,
        "linear_solves": 0,
        "cache_hits": 0,
        "zero_bypasses": 0,
        "solve_seconds": 0.0,
        "linear_iterations": 0,
        "maximum_linear_iterations": 0,
        "schur_solves": 0,
        "full_solves": 0,
    }
    z0 = None
    while True:
        solve_rounds += 1
        print(
            f"potential-zone round {solve_rounds}: "
            f"preparing {candidate_indices.size} x {candidate_indices.size} operator"
        )
        start = perf_counter()
        start_cpu = process_time()
        if _uses_hmatrix(args.compliance_strategy):
            contact_operator, h_stats, query_stats = build_potential_hmatrix(
                candidate_indices
            )
        elif args.compliance_strategy == "fe_matrix_free":
            candidate_dofs = ordering.parent_z_dofs.ravel()[candidate_indices]
            contact_operator = FactorizedComplianceOperator(
                A, ksp, candidate_dofs
            )
            print(
                "FE matrix-free compliance operator: 0 stored compliance entries"
            )
        elif args.compliance_strategy == "fe_iterative":
            candidate_dofs = ordering.parent_z_dofs.ravel()[candidate_indices]
            contact_operator = IterativeComplianceOperator(
                A, ksp, candidate_dofs
            )
            diagnostics = probe_spd_operator(contact_operator)
            tolerance = max(1.0e-8, 100.0 * args.fe_rtol)
            if (
                diagnostics["reciprocity_error"] > tolerance
                or diagnostics["minimum_probed_rayleigh"] <= 0.0
            ):
                raise RuntimeError(
                    "Iterative compliance action failed its SPD probe: "
                    f"{diagnostics}"
                )
            print(
                "iterative FE compliance operator: 0 stored compliance entries, "
                f"reciprocity probe={diagnostics['reciprocity_error']:.3e}, "
                "minimum Rayleigh="
                f"{diagnostics['minimum_probed_rayleigh']:.3e}"
            )
        else:
            if not np.array_equal(candidate_indices, schur_candidate_indices):
                raise RuntimeError(
                    "The MUMPS selected-Schur contact set is fixed; increase "
                    "warning_distance and restart"
                )
            contact_operator = schur_operator
        build_seconds = perf_counter() - start
        build_cpu_seconds = process_time() - start_cpu
        if args.compliance_strategy == "hmatrix_full":
            sampling_wall = float(query_stats["solve_seconds"])
            sampling_cpu = float(query_stats["solve_cpu_seconds"])
            compliance_sampling_seconds += sampling_wall
            compliance_sampling_cpu_seconds += sampling_cpu
            build_seconds = max(0.0, build_seconds - sampling_wall)
            build_cpu_seconds = max(0.0, build_cpu_seconds - sampling_cpu)
        operator_build_seconds += build_seconds
        operator_build_cpu_seconds += build_cpu_seconds
        if args.sampling_only:
            break

        options: dict[str, object] = {
            "tol": args.tol,
            "max_iter": args.max_iter,
            "record_history": True,
        }
        if z0 is not None:
            options["z0"] = z0
        if args.contact_solver == "ppcg":
            options["beta_method"] = args.pcg_beta_method
            if full_spectral_preconditioner is not None:
                options["preconditioner"] = RestrictedProjectedPreconditioner(
                    full_spectral_preconditioner, candidate_indices
                )

        start = perf_counter()
        start_cpu = process_time()
        result = solve(
            LCP(contact_operator, full_gap[candidate_indices]),
            method=args.contact_solver,
            **options,
        )
        contact_solve_seconds += perf_counter() - start
        contact_solve_cpu_seconds += process_time() - start_cpu
        print(result.message)
        total_iterations += result.iterations
        if not result.converged:
            raise RuntimeError(
                f"Contact solve did not converge: {result.status.value}"
            )

        full_force.fill(0.0)
        full_force[candidate_indices] = result.z
        excluded = np.ones(full_unknowns, dtype=bool)
        excluded[candidate_indices] = False
        if not np.any(excluded):
            full_clearance = result.w.copy()
            if not _uses_hmatrix(args.compliance_strategy):
                for name, value in contact_operator.stats().items():
                    if name in fe_totals:
                        if name == "maximum_linear_iterations":
                            fe_totals[name] = max(fe_totals[name], value)
                        else:
                            fe_totals[name] += value
            break

        start = perf_counter()
        start_cpu = process_time()
        if _uses_hmatrix(args.compliance_strategy):
            compliance_source.reset_stats()
            full_clearance = restricted_source_clearance(
                compliance_source,
                full_gap,
                candidate_indices,
                result.z,
            )
            verification_stats = compliance_source.stats()
            verification_detail = (
                f"queried entries={verification_stats['queried_entries']}"
            )
        else:
            contact_displacement = contact_operator.apply(
                result.z,
                response_dofs=ordering.parent_z_dofs.ravel(),
            )
            full_clearance = full_gap + contact_displacement
            round_fe_stats = contact_operator.stats()
            verification_detail = (
                f"FE linear solves={round_fe_stats['linear_solves']}, "
                f"cache hits={round_fe_stats['cache_hits']}"
            )
            for name, value in round_fe_stats.items():
                if name in fe_totals:
                    if name == "maximum_linear_iterations":
                        fe_totals[name] = max(fe_totals[name], value)
                    else:
                        fe_totals[name] += value
        verification_seconds += perf_counter() - start
        verification_cpu_seconds += process_time() - start_cpu
        violations = excluded & (
            full_clearance < -args.warning_verification_tol
        )
        print(
            (
                "tagged-patch verification: "
                if args.local_symmetry_tag is not None
                else "full-surface verification: "
            )
            +
            f"minimum excluded clearance={full_clearance[excluded].min():.3e}, "
            f"violations={np.count_nonzero(violations)}, "
            f"{verification_detail}"
        )
        if not np.any(violations):
            break
        if solve_rounds >= args.warning_max_rounds:
            raise RuntimeError(
                "Potential contact zone could not be certified after "
                f"{solve_rounds} rounds; increase --warning-distance or "
                "--warning-max-rounds"
            )
        if args.compliance_strategy == "mumps_schur":
            raise RuntimeError(
                "Contact escaped the fixed MUMPS Schur set. Increase "
                "warning_distance and restart."
            )

        if ordering.structured and args.local_symmetry_tag is None:
            violation_mask = violations.reshape(
                ordering.n_sectors, ordering.n_axial
            )
            addition = dilate_sector_axial_mask(
                violation_mask, halo=args.warning_halo
            ).reshape(-1)
        else:
            addition = _dilate_unstructured_point_mask(
                violations,
                full_points,
                halo=args.warning_halo,
            )
        candidate_mask = np.zeros(full_unknowns, dtype=bool)
        candidate_mask[candidate_indices] = True
        candidate_mask |= addition
        previous_force = full_force.copy()
        candidate_indices = np.flatnonzero(candidate_mask).astype(np.int64)
        z0 = previous_force[candidate_indices]
        print(
            f"expanded potential contact zone to {candidate_indices.size} "
            "unknowns around verification violations"
        )

    if args.compliance_strategy == "hmatrix":
        np.savez(
            output_dir / "compliance.npz",
            points=full_points,
            gap=full_gap,
            candidate_indices=candidate_indices,
            warning_distance=args.warning_distance,
            potential_contact_unknowns=candidate_indices.size,
            global_contact_unknowns=full_unknowns,
            potential_rounds=0 if args.sampling_only else solve_rounds,
            inflation_displacement=inflation_values,
            inflation_pressure=args.inflation_pressure,
            floor_kind=args.floor_kind,
            floor_level=args.floor_level,
            floor_grid_size=args.floor_grid_size,
            archive_format_version=8,
            local_symmetry_tag=(
                -1 if args.local_symmetry_tag is None else args.local_symmetry_tag
            ),
            symmetry_mode=(
                "global_dihedral"
                if args.local_symmetry_tag is None
                else "local_open_patch"
            ),
            boundary_condition_id=BOUNDARY_CONDITION_ID,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            axial_divisions=args.axial_divisions,
            circumferential_divisions=args.circumferential_divisions,
            circumferential_layout=args.circumferential_layout,
            total_circumferential_divisions=(
                ordering.n_sectors if ordering.structured else 0
            ),
            coarsening_factor=args.coarsening_factor,
            h_leaf_size=args.h_leaf_size,
            h_eta=args.h_eta,
            h_tolerance=args.h_tol,
            h_max_rank=args.h_max_rank,
            h_stored_entries=h_stats["memory_entries"],
            h_low_rank_blocks=h_stats["low_rank"],
            h_dense_blocks=h_stats["dense"],
            source_queried_entries=query_stats["queried_entries"],
            reference_vertical_shift=reference_vertical_shift,
            **_compliance_sample_archive_fields(samples, output_dir),
        )
    if args.sampling_only:
        print(f"saved reference compliance data to {output_dir / 'compliance.npz'}")
        return

    strategy_total_seconds = perf_counter() - strategy_start
    strategy_total_cpu_seconds = process_time() - strategy_cpu_start
    primal_violation = max(0.0, -float(full_force.min()))
    dual_violation = max(0.0, -float(full_clearance.min()))
    complementarity = float(
        np.linalg.norm(full_force * full_clearance, ord=np.inf)
    )
    print(
        f"global primal={primal_violation:.3e}, dual={dual_violation:.3e}, "
        f"complementarity={complementarity:.3e}"
    )
    print(
        f"strategy timings: floor={floor_build_seconds:.3f}s, "
        f"factorization={factorization_seconds:.3f}s, "
        f"inflation solve={inflation_solve_seconds:.3f}s, "
        f"compliance load={compliance_load_seconds:.3f}s, "
        f"compliance sampling={compliance_sampling_seconds:.3f}s, "
        f"operator build={operator_build_seconds:.3f}s, "
        f"contact solve={contact_solve_seconds:.3f}s, "
        f"verification={verification_seconds:.3f}s, "
        f"total={strategy_total_seconds:.3f}s"
    )
    if not _uses_hmatrix(args.compliance_strategy):
        print(
            f"{args.compliance_strategy} statistics: "
            f"operator applications={int(fe_totals['operator_applications'])}, "
            f"linear solves={int(fe_totals['linear_solves'])}, "
            f"cache hits={int(fe_totals['cache_hits'])}, "
            f"zero bypasses={int(fe_totals['zero_bypasses'])}, "
            f"linear-solve time={fe_totals['solve_seconds']:.3f}s"
        )
        if args.compliance_strategy == "fe_iterative":
            print(
                "inner KSP iterations: total="
                f"{int(fe_totals['linear_iterations'])}, maximum/application="
                f"{int(fe_totals['maximum_linear_iterations'])}"
            )
        if args.compliance_strategy == "mumps_schur":
            print(
                f"selected-Schur solves={int(fe_totals['schur_solves'])}, "
                f"full solves={int(fe_totals['full_solves'])}"
            )

    rhs = pressure_rhs.copy()
    displacement = A.createVecLeft()
    rhs.setValues(
        ordering.parent_z_dofs.ravel()[candidate_indices],
        full_force[candidate_indices],
        addv=PETSc.InsertMode.ADD_VALUES,
    )
    rhs.assemble()
    start = perf_counter()
    start_cpu = process_time()
    ksp.solve(rhs, displacement)
    final_solve_seconds = perf_counter() - start
    final_solve_cpu_seconds = process_time() - start_cpu

    u_result = fem.Function(V)
    u_result.name = "displacement"
    displacement_values = displacement.getArray(readonly=True)
    if args.local_symmetry_tag is not None:
        outside_patch = ~np.isin(
            all_parent_z, ordering.parent_z_dofs.ravel()
        )
        outside_clearance = (
            all_contact_points[outside_patch, 2]
            - moving_floors[0].height_at(all_contact_points[outside_patch, :2])
            + displacement_values[all_parent_z[outside_patch]]
        )
        minimum_outside = float(outside_clearance.min(initial=np.inf))
        print(
            "exact FE verification outside local symmetry tag: "
            f"minimum clearance={minimum_outside:.3e}"
        )
        if minimum_outside < -args.warning_verification_tol:
            raise RuntimeError(
                "Contact escaped local symmetry tag "
                f"{args.local_symmetry_tag}; enlarge the regular patch or use "
                "FE matrix-free contact"
            )
    if len(u_result.x.array) != len(displacement_values):
        raise RuntimeError("Unexpected serial PETSc/DOLFINx vector-size mismatch")
    u_result.x.array[:] = displacement_values
    u_result.x.scatter_forward()

    contact_displacement = fem.Function(V)
    contact_displacement.name = "contact_displacement"
    contact_displacement.x.array[:] = displacement_values - inflation_values
    contact_displacement.x.scatter_forward()

    start = perf_counter()
    start_cpu = process_time()
    contact_scalar_dofs = ordering.scalar_dofs.ravel()
    contact_areas = preconditioner_contact_areas
    contact_pressure_force_based = force_based_contact_pressure(
        Wz,
        contact_scalar_dofs,
        full_force,
        contact_areas,
        name="contact_pressure_force_based",
    )
    if args.stress_recovery == "equilibrated":
        equilibrated_projector = EquilibratedContactStressProjector(
            contact_displacement,
            A,
            Wz,
            facet_tags,
            contact_surface_tags,
            contact_scalar_dofs,
            ordering.parent_z_dofs.ravel(),
            projection=args.stress_projection,
            nodal_areas=contact_areas,
            name="contact_pressure_stress",
        )
        contact_pressure_stress = equilibrated_projector.project()
    else:
        contact_pressure_stress = project_compressive_normal_stress(
            contact_displacement,
            Wz,
            facet_tags,
            contact_surface_tags,
            contact_scalar_dofs,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            projection=args.stress_projection,
            recovery=args.stress_recovery,
            nodal_areas=contact_areas,
            name="contact_pressure_stress",
        )
    pressure_force_values = np.array(
        contact_pressure_force_based.x.array[contact_scalar_dofs], copy=True
    )
    pressure_stress_values = np.array(
        contact_pressure_stress.x.array[contact_scalar_dofs], copy=True
    )
    force_resultant_from_pressure = float(
        pressure_force_values @ contact_areas
    )
    if not np.isclose(
        force_resultant_from_pressure,
        float(full_force.sum()),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("force-based contact pressure does not preserve force")
    pressure_postprocess_seconds = perf_counter() - start
    pressure_postprocess_cpu_seconds = process_time() - start_cpu
    equilibrated_detail = (
        "stress-derived vertical resultant="
        f"{equilibrated_projector.vertical_resultant:.6e}, "
        if args.stress_recovery == "equilibrated"
        else ""
    )
    print(
        "contact pressure: force/area resultant="
        f"{force_resultant_from_pressure:.6e}, stress-normal resultant="
        f"{float(pressure_stress_values @ contact_areas):.6e}, "
        f"{equilibrated_detail}"
        f"associated area={contact_areas.sum():.6e}, "
        f"recovery time={pressure_postprocess_seconds:.3f}s"
    )

    nodal_force = fem.Function(Wz)
    nodal_force.name = "nodal_contact_force"
    nodal_force.x.array[:] = 0.0
    nodal_force.x.array[ordering.scalar_dofs.ravel()] = full_force
    nodal_force.x.scatter_forward()

    potential_zone = fem.Function(Wz)
    potential_zone.name = "potential_contact_zone"
    potential_zone.x.array[:] = 0.0
    potential_zone.x.array[
        ordering.scalar_dofs.ravel()[candidate_indices]
    ] = 1.0
    potential_zone.x.scatter_forward()

    initial_gap_field = _contact_scalar_field(
        Wz, contact_scalar_dofs, geometric_gap, "initial_gap"
    )
    floor_height_field = _contact_scalar_field(
        Wz, contact_scalar_dofs, floor_height, "floor_height_projection"
    )
    associated_area_field = _contact_scalar_field(
        Wz, contact_scalar_dofs, contact_areas, "contact_associated_area"
    )

    vtk_path = output_dir / f"tyre_dihedral_contact_{args.compliance_strategy}.pvd"
    with VTKFile(comm, str(vtk_path), "w") as vtk:
        vtk.write_mesh(mesh)
        vtk.write_function(
            [
                u_result,
                contact_displacement,
                nodal_force,
                potential_zone,
                initial_gap_field,
                floor_height_field,
                associated_area_field,
                contact_pressure_stress,
                contact_pressure_force_based,
            ]
        )

    result_payload = {
        "contact_points": full_points,
        "force": full_force,
        "gap": full_gap,
        "clearance": full_clearance,
        "initial_gap": geometric_gap,
        "floor_height": floor_height,
        "contact_associated_area": contact_areas,
        "contact_pressure_stress": pressure_stress_values,
        "contact_pressure_force_based": pressure_force_values,
        "candidate_indices": candidate_indices,
        "initial_potential_contact_unknowns": initial_candidate_count,
        "potential_contact_unknowns": candidate_indices.size,
        "global_contact_unknowns": full_unknowns,
        "fe_unknowns": A.getSize()[0],
        "volume_cells": mesh.topology.index_map(
            mesh.topology.dim
        ).size_global,
        "load_states": 1,
        "road_surface_unknowns": verification_points.shape[0],
        "local_symmetry_tag": (
            -1 if args.local_symmetry_tag is None else args.local_symmetry_tag
        ),
        "warning_distance": args.warning_distance,
        "warning_verification_tol": args.warning_verification_tol,
        "potential_rounds": solve_rounds,
        "axial_divisions": args.axial_divisions,
        "circumferential_divisions": args.circumferential_divisions,
        "circumferential_layout": args.circumferential_layout,
        "total_circumferential_divisions": (
            ordering.n_sectors if ordering.structured else 0
        ),
        "coarsening_factor": args.coarsening_factor,
        "scale": args.scale,
        "indentation": args.indentation,
        "floor_rotation_y_deg": args.rotate_floor,
        "floor_rotation_z_deg": args.torsion_floor,
        "floor_translation_x": args.floor_translation_x,
        "floor_translation_y": args.floor_translation_y,
        "floor_pivot": floor_pivot,
        "floor_kind": args.floor_kind,
        "floor_level": args.floor_level,
        "floor_grid_size": args.floor_grid_size,
        "floor_margin": args.floor_margin,
        "roughness_rms": args.roughness_rms,
        "roughness_hurst": args.roughness_hurst,
        "roughness_k_low": args.roughness_k_low,
        "roughness_k_high": args.roughness_k_high,
        "roughness_seed": args.roughness_seed,
        "roughness_plateau": args.roughness_plateau,
        "roughness_noise": args.roughness_noise,
        "young_modulus": args.young_modulus,
        "poisson_ratio": args.poisson_ratio,
        "inflation_pressure": args.inflation_pressure,
        "displacement": displacement_values,
        "contact_displacement": displacement_values - inflation_values,
        "inflation_displacement": inflation_values,
        "residual": result.residual,
        "status": result.status.value,
        "iterations": result.iterations,
        "potential_total_iterations": total_iterations,
        "contact_solver": args.contact_solver,
        "preconditioner": preconditioner_name,
        "compliance_strategy": args.compliance_strategy,
        "boundary_condition_id": BOUNDARY_CONDITION_ID,
        "floor_build_seconds": floor_build_seconds,
        "floor_build_cpu_seconds": floor_build_cpu_seconds,
        "factorization_seconds": factorization_seconds,
        "factorization_cpu_seconds": factorization_cpu_seconds,
        "inflation_solve_seconds": inflation_solve_seconds,
        "inflation_solve_cpu_seconds": inflation_solve_cpu_seconds,
        "compliance_load_seconds": compliance_load_seconds,
        "compliance_load_cpu_seconds": compliance_load_cpu_seconds,
        "compliance_sampling_seconds": compliance_sampling_seconds,
        "compliance_sampling_cpu_seconds": compliance_sampling_cpu_seconds,
        "operator_build_seconds": operator_build_seconds,
        "operator_build_cpu_seconds": operator_build_cpu_seconds,
        "contact_solve_seconds": contact_solve_seconds,
        "contact_solve_cpu_seconds": contact_solve_cpu_seconds,
        "verification_seconds": verification_seconds,
        "verification_cpu_seconds": verification_cpu_seconds,
        "strategy_total_seconds": strategy_total_seconds,
        "strategy_total_cpu_seconds": strategy_total_cpu_seconds,
        "final_solve_seconds": final_solve_seconds,
        "final_solve_cpu_seconds": final_solve_cpu_seconds,
        "pressure_postprocess_seconds": pressure_postprocess_seconds,
        "pressure_postprocess_cpu_seconds": pressure_postprocess_cpu_seconds,
        "compliance_stored_entries": (
            h_stats["memory_entries"]
            if _uses_hmatrix(args.compliance_strategy)
            else 0
        ),
        "fe_operator_applications": int(fe_totals["operator_applications"]),
        "compliance_sampling_solves": (
            sampling_solves
            if query_stats is None
            else int(query_stats.get("linear_solves", sampling_solves))
        ),
        "fe_linear_solves": int(fe_totals["linear_solves"]),
        "fe_cache_hits": int(fe_totals["cache_hits"]),
        "fe_zero_bypasses": int(fe_totals["zero_bypasses"]),
        "fe_linear_solve_seconds": fe_totals["solve_seconds"],
        "fe_linear_iterations": int(fe_totals["linear_iterations"]),
        "fe_maximum_linear_iterations": int(
            fe_totals["maximum_linear_iterations"]
        ),
        "schur_solves": int(fe_totals["schur_solves"]),
        "schur_full_solves": int(fe_totals["full_solves"]),
        "schur_estimated_memory_bytes": (
            0
            if schur_operator is None
            else schur_operator.estimated_memory_bytes
        ),
        "factorization_count": factorization_count,
        "peak_rss_bytes": _peak_rss_bytes(),
        "reference_vertical_shift": reference_vertical_shift,
        "input_file": str(args.input_file or ""),
        "stress_projection": args.stress_projection,
        "stress_recovery": args.stress_recovery,
    }
    np.savez(output_dir / "contact_result.npz", **result_payload)
    strategy_result_path = (
        output_dir / f"contact_result_{args.compliance_strategy}.npz"
    )
    np.savez(strategy_result_path, **result_payload)
    print(f"wrote contact results under {output_dir}")


if __name__ == "__main__":
    main()

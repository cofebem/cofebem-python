"""Tyre-road contact with H-matrix or factorized-FE compliance actions.

The mesh is generated from ``geo_files/geometry_v2.geo`` as a structured full
tyre. The H-matrix strategy loads only the axial contact nodes of one reference
meridian in two transverse directions, then answers ACA queries using dihedral
symmetry. The flexibility-matrix-free strategy instead applies compliance by
back-solving the factorized FE stiffness for every PPCG request. Neither path
constructs a global dense compliance.

Example
-------
conda run -n fenicsx-env python examples/tyre_dihedral_contact.py \
    --axial-divisions 24 --circumferential-divisions 32 --regenerate
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

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
    FactorizedComplianceOperator,
    create_lu_solver,
    dilate_sector_axial_mask,
    dihedral_reflection_error,
    infer_regular_sector_shape,
    load_dihedral_compliance_archive,
    order_contact_sectors,
    potential_contact_indices,
    restricted_source_clearance,
    sample_reference_transverse_compliance,
)
from cofebem.hmatrices import HMatrix, IndexedEntrySource
from cofebem.lcp import (
    LCP,
    RestrictedProjectedPreconditioner,
    SectorSurfaceSpectralPreconditioner,
    solve,
)
from cofebem.mesh.tyre_dihedral_hex import (
    CONTACT_TAG,
    FIXED_TAG,
    INNER_SURFACE_TAG,
    generate_tyre_mesh,
)


def _surface_vertices(mesh, facets: np.ndarray) -> np.ndarray:
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    return np.unique(
        np.concatenate([facet_to_vertex.links(int(facet)) for facet in facets])
    )


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


def _assemble_elasticity(
    mesh,
    fixed_facets,
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
    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
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


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--axial-divisions", type=int, default=24)
    parser.add_argument("--circumferential-divisions", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0e-3)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--indentation", type=float, default=5.0e-3)
    parser.add_argument("--young-modulus", type=float, default=2.5e8)
    parser.add_argument("--poisson-ratio", type=float, default=0.48)
    parser.add_argument("--inflation-pressure", type=float, default=1.5e5)
    parser.add_argument(
        "--compliance-strategy",
        choices=("hmatrix", "fe_matrix_free"),
        default="hmatrix",
        help=(
            "use a sampled dihedral H-matrix or apply compliance through a "
            "factorized FE solve on every operator application"
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
        help="sector/axial halo added around verification violations",
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
    parser.add_argument(
        "--load-compliance",
        type=Path,
        metavar="PATH",
        help=(
            "load reference-meridian samples from a previous compliance.npz "
            "and skip the compliance sampling solves"
        ),
    )
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("This example is currently serial; run without mpiexec.")
    if not (0.0 <= args.poisson_ratio < 0.5):
        raise ValueError("poisson-ratio must lie in [0, 0.5)")
    if args.indentation < 0.0:
        raise ValueError("indentation must be non-negative")
    if args.inflation_pressure < 0.0:
        raise ValueError("inflation-pressure must be non-negative")
    if args.h_leaf_size <= 0 or args.h_max_rank <= 0:
        raise ValueError("H-matrix leaf size and maximum rank must be positive")
    if args.h_eta <= 0.0 or args.h_tol <= 0.0:
        raise ValueError("H-matrix eta and tolerance must be positive")
    if args.pcg_zero_mode_factor <= 0.0:
        raise ValueError("pcg-zero-mode-factor must be positive")
    if args.axial_divisions < 6 or args.axial_divisions % 2:
        raise ValueError("axial-divisions must be an even integer >= 6")
    if (
        args.circumferential_divisions < 4
        or args.circumferential_divisions % 2
    ):
        raise ValueError(
            "circumferential-divisions must be an even integer >= 4"
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
    if args.compliance_strategy == "fe_matrix_free" and args.load_compliance:
        raise ValueError(
            "--load-compliance is only valid with --compliance-strategy hmatrix"
        )
    if args.compliance_strategy == "fe_matrix_free" and args.sampling_only:
        raise ValueError(
            "--sampling-only is only valid with --compliance-strategy hmatrix"
        )

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
                scale=args.scale,
            )

        mesh, _, facet_tags = gmshio.read_from_msh(
            mesh_path, comm, rank=0, gdim=3
        )
        contact_facets = facet_tags.find(CONTACT_TAG)
        fixed_facets = facet_tags.find(FIXED_TAG)
        if contact_facets.size == 0 or fixed_facets.size == 0:
            raise RuntimeError("Generated mesh is missing contact or bead facet tags")

        contact_vertices = _surface_vertices(mesh, contact_facets)
        actual_sectors, actual_axial_nodes = infer_regular_sector_shape(
            mesh.geometry.x[contact_vertices]
        )
        actual_axial_divisions = actual_axial_nodes - 1
        density_matches = (
            actual_sectors == args.circumferential_divisions
            and actual_axial_divisions == args.axial_divisions
        )
        if density_matches:
            break
        mismatch = (
            f"existing mesh density is axial={actual_axial_divisions}, "
            f"circumferential={actual_sectors}; requested axial="
            f"{args.axial_divisions}, circumferential="
            f"{args.circumferential_divisions}"
        )
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

    # Put the undeformed outer tread at a prescribed penetration into z=0.
    vertical_shift = -float(mesh.geometry.x[contact_vertices, 2].min()) - args.indentation
    mesh.geometry.x[:, 2] += vertical_shift

    V, A, pressure_rhs = _assemble_elasticity(
        mesh,
        fixed_facets,
        facet_tags,
        args.young_modulus,
        args.poisson_ratio,
        args.inflation_pressure,
    )
    Wz, scalar_dofs, parent_y, parent_z, contact_points = _contact_component_data(
        V, mesh, contact_facets
    )
    ordering = order_contact_sectors(
        contact_points,
        scalar_dofs,
        parent_y,
        parent_z,
        args.circumferential_divisions,
        axis_yz=(0.0, vertical_shift),
    )

    full_points = ordering.points.reshape(-1, 3)
    strategy_start = perf_counter()
    samples = None
    compliance_source = None
    compliance_load_seconds = 0.0
    compliance_sampling_seconds = 0.0
    sampling_solves = 0
    if args.compliance_strategy == "hmatrix":
        if args.load_compliance is not None:
            start = perf_counter()
            samples = load_dihedral_compliance_archive(
                args.load_compliance,
                full_points,
                n_axial=ordering.n_axial,
                n_sectors=ordering.n_sectors,
                young_modulus=args.young_modulus,
                poisson_ratio=args.poisson_ratio,
            )
            compliance_load_seconds = perf_counter() - start
            print(
                "loaded reference compliance data from "
                f"{args.load_compliance.expanduser().resolve()}"
            )
        else:
            sampling_solves = 2 * ordering.n_axial

    start = perf_counter()
    ksp = create_lu_solver(
        A, comm, factor_solver_type=args.factor_solver_type
    )
    factorization_seconds = perf_counter() - start
    inflation_displacement = A.createVecLeft()
    start = perf_counter()
    ksp.solve(pressure_rhs, inflation_displacement)
    inflation_solve_seconds = perf_counter() - start

    if args.compliance_strategy == "hmatrix":
        if samples is None:
            start = perf_counter()
            samples = sample_reference_transverse_compliance(
                A,
                ksp,
                ordering,
                show_progress=not args.no_progress,
            )
            compliance_sampling_seconds = perf_counter() - start
        compliance_source = DihedralComplianceEntrySource(samples)
        reflection_error = dihedral_reflection_error(samples)
        reciprocity_error = compliance_source.reciprocity_error(
            sample_size=min(4096, compliance_source.shape[0] ** 2)
        )
        print(
            f"contact unknowns={compliance_source.shape[0]}, "
            f"reference axial nodes={ordering.n_axial}, "
            f"PETSc LU compliance solves={sampling_solves}"
        )
        print(f"maximum sector angle error={ordering.sector_angle_error:.3e} rad")
        print(f"dihedral reflection error={reflection_error:.3e}")
        print(f"sampled compliance reciprocity error={reciprocity_error:.3e}")
        if reciprocity_error > 1.0e-6:
            raise RuntimeError(
                "Compliance reciprocity error is too large; check DOF ordering "
                "and mesh symmetry"
            )
        compliance_source.reset_stats()
    else:
        print(
            f"contact unknowns={full_points.shape[0]}, "
            f"reference axial nodes={ordering.n_axial}, compliance storage=none"
        )
        print(f"maximum sector angle error={ordering.sector_angle_error:.3e} rad")
        print(
            "flexibility-matrix-free strategy: every compliance action reuses "
            "the factorized FE stiffness"
        )

    output_dir = mesh_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    inflation_values = inflation_displacement.getArray(readonly=True)
    full_gap = (
        full_points[:, 2]
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
        return matrix, stats, queries

    full_spectral_preconditioner = None
    preconditioner_name = "none"
    if args.contact_solver == "ppcg" and args.pcg_preconditioner == "spectral":
        full_spectral_preconditioner = SectorSurfaceSpectralPreconditioner(
            ordering.points,
            zero_mode_factor=args.pcg_zero_mode_factor,
        )
        preconditioner_name = (
            "restricted_sector_spectral"
            if candidate_indices.size < full_unknowns
            else "sector_spectral"
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
    contact_solve_seconds = 0.0
    verification_seconds = 0.0
    h_stats = None
    query_stats = None
    fe_totals = {
        "operator_applications": 0,
        "linear_solves": 0,
        "cache_hits": 0,
        "zero_bypasses": 0,
        "solve_seconds": 0.0,
    }
    z0 = None
    while True:
        solve_rounds += 1
        print(
            f"potential-zone round {solve_rounds}: "
            f"preparing {candidate_indices.size} x {candidate_indices.size} operator"
        )
        start = perf_counter()
        if args.compliance_strategy == "hmatrix":
            contact_operator, h_stats, query_stats = build_potential_hmatrix(
                candidate_indices
            )
        else:
            candidate_dofs = ordering.parent_z_dofs.ravel()[candidate_indices]
            contact_operator = FactorizedComplianceOperator(
                A, ksp, candidate_dofs
            )
            print(
                "FE matrix-free compliance operator: 0 stored compliance entries"
            )
        operator_build_seconds += perf_counter() - start
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
        result = solve(
            LCP(contact_operator, full_gap[candidate_indices]),
            method=args.contact_solver,
            **options,
        )
        contact_solve_seconds += perf_counter() - start
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
            if args.compliance_strategy == "fe_matrix_free":
                for name, value in contact_operator.stats().items():
                    fe_totals[name] += value
            break

        start = perf_counter()
        if args.compliance_strategy == "hmatrix":
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
                fe_totals[name] += value
        verification_seconds += perf_counter() - start
        violations = excluded & (
            full_clearance < -args.warning_verification_tol
        )
        print(
            "full-surface verification: "
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

        violation_mask = violations.reshape(
            ordering.n_sectors, ordering.n_axial
        )
        addition = dilate_sector_axial_mask(
            violation_mask, halo=args.warning_halo
        ).reshape(-1)
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
            samples=samples,
            points=full_points,
            gap=full_gap,
            candidate_indices=candidate_indices,
            warning_distance=args.warning_distance,
            potential_contact_unknowns=candidate_indices.size,
            global_contact_unknowns=full_unknowns,
            potential_rounds=0 if args.sampling_only else solve_rounds,
            inflation_displacement=inflation_values,
            inflation_pressure=args.inflation_pressure,
            archive_format_version=3,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            axial_divisions=args.axial_divisions,
            circumferential_divisions=args.circumferential_divisions,
            h_leaf_size=args.h_leaf_size,
            h_eta=args.h_eta,
            h_tolerance=args.h_tol,
            h_max_rank=args.h_max_rank,
            h_stored_entries=h_stats["memory_entries"],
            h_low_rank_blocks=h_stats["low_rank"],
            h_dense_blocks=h_stats["dense"],
            source_queried_entries=query_stats["queried_entries"],
        )
    if args.sampling_only:
        print(f"saved reference compliance data to {output_dir / 'compliance.npz'}")
        return

    strategy_total_seconds = perf_counter() - strategy_start
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
        f"strategy timings: factorization={factorization_seconds:.3f}s, "
        f"inflation solve={inflation_solve_seconds:.3f}s, "
        f"compliance load={compliance_load_seconds:.3f}s, "
        f"compliance sampling={compliance_sampling_seconds:.3f}s, "
        f"operator build={operator_build_seconds:.3f}s, "
        f"contact solve={contact_solve_seconds:.3f}s, "
        f"verification={verification_seconds:.3f}s, "
        f"total={strategy_total_seconds:.3f}s"
    )
    if args.compliance_strategy == "fe_matrix_free":
        print(
            "FE matrix-free statistics: "
            f"operator applications={int(fe_totals['operator_applications'])}, "
            f"linear solves={int(fe_totals['linear_solves'])}, "
            f"cache hits={int(fe_totals['cache_hits'])}, "
            f"zero bypasses={int(fe_totals['zero_bypasses'])}, "
            f"linear-solve time={fe_totals['solve_seconds']:.3f}s"
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
    ksp.solve(rhs, displacement)
    final_solve_seconds = perf_counter() - start

    u_result = fem.Function(V)
    u_result.name = "displacement"
    displacement_values = displacement.getArray(readonly=True)
    if len(u_result.x.array) != len(displacement_values):
        raise RuntimeError("Unexpected serial PETSc/DOLFINx vector-size mismatch")
    u_result.x.array[:] = displacement_values
    u_result.x.scatter_forward()

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

    vtk_path = output_dir / f"tyre_dihedral_contact_{args.compliance_strategy}.pvd"
    with VTKFile(comm, str(vtk_path), "w") as vtk:
        vtk.write_mesh(mesh)
        vtk.write_function([u_result, nodal_force, potential_zone])

    result_payload = {
        "force": full_force,
        "gap": full_gap,
        "clearance": full_clearance,
        "candidate_indices": candidate_indices,
        "initial_potential_contact_unknowns": initial_candidate_count,
        "potential_contact_unknowns": candidate_indices.size,
        "global_contact_unknowns": full_unknowns,
        "warning_distance": args.warning_distance,
        "warning_verification_tol": args.warning_verification_tol,
        "potential_rounds": solve_rounds,
        "axial_divisions": args.axial_divisions,
        "circumferential_divisions": args.circumferential_divisions,
        "scale": args.scale,
        "indentation": args.indentation,
        "young_modulus": args.young_modulus,
        "poisson_ratio": args.poisson_ratio,
        "inflation_pressure": args.inflation_pressure,
        "displacement": displacement_values,
        "inflation_displacement": inflation_values,
        "residual": result.residual,
        "status": result.status.value,
        "iterations": result.iterations,
        "potential_total_iterations": total_iterations,
        "contact_solver": args.contact_solver,
        "preconditioner": preconditioner_name,
        "compliance_strategy": args.compliance_strategy,
        "factorization_seconds": factorization_seconds,
        "inflation_solve_seconds": inflation_solve_seconds,
        "compliance_load_seconds": compliance_load_seconds,
        "compliance_sampling_seconds": compliance_sampling_seconds,
        "operator_build_seconds": operator_build_seconds,
        "contact_solve_seconds": contact_solve_seconds,
        "verification_seconds": verification_seconds,
        "strategy_total_seconds": strategy_total_seconds,
        "final_solve_seconds": final_solve_seconds,
        "compliance_stored_entries": (
            h_stats["memory_entries"]
            if args.compliance_strategy == "hmatrix"
            else 0
        ),
        "fe_operator_applications": int(fe_totals["operator_applications"]),
        "fe_linear_solves": int(fe_totals["linear_solves"]),
        "fe_cache_hits": int(fe_totals["cache_hits"]),
        "fe_zero_bypasses": int(fe_totals["zero_bypasses"]),
        "fe_linear_solve_seconds": fe_totals["solve_seconds"],
    }
    np.savez(output_dir / "contact_result.npz", **result_payload)
    strategy_result_path = (
        output_dir / f"contact_result_{args.compliance_strategy}.npz"
    )
    np.savez(strategy_result_path, **result_payload)
    print(f"wrote contact results under {output_dir}")


if __name__ == "__main__":
    main()

import numpy as np
import pytest
from petsc4py import PETSc

from cofebem.fenics.dihedral_compliance import (
    compact_normal_compliance_samples,
    DihedralComplianceEntrySource,
    FactorizedComplianceOperator,
    FactorizedComplianceEntrySource,
    IterativeComplianceOperator,
    LocalDihedralComplianceEntrySource,
    RestrictedLocalDihedralComplianceEntrySource,
    create_lu_solver,
    create_iterative_solver,
    dihedral_reflection_error,
    infer_meridian_shape,
    infer_regular_sector_shape,
    load_dihedral_compliance_archive,
    memory_map_compliance_samples,
    order_contact_meridians,
    order_contact_sectors,
    order_unstructured_contact,
    dilate_sector_axial_mask,
    potential_contact_indices,
    probe_spd_operator,
    restrict_local_dihedral_compliance_samples,
    reconstruct_vertical_compliance,
    restricted_source_clearance,
)
from cofebem.fenics.petsc_schur import (
    MumpsSchurComplianceOperator,
    selected_schur_memory_estimate,
)
from cofebem.lcp import LCP, solve


def test_order_contact_sectors_recovers_sector_and_axial_order():
    n_sectors = 4
    x_values = np.array([-1.0, 0.0, 1.0])
    points = []
    labels = []
    for sector in range(n_sectors):
        angle = 2.0 * np.pi * sector / n_sectors
        for axial, x in enumerate(x_values):
            points.append([x, 2.0 * np.cos(angle), 2.0 * np.sin(angle)])
            labels.append(sector * len(x_values) + axial)

    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(points))
    points = np.asarray(points)[permutation]
    labels = np.asarray(labels, dtype=np.int32)[permutation]
    ordering = order_contact_sectors(
        points,
        scalar_dofs=labels,
        parent_y_dofs=labels + 100,
        parent_z_dofs=labels + 200,
        n_sectors=n_sectors,
    )

    np.testing.assert_array_equal(
        ordering.scalar_dofs,
        np.arange(n_sectors * len(x_values)).reshape(n_sectors, len(x_values)),
    )
    np.testing.assert_allclose(
        ordering.points[:, :, 0], np.tile(x_values, (n_sectors, 1))
    )


def test_unstructured_contact_order_is_flat_deterministic_and_aligned():
    points = np.array(
        [
            [1.0, 0.0, -2.0],
            [-1.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.5, -2.0, 0.0],
        ]
    )
    labels = np.array([10, 20, 30, 40], dtype=np.int32)

    ordering = order_unstructured_contact(
        points,
        labels,
        labels + 100,
        labels + 200,
    )

    assert not ordering.structured
    assert ordering.scalar_dofs.shape == (1, 4)
    np.testing.assert_array_equal(
        ordering.parent_z_dofs - ordering.scalar_dofs, 200
    )
    np.testing.assert_array_equal(
        np.sort(ordering.scalar_dofs.ravel()), np.sort(labels)
    )


def test_infer_regular_sector_shape_from_unordered_translated_points():
    n_sectors = 10
    n_axial = 4
    points = []
    for sector in range(n_sectors):
        angle = 2.0 * np.pi * sector / n_sectors
        for axial in range(n_axial):
            radius = 2.0 + 0.1 * axial
            points.append(
                [axial, 3.0 + radius * np.cos(angle), -1.0 + radius * np.sin(angle)]
            )
    points = np.asarray(points)[np.random.default_rng(9).permutation(40)]

    assert infer_regular_sector_shape(points, axis_yz=(3.0, -1.0)) == (10, 4)


def test_infer_and_order_nonuniform_meridians():
    expected_angles = np.array([0.0, 0.2, 0.7, 2.0, 4.0])
    x = np.array([-1.0, 0.0, 1.0])
    points = np.array(
        [
            [x_value, 2.0 * np.cos(angle), 2.0 * np.sin(angle)]
            for angle in expected_angles
            for x_value in x
        ]
    )
    permutation = np.random.default_rng(4).permutation(len(points))
    shuffled = points[permutation]
    dofs = np.arange(len(points), dtype=np.int32)[permutation]

    angles, n_axial = infer_meridian_shape(shuffled)
    ordering = order_contact_meridians(
        shuffled, dofs, dofs + 100, dofs + 200, expected_angles
    )

    np.testing.assert_allclose(angles, expected_angles)
    assert n_axial == 3
    np.testing.assert_allclose(ordering.sector_angles, expected_angles)
    np.testing.assert_allclose(ordering.points[:, :, 0], np.tile(x, (5, 1)))


def test_infer_regular_sector_shape_rejects_unequal_meridians():
    points = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    with pytest.raises(ValueError, match="meridian point counts differ"):
        infer_regular_sector_shape(points)


def test_reconstruct_vertical_compliance_gives_expected_circulant_matrix():
    n_sectors = 6
    kernel = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    samples = np.zeros((2, 1, 2, n_sectors, 1))
    for delta, value in enumerate(kernel):
        samples[:, 0, :, delta, 0] = value * np.eye(2)

    Sc = reconstruct_vertical_compliance(samples)
    expected = np.empty_like(Sc)
    for target in range(n_sectors):
        for source in range(n_sectors):
            expected[target, source] = kernel[(target - source) % n_sectors]

    np.testing.assert_allclose(Sc, expected, atol=1.0e-14)
    assert dihedral_reflection_error(samples) == 0.0


def test_entry_source_matches_dense_reconstruction_for_selected_cross():
    rng = np.random.default_rng(7)
    samples = rng.standard_normal((2, 3, 2, 6, 3))
    dense = reconstruct_vertical_compliance(samples)
    source = DihedralComplianceEntrySource(samples)
    rows = np.array([0, 4, 11, 17])
    columns = np.array([1, 7, 13])

    np.testing.assert_allclose(
        source.get_block(rows, columns), dense[np.ix_(rows, columns)]
    )
    assert source.stats()["largest_query"] == (4, 3)


def test_local_dihedral_source_uses_open_offsets_and_reciprocity():
    kernel = np.array([4.0, 2.0, 0.5, 0.125])
    samples = np.zeros((3, 1, kernel.size, 1))
    samples[0, 0, :, 0] = kernel
    samples[2, 0, :, 0] = kernel
    source = LocalDihedralComplianceEntrySource(
        samples, sector_step=np.deg2rad(10.0)
    )
    indices = np.arange(kernel.size)
    matrix = source.get_block(indices, indices)
    expected = kernel[np.abs(indices[:, None] - indices[None, :])]

    np.testing.assert_allclose(matrix, expected, atol=1.0e-14)
    np.testing.assert_allclose(matrix, matrix.T, atol=0.0)
    assert source.reciprocity_error(sample_size=64) == 0.0


def test_restricted_local_source_matches_candidate_principal_matrix():
    rng = np.random.default_rng(1729)
    n_sectors = 7
    n_axial = 5
    samples = rng.standard_normal((3, n_axial, n_sectors, n_axial))
    sector_step = np.deg2rad(2.0)
    candidates = np.array([6, 7, 8, 12, 13, 17, 18, 22, 23, 27, 28])
    full = LocalDihedralComplianceEntrySource(
        samples, sector_step=sector_step
    )
    restricted = restrict_local_dihedral_compliance_samples(
        samples,
        candidates,
        sector_step=sector_step,
    )

    assert isinstance(restricted, RestrictedLocalDihedralComplianceEntrySource)
    assert restricted.shape == (candidates.size, candidates.size)
    assert restricted.samples.size < samples.size
    np.testing.assert_allclose(
        restricted.get_block(
            np.arange(candidates.size), np.arange(candidates.size)
        ),
        full.get_block(candidates, candidates),
    )
    assert restricted.stats()["sampled_axial_nodes"] == 3


def test_compact_normal_samples_preserve_all_scalar_entries():
    rng = np.random.default_rng(17)
    transverse = rng.standard_normal((2, 3, 2, 6, 3))
    compact = compact_normal_compliance_samples(transverse)
    indices = np.arange(18)

    assert compact.shape == (3, 3, 6, 3)
    np.testing.assert_allclose(
        DihedralComplianceEntrySource(compact).get_block(indices, indices),
        DihedralComplianceEntrySource(transverse).get_block(indices, indices),
    )


def test_dihedral_reflection_error_detects_wrong_cross_component_parity():
    samples = np.zeros((2, 1, 2, 4, 1))
    samples[0, 0, 1, 1, 0] = 1.0
    samples[0, 0, 1, 3, 0] = 1.0  # should have the opposite sign

    assert dihedral_reflection_error(samples) > 1.0


def test_potential_contact_indices_and_periodic_halo():
    gap = np.array([0.3, -0.1, 0.02, 0.08, 0.5, 0.01])
    np.testing.assert_array_equal(
        potential_contact_indices(gap, 0.02), np.array([1, 2, 5])
    )
    mask = np.zeros((4, 3), dtype=bool)
    mask[0, 1] = True
    expanded = dilate_sector_axial_mask(mask, halo=1)
    assert expanded[3, 1] and expanded[1, 1]
    assert expanded[0, 0] and expanded[0, 2]
    assert expanded.sum() == 5


def test_restricted_source_clearance_matches_dense_product():
    rng = np.random.default_rng(21)
    matrix = rng.standard_normal((9, 9))
    gap = rng.standard_normal(9)
    indices = np.array([1, 4, 7])
    forces = np.array([2.0, 0.5, 1.25])

    class Source:
        shape = matrix.shape

        def get_block(self, rows, columns):
            return matrix[np.ix_(rows, columns)]

    clearance = restricted_source_clearance(
        Source(), gap, indices, forces, chunk_size=2
    )
    np.testing.assert_allclose(clearance, gap + matrix[:, indices] @ forces)


def test_factorized_compliance_operator_matches_inverse_and_caches_solution():
    matrix = np.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ]
    )
    A = PETSc.Mat().createAIJ([4, 4], comm=PETSc.COMM_SELF)
    dofs = np.arange(4, dtype=np.int32)
    A.setValues(dofs, dofs, matrix)
    A.assemble()
    ksp = create_lu_solver(A, PETSc.COMM_SELF)
    selected = np.array([1, 3], dtype=np.int32)
    operator = FactorizedComplianceOperator(A, ksp, selected)
    forces = np.array([2.0, -0.5])
    rhs = np.zeros(4)
    rhs[selected] = forces
    expected_full = np.linalg.solve(matrix, rhs)

    np.testing.assert_allclose(operator @ forces, expected_full[selected])
    np.testing.assert_allclose(
        operator.apply(forces, response_dofs=dofs), expected_full
    )
    stats = operator.stats()
    assert stats["operator_applications"] == 2
    assert stats["linear_solves"] == 1
    assert stats["cache_hits"] == 1

    np.testing.assert_array_equal(operator @ np.zeros(2), np.zeros(2))
    assert operator.stats()["zero_bypasses"] == 1

    dense_compliance = np.linalg.inv(matrix)[np.ix_(selected, selected)]
    expected_pressure = np.array([1.5, 0.75])
    result = solve(
        LCP(operator, -(dense_compliance @ expected_pressure)),
        method="ppcg",
        tol=1.0e-12,
    )
    assert result.converged
    np.testing.assert_allclose(result.z, expected_pressure, rtol=1.0e-9)


def test_factorized_entry_source_uses_reciprocity_and_cached_columns():
    matrix, A = _small_spd_petsc_matrix()
    ksp = create_lu_solver(A, PETSc.COMM_SELF)
    source = FactorizedComplianceEntrySource(
        A, ksp, np.arange(4, dtype=np.int32)
    )
    exact = np.linalg.inv(matrix)

    np.testing.assert_allclose(
        source.get_block(np.array([0]), np.array([1, 2, 3])),
        exact[np.ix_([0], [1, 2, 3])],
    )
    assert source.stats()["linear_solves"] == 1
    np.testing.assert_allclose(
        source.get_block(np.array([0, 2]), np.array([1, 3])),
        exact[np.ix_([0, 2], [1, 3])],
    )
    assert source.stats()["linear_solves"] == 2
    assert source.stats()["cached_columns"] == 2


def _small_spd_petsc_matrix():
    matrix = np.array(
        [
            [4.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ]
    )
    A = PETSc.Mat().createAIJ([4, 4], comm=PETSc.COMM_SELF)
    dofs = np.arange(4, dtype=np.int32)
    A.setValues(dofs, dofs, matrix)
    A.assemble()
    return matrix, A


def test_iterative_compliance_operator_matches_direct_action():
    matrix, A = _small_spd_petsc_matrix()
    ksp = create_iterative_solver(
        A,
        PETSc.COMM_SELF,
        pc_type="jacobi",
        relative_tolerance=1.0e-13,
        absolute_tolerance=1.0e-15,
        max_iterations=100,
        options_prefix="cofebem_test_fe_",
    )
    selected = np.array([1, 3], dtype=np.int32)
    operator = IterativeComplianceOperator(A, ksp, selected)
    forces = np.array([2.0, -0.5])
    rhs = np.zeros(4)
    rhs[selected] = forces

    np.testing.assert_allclose(
        operator @ forces,
        np.linalg.solve(matrix, rhs)[selected],
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert operator.stats()["linear_iterations"] > 0
    assert operator.stats()["maximum_linear_iterations"] > 0
    diagnostics = probe_spd_operator(operator)
    assert diagnostics["reciprocity_error"] < 1.0e-12
    assert diagnostics["minimum_probed_rayleigh"] > 0.0


@pytest.mark.parametrize("factor_type", ["lu", "cholesky"])
def test_selected_schur_memory_guard_and_exact_action(factor_type):
    assert selected_schur_memory_estimate(10) == 300 * np.dtype(
        PETSc.ScalarType
    ).itemsize
    matrix, A = _small_spd_petsc_matrix()
    selected = np.array([1, 3], dtype=np.int32)
    try:
        operator = MumpsSchurComplianceOperator(
            A, selected, factor_type=factor_type
        )
    except RuntimeError as exc:
        if "optional native bridge" in str(exc):
            pytest.skip(str(exc))
        raise
    forces = np.array([2.0, -0.5])
    rhs = np.zeros(4)
    rhs[selected] = forces

    np.testing.assert_allclose(
        operator @ forces,
        np.linalg.solve(matrix, rhs)[selected],
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    with pytest.raises(MemoryError, match="estimated to require"):
        MumpsSchurComplianceOperator(A, selected, max_memory_gib=1.0e-12)


def _write_compliance_archive(path, samples, points, **metadata):
    np.savez(path, samples=samples, points=points, **metadata)


def test_load_compliance_accepts_rigid_translation(tmp_path):
    samples = np.ones((2, 3, 2, 4, 3))
    points = np.arange(36, dtype=float).reshape(12, 3)
    path = tmp_path / "compliance.npz"
    _write_compliance_archive(
        path,
        samples,
        points,
        axial_divisions=2,
        circumferential_divisions=4,
        young_modulus=2.5e8,
        poisson_ratio=0.48,
    )

    loaded = load_dihedral_compliance_archive(
        path,
        points + np.array([0.0, 0.0, 0.005]),
        n_axial=3,
        n_sectors=4,
        young_modulus=2.5e8,
        poisson_ratio=0.48,
    )

    np.testing.assert_array_equal(loaded, samples)


def test_load_local_patch_archive_validates_tag_and_interval_count(tmp_path):
    samples = np.ones((3, 2, 5, 2))
    points = np.arange(30, dtype=float).reshape(10, 3)
    path = tmp_path / "local_compliance.npz"
    _write_compliance_archive(
        path,
        samples,
        points,
        axial_divisions=1,
        circumferential_divisions=4,
        local_symmetry_tag=204,
    )

    loaded = load_dihedral_compliance_archive(
        path,
        points,
        n_axial=2,
        n_sectors=5,
        circumferential_divisions=4,
        local_symmetry_tag=204,
    )
    np.testing.assert_array_equal(loaded, samples)

    with pytest.raises(ValueError, match="local_symmetry_tag"):
        load_dihedral_compliance_archive(
            path,
            points,
            n_axial=2,
            n_sectors=5,
            circumferential_divisions=4,
            local_symmetry_tag=205,
        )


def test_load_compliance_memory_mapped_sidecar(tmp_path):
    samples = np.arange(96, dtype=float).reshape(2, 2, 2, 6, 2)
    points = np.arange(36, dtype=float).reshape(12, 3)
    samples_path = tmp_path / "compliance_samples.npy"
    mapped = memory_map_compliance_samples(samples, samples_path)
    np.savez(
        tmp_path / "compliance.npz",
        samples_file=samples_path.name,
        points=points,
        axial_divisions=1,
        circumferential_divisions=6,
    )

    loaded = load_dihedral_compliance_archive(
        tmp_path / "compliance.npz",
        points,
        n_axial=2,
        n_sectors=6,
    )

    assert isinstance(mapped, np.memmap)
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(
        loaded, compact_normal_compliance_samples(samples)
    )


def test_load_compliance_rejects_geometry_or_material_mismatch(tmp_path):
    samples = np.ones((2, 2, 2, 3, 2))
    points = np.arange(18, dtype=float).reshape(6, 3)
    path = tmp_path / "compliance.npz"
    _write_compliance_archive(
        path,
        samples,
        points,
        axial_divisions=1,
        circumferential_divisions=3,
        young_modulus=10.0,
        poisson_ratio=0.3,
    )

    changed_points = points.copy()
    changed_points[1, 0] += 0.1
    with pytest.raises(ValueError, match="geometry or sector-major ordering"):
        load_dihedral_compliance_archive(
            path, changed_points, n_axial=2, n_sectors=3
        )

    with pytest.raises(ValueError, match="young_modulus"):
        load_dihedral_compliance_archive(
            path,
            points,
            n_axial=2,
            n_sectors=3,
            young_modulus=11.0,
            poisson_ratio=0.3,
        )


def test_load_legacy_compliance_warns_about_unverified_material(tmp_path):
    samples = np.ones((2, 1, 2, 2, 1))
    points = np.arange(6, dtype=float).reshape(2, 3)
    path = tmp_path / "legacy.npz"
    _write_compliance_archive(path, samples, points)

    with pytest.warns(UserWarning, match="material compatibility"):
        loaded = load_dihedral_compliance_archive(
            path,
            points,
            n_axial=1,
            n_sectors=2,
            young_modulus=10.0,
            poisson_ratio=0.3,
        )

    np.testing.assert_array_equal(loaded, samples)


def test_load_compliance_rejects_missing_or_changed_boundary_condition(tmp_path):
    samples = np.ones((2, 1, 2, 2, 1))
    points = np.arange(6, dtype=float).reshape(2, 3)
    legacy_path = tmp_path / "legacy_boundary.npz"
    _write_compliance_archive(legacy_path, samples, points)

    with pytest.raises(ValueError, match="does not identify its boundary"):
        load_dihedral_compliance_archive(
            legacy_path,
            points,
            n_axial=1,
            n_sectors=2,
            boundary_condition_id="disk_edge_short_curves_v1",
        )

    current_path = tmp_path / "current_boundary.npz"
    _write_compliance_archive(
        current_path,
        samples,
        points,
        boundary_condition_id="disk_edge_short_curves_v1",
    )
    with pytest.raises(ValueError, match="boundary_condition_id"):
        load_dihedral_compliance_archive(
            current_path,
            points,
            n_axial=1,
            n_sectors=2,
            boundary_condition_id="another_constraint",
        )

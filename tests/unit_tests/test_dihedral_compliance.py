import numpy as np

from cofebem.fenics.dihedral_compliance import (
    DihedralComplianceEntrySource,
    dihedral_reflection_error,
    order_contact_sectors,
    reconstruct_vertical_compliance,
)


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


def test_dihedral_reflection_error_detects_wrong_cross_component_parity():
    samples = np.zeros((2, 1, 2, 4, 1))
    samples[0, 0, 1, 1, 0] = 1.0
    samples[0, 0, 1, 3, 0] = 1.0  # should have the opposite sign

    assert dihedral_reflection_error(samples) > 1.0

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh

from cofebem.fenics.contact_postprocess import (
    force_based_contact_pressure,
    project_compressive_normal_stress,
    surface_lumped_nodal_areas,
)


def _unit_cube_top_surface():
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 2, 2, 2, cell_type=mesh.CellType.hexahedron
    )
    fdim = domain.topology.dim - 1
    facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], 1.0)
    )
    facets = np.sort(facets.astype(np.int32))
    tags = mesh.meshtags(
        domain, fdim, facets, np.full(facets.size, 7, dtype=np.int32)
    )
    scalar_space = fem.functionspace(domain, ("Lagrange", 1))
    dofs = fem.locate_dofs_topological(scalar_space, fdim, facets)
    return domain, tags, scalar_space, np.asarray(dofs, dtype=np.int32)


def test_lumped_nodal_areas_and_force_pressure_preserve_resultant():
    _, tags, scalar_space, dofs = _unit_cube_top_surface()
    areas = surface_lumped_nodal_areas(scalar_space, tags, 7, dofs)
    forces = 3.5 * areas
    pressure = force_based_contact_pressure(
        scalar_space, dofs, forces, areas
    )

    assert pressure.name == "contact_pressure_force_based"
    np.testing.assert_allclose(areas.sum(), 1.0)
    np.testing.assert_allclose(pressure.x.array[dofs], 3.5)
    np.testing.assert_allclose(pressure.x.array[dofs] @ areas, forces.sum())


def test_stress_projection_recovers_uniform_compressive_pressure():
    domain, tags, scalar_space, dofs = _unit_cube_top_surface()
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(vector_space)
    strain_zz = 0.01
    displacement.interpolate(
        lambda x: np.vstack(
            (
                np.zeros(x.shape[1]),
                np.zeros(x.shape[1]),
                -strain_zz * x[2],
            )
        )
    )
    young_modulus = 100.0
    poisson_ratio = 0.25
    pressure = project_compressive_normal_stress(
        displacement,
        scalar_space,
        tags,
        7,
        dofs,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    expected = (lmbda + 2.0 * mu) * strain_zz

    assert pressure.name == "contact_pressure_stress"
    np.testing.assert_allclose(pressure.x.array[dofs], expected, rtol=1.0e-12)

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh

from cofebem.fenics.contact_postprocess import (
    EquilibratedContactStressProjector,
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

    areas = surface_lumped_nodal_areas(scalar_space, tags, 7, dofs)
    lumped_pressure = project_compressive_normal_stress(
        displacement,
        scalar_space,
        tags,
        7,
        dofs,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        projection="lumped",
        nodal_areas=areas,
    )
    np.testing.assert_allclose(
        lumped_pressure.x.array[dofs], expected, rtol=1.0e-12
    )


def test_equilibrated_stress_recovery_uses_discrete_boundary_residual():
    domain = mesh.create_unit_cube(
        MPI.COMM_SELF, 2, 2, 2, cell_type=mesh.CellType.hexahedron
    )
    fdim = domain.topology.dim - 1
    facets = np.sort(
        mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.isclose(x[2], 0.0)
        ).astype(np.int32)
    )
    tags = mesh.meshtags(
        domain, fdim, facets, np.full(facets.size, 9, dtype=np.int32)
    )
    vector_space = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    scalar_space, scalar_to_vector = vector_space.sub(2).collapse()
    dofs = np.asarray(
        fem.locate_dofs_topological(scalar_space, fdim, facets),
        dtype=np.int32,
    )
    parent_z = np.asarray(scalar_to_vector, dtype=np.int32)[dofs]
    areas = surface_lumped_nodal_areas(scalar_space, tags, 9, dofs)

    displacement = fem.Function(vector_space)
    pressure_value = 3.5
    displacement.x.array[parent_z] = pressure_value * areas
    size = displacement.x.array.size
    stiffness = PETSc.Mat().createAIJ([size, size], comm=domain.comm)
    stiffness.setUp()
    diagonal = PETSc.Vec().createSeq(size, comm=domain.comm)
    diagonal.set(1.0)
    stiffness.setDiagonal(diagonal)
    stiffness.assemble()

    pressure = EquilibratedContactStressProjector(
        displacement,
        stiffness,
        scalar_space,
        tags,
        9,
        dofs,
        parent_z,
        nodal_areas=areas,
    ).project()

    np.testing.assert_allclose(
        pressure.x.array[dofs], pressure_value, rtol=1.0e-12
    )

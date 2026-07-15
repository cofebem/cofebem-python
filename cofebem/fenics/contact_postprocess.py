"""Contact-surface area and pressure recovery for serial FEniCSx models."""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    dot,
    grad,
    inner,
    sym,
    tr,
)


def surface_lumped_nodal_areas(
    scalar_space,
    facet_tags,
    facet_tag: int,
    surface_dofs: np.ndarray,
) -> np.ndarray:
    """Return consistent nodal areas ``integral(N_i, Gamma)``."""
    mesh = scalar_space.mesh
    _require_serial(mesh)
    dofs = _validate_surface_dofs(scalar_space, surface_dofs)
    test = TestFunction(scalar_space)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    area_vector = assemble_vector(fem.form(test * ds(facet_tag)))
    area_vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
    )
    areas = np.array(
        area_vector.getArray(readonly=True)[dofs], dtype=np.float64, copy=True
    )
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise RuntimeError("contact surface contains non-positive nodal area")
    return areas


def force_based_contact_pressure(
    scalar_space,
    surface_dofs: np.ndarray,
    nodal_forces: np.ndarray,
    nodal_areas: np.ndarray,
    *,
    name: str = "contact_pressure_force_based",
):
    """Create a scalar field from nodal force divided by associated area."""
    dofs = _validate_surface_dofs(scalar_space, surface_dofs)
    forces = np.asarray(nodal_forces, dtype=np.float64).reshape(-1)
    areas = np.asarray(nodal_areas, dtype=np.float64).reshape(-1)
    if forces.shape != (dofs.size,) or areas.shape != (dofs.size,):
        raise ValueError("nodal forces and areas must match surface_dofs")
    if not np.all(np.isfinite(forces)) or np.any(areas <= 0.0):
        raise ValueError("nodal forces/areas are invalid")
    pressure = fem.Function(scalar_space)
    pressure.name = name
    pressure.x.array[:] = 0.0
    pressure.x.array[dofs] = forces / areas
    pressure.x.scatter_forward()
    return pressure


def project_compressive_normal_stress(
    displacement,
    scalar_space,
    facet_tags,
    facet_tag: int,
    surface_dofs: np.ndarray,
    *,
    young_modulus: float,
    poisson_ratio: float,
    name: str = "contact_pressure_stress",
):
    """L2-project ``-n . sigma(u) . n`` onto contact-surface scalar DOFs."""
    mesh = scalar_space.mesh
    _require_serial(mesh)
    dofs = _validate_surface_dofs(scalar_space, surface_dofs)
    if young_modulus <= 0.0 or not 0.0 <= poisson_ratio < 0.5:
        raise ValueError("invalid isotropic elastic constants")

    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    strain = sym(grad(displacement))
    stress = lmbda * tr(strain) * Identity(mesh.geometry.dim) + 2.0 * mu * strain
    normal = FacetNormal(mesh)
    compressive_pressure = -inner(dot(stress, normal), normal)

    trial = TrialFunction(scalar_space)
    test = TestFunction(scalar_space)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    mass = assemble_matrix(fem.form(inner(trial, test) * ds(facet_tag)))
    mass.assemble()
    rhs = assemble_vector(
        fem.form(compressive_pressure * test * ds(facet_tag))
    )
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    index_set = PETSc.IS().createGeneral(dofs, comm=mesh.comm)
    surface_mass = mass.createSubMatrix(index_set, index_set)
    rhs_values = np.asarray(rhs.getValues(dofs), dtype=PETSc.ScalarType)
    solution_values = np.zeros(dofs.size, dtype=PETSc.ScalarType)
    surface_rhs = PETSc.Vec().createWithArray(rhs_values, comm=mesh.comm)
    surface_solution = PETSc.Vec().createWithArray(
        solution_values, comm=mesh.comm
    )
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(surface_mass)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.setUp()
    solver.solve(surface_rhs, surface_solution)
    if int(solver.getConvergedReason()) <= 0:
        raise RuntimeError(
            "contact stress projection failed with PETSc reason "
            f"{solver.getConvergedReason()}"
        )

    pressure = fem.Function(scalar_space)
    pressure.name = name
    pressure.x.array[:] = 0.0
    pressure.x.array[dofs] = np.asarray(solution_values, dtype=np.float64)
    pressure.x.scatter_forward()
    return pressure


def _validate_surface_dofs(scalar_space, surface_dofs: np.ndarray) -> np.ndarray:
    dofs = np.asarray(surface_dofs, dtype=np.int32).reshape(-1)
    if dofs.size == 0 or np.unique(dofs).size != dofs.size:
        raise ValueError("surface_dofs must be nonempty and unique")
    coordinates = scalar_space.tabulate_dof_coordinates()
    if np.any(dofs < 0) or np.any(dofs >= coordinates.shape[0]):
        raise IndexError("surface DOF outside scalar function space")
    return dofs


def _require_serial(mesh) -> None:
    if mesh.comm.size != 1:
        raise NotImplementedError("contact pressure recovery is currently serial")

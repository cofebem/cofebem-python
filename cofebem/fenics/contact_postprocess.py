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


class ContactStressProjector:
    """Reusable consistent or mass-lumped normal-stress recovery.

    The lumped method divides the assembled stress load vector by the same
    consistent nodal areas used for force-based pressure. It avoids assembling
    and factorizing a second surface matrix, which is essential when the bulk
    elasticity LU already occupies most available memory.
    """

    def __init__(
        self,
        displacement,
        scalar_space,
        facet_tags,
        facet_tag: int,
        surface_dofs: np.ndarray,
        *,
        young_modulus: float,
        poisson_ratio: float,
        projection: str = "lumped",
        nodal_areas: np.ndarray | None = None,
        name: str = "contact_pressure_stress",
    ) -> None:
        mesh = scalar_space.mesh
        _require_serial(mesh)
        self.dofs = _validate_surface_dofs(scalar_space, surface_dofs)
        if young_modulus <= 0.0 or not 0.0 <= poisson_ratio < 0.5:
            raise ValueError("invalid isotropic elastic constants")
        if projection not in {"lumped", "consistent"}:
            raise ValueError("stress projection must be 'lumped' or 'consistent'")

        lmbda = young_modulus * poisson_ratio / (
            (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
        )
        mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
        strain = sym(grad(displacement))
        stress = (
            lmbda * tr(strain) * Identity(mesh.geometry.dim) + 2.0 * mu * strain
        )
        normal = FacetNormal(mesh)
        compressive_pressure = -inner(dot(stress, normal), normal)
        test = TestFunction(scalar_space)
        ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
        self._rhs_form = fem.form(
            compressive_pressure * test * ds(facet_tag)
        )
        self.projection = projection
        self.pressure = fem.Function(scalar_space)
        self.pressure.name = name
        self._solver = None
        self._surface_rhs = None
        self._surface_solution = None

        if projection == "lumped":
            areas = (
                surface_lumped_nodal_areas(
                    scalar_space, facet_tags, facet_tag, self.dofs
                )
                if nodal_areas is None
                else np.asarray(nodal_areas, dtype=np.float64).reshape(-1)
            )
            if areas.shape != (self.dofs.size,) or np.any(areas <= 0.0):
                raise ValueError("nodal_areas must be positive and match surface_dofs")
            self._nodal_areas = areas.copy()
        else:
            trial = TrialFunction(scalar_space)
            mass = assemble_matrix(
                fem.form(inner(trial, test) * ds(facet_tag))
            )
            mass.assemble()
            index_set = PETSc.IS().createGeneral(self.dofs, comm=mesh.comm)
            surface_mass = mass.createSubMatrix(index_set, index_set)
            self._surface_rhs = PETSc.Vec().createSeq(self.dofs.size, comm=mesh.comm)
            self._surface_solution = PETSc.Vec().createSeq(
                self.dofs.size, comm=mesh.comm
            )
            self._solver = PETSc.KSP().create(mesh.comm)
            self._solver.setOperators(surface_mass)
            self._solver.setType(PETSc.KSP.Type.PREONLY)
            self._solver.getPC().setType(PETSc.PC.Type.LU)
            self._solver.setUp()

    def project(self):
        """Recover pressure for the displacement's current coefficient values."""
        rhs = assemble_vector(self._rhs_form)
        rhs.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        rhs_values = np.asarray(rhs.getValues(self.dofs), dtype=np.float64)
        if self.projection == "lumped":
            values = rhs_values / self._nodal_areas
        else:
            self._surface_rhs.getArray()[:] = rhs_values
            self._solver.solve(self._surface_rhs, self._surface_solution)
            if int(self._solver.getConvergedReason()) <= 0:
                raise RuntimeError(
                    "contact stress projection failed with PETSc reason "
                    f"{self._solver.getConvergedReason()}"
                )
            values = self._surface_solution.getArray(readonly=True)
        self.pressure.x.array[:] = 0.0
        self.pressure.x.array[self.dofs] = values
        self.pressure.x.scatter_forward()
        return self.pressure


def project_compressive_normal_stress(
    displacement,
    scalar_space,
    facet_tags,
    facet_tag: int,
    surface_dofs: np.ndarray,
    *,
    young_modulus: float,
    poisson_ratio: float,
    projection: str = "consistent",
    nodal_areas: np.ndarray | None = None,
    name: str = "contact_pressure_stress",
):
    """Project ``-n . sigma(u) . n`` onto contact-surface scalar DOFs."""
    projector = ContactStressProjector(
        displacement,
        scalar_space,
        facet_tags,
        facet_tag,
        surface_dofs,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        projection=projection,
        nodal_areas=nodal_areas,
        name=name,
    )
    return projector.project()


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

"""Contact-surface area and pressure recovery for serial FEniCSx models."""

from __future__ import annotations

from collections.abc import Sequence

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
    as_tensor,
    dot,
    dx,
    grad,
    inner,
    sym,
    tr,
)


def surface_lumped_nodal_areas(
    scalar_space,
    facet_tags,
    facet_tag: int | Sequence[int],
    surface_dofs: np.ndarray,
) -> np.ndarray:
    """Return consistent nodal areas ``integral(N_i, Gamma)``."""
    mesh = scalar_space.mesh
    _require_serial(mesh)
    dofs = _validate_surface_dofs(scalar_space, surface_dofs)
    test = TestFunction(scalar_space)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
    surface_measure = _tagged_surface_measure(ds, facet_tag)
    area_vector = assemble_vector(fem.form(test * surface_measure))
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
        facet_tag: int | Sequence[int],
        surface_dofs: np.ndarray,
        *,
        young_modulus: float,
        poisson_ratio: float,
        projection: str = "lumped",
        recovery: str = "nodal_average",
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
        if recovery not in {"raw", "nodal_average"}:
            raise ValueError("stress recovery must be 'raw' or 'nodal_average'")

        lmbda = young_modulus * poisson_ratio / (
            (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
        )
        mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
        strain = sym(grad(displacement))
        stress = (
            lmbda * tr(strain) * Identity(mesh.geometry.dim) + 2.0 * mu * strain
        )
        test = TestFunction(scalar_space)
        self.recovery = recovery
        self._stress_recovery = []
        if recovery == "nodal_average":
            volume_vector = assemble_vector(fem.form(test * dx))
            volume_vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            self._nodal_volumes = np.array(
                volume_vector.getArray(readonly=True),
                dtype=np.float64,
                copy=True,
            )
            if (
                not np.all(np.isfinite(self._nodal_volumes))
                or np.any(self._nodal_volumes <= 0.0)
            ):
                raise RuntimeError(
                    "volume stress recovery contains non-positive nodal volume"
                )

            recovered_components = {}
            for row in range(mesh.geometry.dim):
                for column in range(row, mesh.geometry.dim):
                    component = fem.Function(scalar_space)
                    component.name = f"recovered_stress_{row}{column}"
                    component_form = fem.form(
                        stress[row, column] * test * dx
                    )
                    self._stress_recovery.append(
                        (component, component_form)
                    )
                    recovered_components[(row, column)] = component
            stress_for_boundary = as_tensor(
                [
                    [
                        recovered_components[
                            (min(row, column), max(row, column))
                        ]
                        for column in range(mesh.geometry.dim)
                    ]
                    for row in range(mesh.geometry.dim)
                ]
            )
        else:
            stress_for_boundary = stress

        normal = FacetNormal(mesh)
        compressive_pressure = -inner(
            dot(stress_for_boundary, normal), normal
        )
        ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
        surface_measure = _tagged_surface_measure(ds, facet_tag)
        self._rhs_form = fem.form(
            compressive_pressure * test * surface_measure
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
                fem.form(inner(trial, test) * surface_measure)
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
        for component, component_form in self._stress_recovery:
            numerator = assemble_vector(component_form)
            numerator.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            component.x.array[:] = (
                numerator.getArray(readonly=True) / self._nodal_volumes
            )
            component.x.scatter_forward()
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


class EquilibratedContactStressProjector:
    """Recover boundary normal stress from the discrete FE equilibrium.

    A CG1 displacement has discontinuous element stress, so its strong
    boundary trace is generally oscillatory and does not satisfy the Neumann
    condition pointwise.  This projector instead evaluates the internal-force
    residual ``A @ u_contact`` at the contact DOFs.  By the FE weak form those
    nodal values are the boundary integral of ``sigma(u_contact) @ n``.  The
    resulting vertical traction is converted to pressure with the averaged
    outward-normal component and optionally projected through the consistent
    surface mass matrix.

    The current contact formulation applies forces in global z, so this class
    intentionally requires the matching parent z DOFs.
    """

    def __init__(
        self,
        displacement,
        stiffness,
        scalar_space,
        facet_tags,
        facet_tag: int | Sequence[int],
        surface_dofs: np.ndarray,
        parent_z_dofs: np.ndarray,
        *,
        projection: str = "lumped",
        nodal_areas: np.ndarray | None = None,
        name: str = "contact_pressure_stress",
    ) -> None:
        mesh = scalar_space.mesh
        _require_serial(mesh)
        self.displacement = displacement
        self.stiffness = stiffness
        self.dofs = _validate_surface_dofs(scalar_space, surface_dofs)
        self.parent_z_dofs = np.asarray(parent_z_dofs, dtype=np.int32).reshape(-1)
        if self.parent_z_dofs.shape != self.dofs.shape:
            raise ValueError("parent_z_dofs must match surface_dofs")
        rows, columns = stiffness.getSize()
        if rows != columns or np.any(self.parent_z_dofs >= rows):
            raise ValueError("parent_z_dofs are incompatible with stiffness")
        if projection not in {"lumped", "consistent"}:
            raise ValueError("stress projection must be 'lumped' or 'consistent'")

        self.areas = (
            surface_lumped_nodal_areas(
                scalar_space, facet_tags, facet_tag, self.dofs
            )
            if nodal_areas is None
            else np.asarray(nodal_areas, dtype=np.float64).reshape(-1)
        )
        if self.areas.shape != self.dofs.shape or np.any(self.areas <= 0.0):
            raise ValueError("nodal_areas must be positive and match surface_dofs")

        test = TestFunction(scalar_space)
        ds = Measure("ds", domain=mesh, subdomain_data=facet_tags)
        surface_measure = _tagged_surface_measure(ds, facet_tag)
        normal_z_vector = assemble_vector(
            fem.form(FacetNormal(mesh)[2] * test * surface_measure)
        )
        normal_z_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        self.normal_z = np.asarray(
            normal_z_vector.getValues(self.dofs), dtype=np.float64
        ) / self.areas
        self.projection = projection
        self.pressure = fem.Function(scalar_space)
        self.pressure.name = name
        self._displacement_vector = stiffness.createVecRight()
        self._internal_force = stiffness.createVecLeft()
        self._solver = None
        self._surface_rhs = None
        self._surface_solution = None
        if projection == "consistent":
            trial = TrialFunction(scalar_space)
            mass = assemble_matrix(
                fem.form(inner(trial, test) * surface_measure)
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
        """Recover the weakly equilibrated positive-compression pressure."""
        displacement_values = np.asarray(
            self.displacement.x.array, dtype=np.float64
        )
        vector_values = self._displacement_vector.getArray()
        if vector_values.shape != displacement_values.shape:
            raise RuntimeError("unexpected serial displacement vector layout")
        vector_values[:] = displacement_values
        self._displacement_vector.assemble()
        self.stiffness.mult(self._displacement_vector, self._internal_force)
        internal_z = np.asarray(
            self._internal_force.getValues(self.parent_z_dofs),
            dtype=np.float64,
        )
        scale = max(float(np.max(np.abs(internal_z), initial=0.0)), 1.0)
        active = internal_z > 1.0e-12 * scale
        if np.any(internal_z < -1.0e-10 * scale):
            raise RuntimeError(
                "equilibrated contact stress contains significant tensile "
                "boundary force"
            )
        if np.any(active & (self.normal_z >= -1.0e-8)):
            raise RuntimeError(
                "active global-z contact force lies on a non-downward-facing "
                "surface"
            )
        lumped_pressure = np.zeros_like(internal_z)
        lumped_pressure[active] = internal_z[active] / (
            -self.normal_z[active] * self.areas[active]
        )
        if self.projection == "lumped":
            values = lumped_pressure
        else:
            self._surface_rhs.getArray()[:] = lumped_pressure * self.areas
            self._solver.solve(self._surface_rhs, self._surface_solution)
            if int(self._solver.getConvergedReason()) <= 0:
                raise RuntimeError(
                    "equilibrated stress projection failed with PETSc reason "
                    f"{self._solver.getConvergedReason()}"
                )
            values = self._surface_solution.getArray(readonly=True)
        self.pressure.x.array[:] = 0.0
        self.pressure.x.array[self.dofs] = values
        self.pressure.x.scatter_forward()
        self.normal_resultant = float(values @ self.areas)
        self.vertical_resultant = float(
            values @ (-self.normal_z * self.areas)
        )
        self.internal_vertical_resultant = float(internal_z.sum())
        return self.pressure


def project_compressive_normal_stress(
    displacement,
    scalar_space,
    facet_tags,
    facet_tag: int | Sequence[int],
    surface_dofs: np.ndarray,
    *,
    young_modulus: float,
    poisson_ratio: float,
    projection: str = "consistent",
    recovery: str = "nodal_average",
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
        recovery=recovery,
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


def _tagged_surface_measure(ds, facet_tag: int | Sequence[int]):
    """Return the sum of one or more disjoint tagged boundary measures."""
    if np.isscalar(facet_tag):
        tags = (int(facet_tag),)
    else:
        tags = tuple(int(tag) for tag in facet_tag)
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("facet_tag must contain one or more unique tags")
    measure = ds(tags[0])
    for tag in tags[1:]:
        measure += ds(tag)
    return measure


def _require_serial(mesh) -> None:
    if mesh.comm.size != 1:
        raise NotImplementedError("contact pressure recovery is currently serial")

from pathlib import Path
from time import perf_counter

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    assemble_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import VTKFile
from dolfinx.mesh import CellType, create_box, locate_entities_boundary, meshtags

comm = MPI.COMM_WORLD
if comm.size != 1:
    raise RuntimeError("This script is written for serial execution.")
if np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError("This contact formulation requires real-valued PETSc.")

Lx, Ly, Lz = 1.0, 1.0, 1.0
nx, ny, nz = 20, 20, 20
E, nu = 1.0e9, 0.3
R, delta = 0.35, 0.1
nsteps = 10
quadrature_degree = 6
output_dir = Path("results/penalty_contact_slide")

mesh = create_box(
    comm,
    [np.zeros(3), np.array([Lx, Ly, Lz])],
    [nx, ny, nz],
    cell_type=CellType.tetrahedron,
)

tdim = mesh.topology.dim
fdim = tdim - 1
gdim = mesh.geometry.dim

lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
eps_n = Constant(mesh, PETSc.ScalarType(1.0e12))  # Penalty stiffness [Pa/m].
indentation = Constant(mesh, PETSc.ScalarType(delta))
xc = Constant(mesh, PETSc.ScalarType(0.0))

V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
u = Function(V)
u.name = "u"
tc_out = Function(V)
tc_out.name = "tc"
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)


def epsilon(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return lmbda * ufl.tr(epsilon(w)) * ufl.Identity(gdim) + 2.0 * mu * epsilon(w)


f = Constant(mesh, np.zeros(gdim, dtype=PETSc.ScalarType))
t0 = Constant(mesh, np.zeros(gdim, dtype=PETSc.ScalarType))
u0 = Constant(mesh, np.zeros(gdim, dtype=PETSc.ScalarType))

Gamma_u = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0.0))
Gamma_t = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
Gamma_c = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], Lz))

Gamma_u_id, Gamma_t_id, Gamma_c_id = 0, 1, 2
boundaries = (Gamma_u, Gamma_t, Gamma_c)
if any(facets.size == 0 for facets in boundaries):
    raise RuntimeError("One or more tagged boundaries are empty.")

facet_indices = np.concatenate(boundaries).astype(np.int32)
facet_values = np.concatenate(
    [
        np.full(Gamma_u.size, Gamma_u_id, dtype=np.int32),
        np.full(Gamma_t.size, Gamma_t_id, dtype=np.int32),
        np.full(Gamma_c.size, Gamma_c_id, dtype=np.int32),
    ]
)
if np.unique(facet_indices).size != facet_indices.size:
    raise ValueError("Gamma_u, Gamma_t and Gamma_c must not share facets.")

order = np.argsort(facet_indices)
mt = meshtags(mesh, fdim, facet_indices[order], facet_values[order])
ds = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=mt,
    metadata={"quadrature_degree": quadrature_degree},
)
dx = ufl.Measure("dx", domain=mesh)
ds_c = ds(Gamma_c_id)

Gamma_u_dofs = locate_dofs_topological(V, fdim, Gamma_u)
bc = dirichletbc(u0, Gamma_u_dofs, V)

r2 = (x[0] - xc) ** 2 + (x[1] - 0.5 * Ly) ** 2
zobs = Lz - indentation + R - ufl.sqrt(ufl.max_value(R**2 - r2, 0.0))
xobs = ufl.as_vector((x[0], x[1], zobs))

g0 = ufl.dot(xobs - x, n)
g = g0 - ufl.dot(u, n)
penetration = ufl.conditional(
    ufl.lt(r2, R**2),
    ufl.conditional(ufl.lt(g, 0.0), -g, 0.0),
    0.0,
)
pn = eps_n * penetration
tc = -pn * n

F = (
    ufl.inner(sigma(u), epsilon(v)) * dx
    - ufl.dot(f, v) * dx
    - ufl.dot(t0, v) * ds(Gamma_t_id)
    - ufl.dot(tc, v) * ds_c
)
J = ufl.derivative(F, u, du)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_atol": 1.0e-6,
    "snes_rtol": 1.0e-8,
    "snes_stol": 0.0,
    "snes_max_it": 100,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "ksp_error_if_not_converged": True,
    "pc_type": "cholesky",
    "pc_factor_mat_solver_type": "mumps",
}

setup_start = perf_counter()
penalty_problem = NonlinearProblem(
    F,
    u,
    bcs=[bc],
    J=J,
    petsc_options=petsc_options,
    petsc_options_prefix="penalty_contact_",
)
penalty_problem.A.setOption(PETSc.Mat.Option.SPD, True)

active = ufl.conditional(ufl.lt(0.0, penetration), 1.0, 0.0)
force_form = form(pn * ds_c)
tc_form = form(ufl.dot(tc, v) * ds_c)
nodal_area_form = form(ufl.dot(ufl.as_vector((1.0,) * gdim), v) * ds_c)
nodal_area = np.zeros_like(tc_out.x.array)
assemble_vector(nodal_area, nodal_area_form)
surface_dofs = nodal_area > 0.0
area_form = form(active * ds_c)
penetration_squared_form = form(penetration**2 * ds_c)
t_setup = perf_counter() - setup_start

output_dir.mkdir(parents=True, exist_ok=True)
history = []
t_pen = 0.0
solver = penalty_problem.solver

print(f"Displacement DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
print(f"Setup, including form compilation: {t_setup:.3f} s")
print(
    "step   x_indenter   iterations    residual       force        area      rms_pen      solve_s"
)

with VTKFile(comm, str(output_dir / "contact.pvd"), "w") as vtk:
    for step, x_step in enumerate(np.linspace(0.0, Lx, nsteps + 1)):
        xc.value = PETSc.ScalarType(x_step)

        step_start = perf_counter()
        penalty_problem.solve()
        t_step = perf_counter() - step_start
        t_pen += t_step
        u.x.scatter_forward()

        tc_out.x.array[:] = 0.0
        assemble_vector(tc_out.x.array, tc_form)
        # Mass-lumped boundary projection: nodal force / nodal area.
        np.divide(
            tc_out.x.array,
            nodal_area,
            out=tc_out.x.array,
            where=surface_dofs,
        )
        tc_out.x.scatter_forward()

        reason = int(solver.getConvergedReason())
        niter = solver.getIterationNumber()
        residual = solver.getFunctionNorm()
        force = float(assemble_scalar(force_form))
        area = float(assemble_scalar(area_form))
        penetration_squared = float(assemble_scalar(penetration_squared_form))
        rms_penetration = np.sqrt(penetration_squared / area) if area > 0.0 else 0.0

        history.append(
            (
                step,
                x_step,
                niter,
                reason,
                residual,
                force,
                area,
                rms_penetration,
                t_step,
            )
        )
        print(
            f"{step:4d}  {x_step:11.4e}  {niter:10d}  {residual:10.3e}  "
            f"{force:10.3e}  {area:10.3e}  {rms_penetration:10.3e}  {t_step:9.3f}"
        )
        vtk.write_function([u, tc_out], float(x_step))

# np.savetxt(
#     output_dir / "history.csv",
#     np.asarray(history),
#     delimiter=",",
#     header="step,x_indenter,iterations,reason,residual,normal_force,contact_area,rms_penetration,solve_s",
#     comments="",
# )
print(f"Total solve time: {t_pen:.3f} s")
print(f"Results: {output_dir.resolve()}")

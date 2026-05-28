import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.mesh import (
    create_box,
    locate_entities_boundary,
    meshtags,
    entities_to_geometry,
    CellType,
)
from dolfinx.io import VTKFile
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    Expression,
)
from dolfinx.fem.petsc import NonlinearProblem


comm = MPI.COMM_WORLD

if comm.size != 1:
    raise RuntimeError("This script is written for serial execution.")


Lx = 1.0
Ly = 1.0
H = 1.0

nx = 20
ny = 20
nz = 20

dirichlet_id = 1
neumann_id = 2
contact_id = 3

E = 1.0e9
nu = 0.3

kpen = 1.0e12

R = 0.35
delta_final = 0.02
nsteps = 10


mesh = create_box(
    comm,
    [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([Lx, Ly, H], dtype=np.float64),
    ],
    [nx, ny, nz],
    cell_type=CellType.tetrahedron,
)

tdim = mesh.topology.dim
fdim = tdim - 1
gdim = mesh.geometry.dim

mesh.topology.create_connectivity(fdim, 0)

facets_bottom = locate_entities_boundary(
    mesh,
    fdim,
    lambda x: np.isclose(x[2], 0.0),
)

facets_top = locate_entities_boundary(
    mesh,
    fdim,
    lambda x: np.isclose(x[2], H),
)

facets_lateral = locate_entities_boundary(
    mesh,
    fdim,
    lambda x: np.logical_or.reduce(
        (
            np.isclose(x[0], 0.0),
            np.isclose(x[0], Lx),
            np.isclose(x[1], 0.0),
            np.isclose(x[1], Ly),
        )
    ),
)

facet_indices = np.hstack(
    [
        facets_bottom,
        facets_lateral,
        facets_top,
    ]
).astype(np.int32)

facet_values = np.hstack(
    [
        np.full(len(facets_bottom), dirichlet_id, dtype=np.int32),
        np.full(len(facets_lateral), neumann_id, dtype=np.int32),
        np.full(len(facets_top), contact_id, dtype=np.int32),
    ]
).astype(np.int32)

perm = np.argsort(facet_indices)

facet_tags = meshtags(
    mesh,
    fdim,
    facet_indices[perm],
    facet_values[perm],
)

facets_c = facet_tags.indices[facet_tags.values == contact_id]

if len(facets_c) == 0:
    raise RuntimeError("No contact facets found.")

gdofs_c = entities_to_geometry(mesh, fdim, facets_c, False)
nodes_c = np.unique(gdofs_c.reshape(-1))
Xc = mesh.geometry.x[nodes_c, :gdim]

contact_center = np.array(
    [
        0.5 * Lx,
        0.5 * Ly,
    ],
    dtype=np.float64,
)

print(f"Contact center: {contact_center}")
print(f"Contact height H: {H}")
print(f"Sphere radius R: {R}")
print(f"Final indentation delta: {delta_final}")


V = functionspace(mesh, ("Lagrange", 1, (gdim,)))

u = Function(V)
u.name = "u"

v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)

mu = Constant(mesh, PETSc.ScalarType(E / (2.0 * (1.0 + nu))))
lmbda = Constant(mesh, PETSc.ScalarType(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))
k = Constant(mesh, PETSc.ScalarType(kpen))

f = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
t = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

xc0 = Constant(mesh, PETSc.ScalarType(contact_center[0]))
yc0 = Constant(mesh, PETSc.ScalarType(contact_center[1]))
Rc = Constant(mesh, PETSc.ScalarType(R))
z0 = Constant(mesh, PETSc.ScalarType(H))


def eps(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)


ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

r2 = (x[0] - xc0) ** 2 + (x[1] - yc0) ** 2

zobs = z0 + Rc - ufl.sqrt(ufl.max_value(Rc**2 - r2, 0.0))

gap = zobs - (x[2] + u[2])

pn = k * ufl.conditional(ufl.lt(gap, 0.0), -gap, 0.0)

F = (
    ufl.inner(sigma(u), eps(v)) * ufl.dx
    - ufl.dot(f, v) * ufl.dx
    - ufl.dot(t, v) * ds(neumann_id)
    + pn * v[2] * ds(contact_id)
)

J = ufl.derivative(F, u)

facets_D = facet_tags.indices[facet_tags.values == dirichlet_id]

if len(facets_D) == 0:
    raise RuntimeError("No Dirichlet facets found.")

dofs_D = locate_dofs_topological(V, fdim, facets_D)

uD = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
bc = dirichletbc(uD, dofs_D, V)

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_max_it": 40,
    "snes_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
}

problem = NonlinearProblem(
    F,
    u,
    bcs=[bc],
    J=J,
    petsc_options=petsc_options,
    petsc_options_prefix="penalty_contact_",
)


for i in range(1, nsteps + 1):
    delta = delta_final * i / nsteps
    z0.value = PETSc.ScalarType(H - delta)

    print()
    print(f"Load step {i}/{nsteps}, delta = {delta}")

    problem.solve()
    u.x.scatter_forward()

    reason = problem.solver.getConvergedReason()
    niter = problem.solver.getIterationNumber()

    print(f"SNES iterations: {niter}")
    print(f"SNES converged reason: {reason}")

    if reason <= 0:
        raise RuntimeError(f"SNES did not converge at load step {i}. Reason: {reason}")


def interpolation_points(V):
    pts = V.element.interpolation_points
    if callable(pts):
        return pts()
    return pts


W = functionspace(mesh, ("DG", 0))

gap_fun = Function(W)
gap_fun.name = "gap"

pn_fun = Function(W)
pn_fun.name = "penalty_pressure"

zobs_fun = Function(W)
zobs_fun.name = "z_obstacle"

pts = interpolation_points(W)

gap_expr = Expression(gap, pts)
pn_expr = Expression(pn, pts)
zobs_expr = Expression(zobs, pts)

gap_fun.interpolate(gap_expr)
pn_fun.interpolate(pn_expr)
zobs_fun.interpolate(zobs_expr)

with VTKFile(comm, "u_penalty_rigid_contact.pvd", "w") as vtk:
    vtk.write_function(u)

with VTKFile(comm, "gap_penalty_rigid_contact.pvd", "w") as vtk:
    vtk.write_function(gap_fun)

with VTKFile(comm, "pressure_penalty_rigid_contact.pvd", "w") as vtk:
    vtk.write_function(pn_fun)

with VTKFile(comm, "obstacle_height.pvd", "w") as vtk:
    vtk.write_function(zobs_fun)

"""
Code that constructs and saves a BEM matrix from 3D FEM simulations using FEniCSx.

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL, Centre des Matériaux)
Date: May 2024
License: BSD 3-Clause
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from numba import jit, prange

# Fenicsx libraries
import ufl
from dolfinx import default_scalar_type, mesh
from dolfinx.fem import (
    Constant,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    locate_entities,
    meshtags,
)
from ufl import dx, ds, grad, inner, sym, Measure

dtype = PETSc.ScalarType


# Create a mesh of a cube with dimensions [0, 1] x [0, 1] x [0, 1] and 15 x 15 x 5 elements
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [15, 15, 5],
    CellType.hexahedron,
    # CellType.tetrahedron,
    ghost_mode=GhostMode.shared_facet,
)

# omega, rho = 300.0, 10.0
# x = ufl.SpatialCoordinate(msh)
# f = ufl.as_vector((rho * omega**2 * x[0], rho * omega**2 * x[1], 0.0))

# Define the elasticity parameters and create a function that computes
# an expression for the stress given a displacement field.

E = 1.0e9
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
Es = E / (1 - nu**2)
Lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def epsilon(u):
    return sym(grad(u))


def sigma(v):
    return 2.0 * mu * epsilon(v) + Lambda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))


V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

# Node selector to apply Dirichlet boundary conditions
selector_tol = 0.1


def moving_nodes(x):
    result = (
        np.isclose(x[0], 0.3, atol=selector_tol)
        & np.isclose(x[1], 0.3, atol=selector_tol)
        & np.isclose(x[2], 1.0, atol=selector_tol)
    )
    return result


fdim = msh.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(msh, fdim, moving_nodes)

# Apply constant displacement along z on the nodes selected by moving_nodes marker
u_D = np.array([0, 0, -0.01], dtype=default_scalar_type)
bc1 = dirichletbc(u_D, locate_dofs_topological(V, fdim, boundary_facets), V)

facets = locate_entities_boundary(msh, dim=2, marker=lambda x: np.isclose(x[2], 0.0))
bc2 = dirichletbc(
    np.zeros(3, dtype=dtype),
    locate_dofs_topological(V, entity_dim=2, entities=facets),
    V=V,
)

# Try to impose Neumann boundary condition
selector_tol = 0.06
boundaries = [
    (
        1,
        lambda x: np.isclose(x[0], 0.75, atol=selector_tol)
        & np.isclose(x[1], 0.75, atol=selector_tol)
        & np.isclose(x[2], 1.0, atol=selector_tol),
    )
]

facet_indices, facet_markers = [], []
fdim = msh.topology.dim - 1
for marker, locator in boundaries:
    facets = locate_entities(msh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(
    msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)

msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
with XDMFFile(msh.comm, "out_elasticity/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_meshtags(facet_tag, msh.geometry)

ds = Measure("ds", domain=msh, subdomain_data=facet_tag)

T = Constant(msh, default_scalar_type((0.0, 0.0, -1.0)))
# Linear form
L = 1e8 * inner(T, v) * ds(1)
# Bilinear form
a = inner(sigma(u), epsilon(v)) * dx

# Solve the proble with Neumann localized BC
problem = LinearProblem(
    a, L, bcs=[bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()

# Actually we could do it differently
# Initiate solver


problem.A.assemble()
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(problem.A)
solver.setType("preonly")
solver.getPC().setType("lu")
solver.setFromOptions()
solver.setUp()  # Do we really need it?

# Initialize solution vector
# Preallocate and set up the right-hand side once
rhs = problem.b.copy()


# Function to update and solve with new RHS values
def update_and_solve(rhs_values, dof_indices):
    # Reset the rhs to zero or some baseline state if needed
    rhs.set(0)
    rhs.setValues(3 * dof_indices[0] + 2, rhs_values[0])
    rhs.assemble()

    # Solve the problem
    solver.solve(rhs, uh)


# @jit(nopython=True, parallel=True)
# def get_deflection(uh, boundary_dofs, force):
#     n = len(boundary_dofs)
#     result = np.zeros(n)
#     for i in prange(n):
#         #   result[i] = 1.
#         result[i] = uh[boundary_dofs[i] * 3 + 2]
#     return result / force


def get_deflection(uh, boundary_dofs, force):
    n = len(boundary_dofs)
    result = np.zeros(n)
    for i in range(n):
        #   result[i] = 1.
        result[i] = uh[boundary_dofs[i] * 3 + 2]
    return result / force


locator = lambda x: np.isclose(x[2], 1.0, atol=1e-5)
boundary_dofs = locate_dofs_geometrical(V, locator)
uh = PETSc.Vec().createMPI(problem.b.getSize(), comm=msh.comm)

K = np.zeros((len(boundary_dofs), len(boundary_dofs)), dtype=default_scalar_type)
force = np.pi * Es
ten_percent = len(boundary_dofs) // 10
for i, u_idf in enumerate(boundary_dofs):
    msg = "DoF index: " + str(i + 1) + "/" + str(len(boundary_dofs))
    rhs_values = force * np.ones(1, dtype=default_scalar_type)  # Example forces
    update_and_solve(rhs_values, [u_idf])
    msg += "... solved."

    K[i, :] = get_deflection(uh.array, boundary_dofs, force)
    if i % ten_percent == 0:
        print("Progress: {0:3.0f}%".format(i / len(boundary_dofs) * 100))


# Extract coordinates of the boundary nodes
boundary_coords = np.zeros((len(boundary_dofs), 3))
for i, u_idf in enumerate(boundary_dofs):
    boundary_coords[i] = V.tabulate_dof_coordinates()[u_idf]

# Save K matrix and nodes
np.savez("out_elasticity/FlexData.npz", K=K, coords=boundary_coords, dofs=boundary_dofs)
print("Successfully saved K matrix to out_elasticity/FlexData.npz")

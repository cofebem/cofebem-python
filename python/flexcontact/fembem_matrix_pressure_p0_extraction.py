"""
Code that constructs and saves a BEM matrix from 3D FEM simulations using FEniCSx.

The matrix is exctracted for the pressure applied at elements and not in nodal-wise manner.

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL, Centre des Matériaux)
Date: Nov 2025
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
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary, locate_entities, meshtags
from ufl import dx, ds, grad, inner, sym, Measure

@jit (nopython=True,parallel=True)
def get_deflection(uh, boundary_dofs, force):
    n = len(boundary_dofs)
    result = np.zeros(n)
    for i in prange(n):
        result[i] = uh[boundary_dofs[i]*3 + 2]
    return result / force   

dtype = PETSc.ScalarType

# Create a mesh of a cube with dimensions [0, 1] x [0, 1] x [0, 1] and 15 x 15 x 5 elements
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [31, 31, 5],
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
Es = E/(1-nu**2)
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
    result = (np.isclose(x[0], 0.3, atol = selector_tol) &
            np.isclose(x[1], 0.3, atol = selector_tol) &
            np.isclose(x[2], 1.0, atol = selector_tol))
    return result

fdim = msh.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(msh, fdim, moving_nodes)

# Apply constant displacement along z on the nodes selected by moving_nodes marker
u_D = np.array([0, 0, -0.01], dtype=default_scalar_type)
bc1 = dirichletbc(u_D, locate_dofs_topological(V, fdim, boundary_facets), V)

facets = locate_entities_boundary(msh, dim=2, marker=lambda x: np.isclose(x[2], 0.0))
bc2 = dirichletbc(np.zeros(3, dtype=dtype), locate_dofs_topological(V, entity_dim=2, entities=facets), V=V)

# Try to impose Neumann boundary condition
selector_tol = 0.06
boundaries = [(1, lambda x: np.isclose(x[0], 0.75, atol = selector_tol) & np.isclose(x[1], 0.75, atol = selector_tol) & np.isclose(x[2], 1.0, atol = selector_tol))]

facet_indices, facet_markers = [], []
fdim = msh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(msh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
with XDMFFile(msh.comm, "out_elasticity/facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_meshtags(facet_tag, msh.geometry)

ds = Measure("ds", domain=msh, subdomain_data=facet_tag)

force = 1.e1

T = Constant(msh, default_scalar_type((0.0, 0.0, -1.0*force)))
# Linear form
L = 1e8 * inner(T, v) * ds(1)
# Bilinear form
a = inner(sigma(u), epsilon(v)) * dx

# Solve the proble with Neumann localized BC
# problem = LinearProblem(a, L, bcs = [bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem = LinearProblem(a, L, bcs = [bc2], petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
uh = problem.solve()

# Loop over all BCs
# Get all top surface facets
locator = lambda x: np.isclose(x[2], 1., atol = 1e-5)
top_facets = locate_entities_boundary(msh, dim=2, marker=locator)

# Get boundary dofs for extraction
boundary_dofs = locate_dofs_geometrical(V, locator)

# Loop over all top facets and solve with Neumann BC on each
K = np.zeros((len(top_facets), len(boundary_dofs)), dtype=default_scalar_type)
force_magnitude = np.pi * Es
ten_percent = max(1, len(top_facets) // 10)

for i, facet_id in enumerate(top_facets):
    # Create meshtags for this single facet
    facet_tag_single = meshtags(msh, fdim, np.array([facet_id], dtype=np.int32), np.array([1], dtype=np.int32))
    
    # Define measure for this facet
    ds_single = Measure("ds", domain=msh, subdomain_data=facet_tag_single)
    
    # Apply force on this facet
    T = Constant(msh, default_scalar_type((0.0, 0.0, -1.0 * force_magnitude)))
    L_single = inner(T, v) * ds_single(1)
    
    # Solve problem with Neumann BC on this facet
    problem_single = LinearProblem(a, L_single, bcs=[bc2], petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
    uh_single = problem_single.solve()
    
    # Extract deflections at boundary dofs
    K[i, :] = get_deflection(uh_single.x.array, boundary_dofs, force_magnitude)
    
    if i % ten_percent == 0:
        print(f"Progress: {i/len(top_facets)*100:3.0f}% (facet {i+1}/{len(top_facets)})")

print(f"Completed: solved for {len(top_facets)} facets")

# Extract facet center coordinates
msh.topology.create_connectivity(fdim, 0)  # facet to vertex connectivity
facet_to_vertex = msh.topology.connectivity(fdim, 0)
facet_centers = np.zeros((len(top_facets), 3))

for i, facet_id in enumerate(top_facets):
    # Get vertices of this facet
    vertices = facet_to_vertex.links(facet_id)
    # Compute center as average of vertex coordinates
    facet_centers[i] = np.mean(msh.geometry.x[vertices], axis=0)

# Extract coordinates of the boundary nodes
boundary_coords = np.zeros((len(boundary_dofs), 3))
for i, dof_id in enumerate(boundary_dofs):
    boundary_coords[i] = V.tabulate_dof_coordinates()[dof_id]

# Save K matrix, facet data, and boundary node data
np.savez("out_elasticity/FlexData.npz", 
         K=K, 
         facet_ids=top_facets,
         facet_centers=facet_centers,
         boundary_dofs=boundary_dofs,
         boundary_coords=boundary_coords)
print("Successfully saved K matrix to out_elasticity/FlexData.npz")
print(f"  - K matrix shape: {K.shape}")
print(f"  - Number of facets: {len(top_facets)}")
print(f"  - Number of boundary DOFs: {len(boundary_dofs)}")

facets = locate_entities_boundary(msh, dim=2, marker=lambda x: np.isclose(x[2], 0.0))

exit(0)






# # Save solution to file
# with XDMFFile(msh.comm, "out_elasticity/solution.xdmf", "w") as xdmf:
#     xdmf.write_mesh(msh)
#     uh.name = "displacement"
#     xdmf.write_function(uh, 0.0)

# # Save deformed mesh by modifying geometry coordinates
# original_coords = msh.geometry.x.copy()
# msh.geometry.x[:] += uh.x.array.reshape((-1, 3))
# with XDMFFile(msh.comm, "out_elasticity/deformed_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(msh)
#     uh.name = "displacement"
#     xdmf.write_function(uh, 0.0)
# # Restore original coordinates
# msh.geometry.x[:] = original_coords

# exit(1)

# # Actually we could do it differently
# # Initiate solver 


# problem.A.assemble()
# solver = PETSc.KSP().create(msh.comm)
# solver.setOperators(problem.A)
# # solver.setType("preonly")
# # solver.getPC().setType("lu")
# solver.setType("cg")
# solver.getPC().setType("hypre")
# solver.setFromOptions()
# solver.setUp() # Do we really need it?

# # Initialize solution vector

# # Preallocate and set up the right-hand side once
# rhs = problem.b.copy()

# # Function to update and solve with new RHS values
# def update_and_solve(rhs_values, dof_indices):
#     # Reset the rhs to zero or some baseline state if needed
#     rhs.set(0)
#     rhs.setValues(3*dof_indices[0]+2, rhs_values[0])
#     rhs.assemble()

#     # Solve the problem
#     solver.solve(rhs, uh)



# locator = lambda x: np.isclose(x[2], 1., atol = 1e-5)
# boundary_dofs = locate_dofs_geometrical(V, locator)
# uh = PETSc.Vec().createMPI(problem.b.getSize(), comm=msh.comm)

# K = np.zeros((len(boundary_dofs), len(boundary_dofs)), dtype=default_scalar_type)
# force = np.pi*Es
# ten_percent = len(boundary_dofs) // 10
# for i, u_idf in enumerate(boundary_dofs):
#     msg = "DoF index: "+str(i+1)+"/"+str(len(boundary_dofs))
#     rhs_values = force * np.ones(1, dtype=default_scalar_type)  # Example forces
#     update_and_solve(rhs_values, [u_idf])
#     msg += "... solved."    
    
#     K[i, :] = get_deflection(uh.array, boundary_dofs, force) 
#     if i % ten_percent == 0:
#         print("Progress: {0:3.0f}%".format(i/len(boundary_dofs)*100))


# # Extract coordinates of the boundary nodes
# boundary_coords = np.zeros((len(boundary_dofs), 3))
# for i, u_idf in enumerate(boundary_dofs):
#     boundary_coords[i] = V.tabulate_dof_coordinates()[u_idf]

# # Save K matrix and nodes
# np.savez("out_elasticity/FlexData.npz", K=K, coords=boundary_coords, dofs=boundary_dofs)
# print("Successfully saved K matrix to out_elasticity/FlexData.npz")


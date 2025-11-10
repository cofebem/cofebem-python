"""
Optimized code that constructs and saves a BEM matrix from 3D FEM simulations using FEniCSx.

Key optimization: Assemble stiffness matrix once, then solve with multiple RHS vectors.

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL, Centre des Matériaux)
Modified for performance optimization
Date: Nov 2025
License: BSD 3-Clause
"""

import time
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from numba import jit, prange
import os
# Fenicsx libraries
import ufl
from dolfinx import default_scalar_type, mesh
from dolfinx import la
from dolfinx.fem import (
    Constant,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
    locate_dofs_geometrical,
    assemble_vector,
    form,
)
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary, locate_entities, meshtags
from ufl import dx, ds, grad, inner, sym, Measure

output_directory = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "out_elasticity")
)
N = 21

@jit(nopython=True, parallel=True)
def get_deflection(uh, boundary_dofs, force):
    n = len(boundary_dofs)
    result = np.zeros(n)
    for i in prange(n):
        result[i] = uh[boundary_dofs[i]*3 + 2]
    return result / force   

dtype = PETSc.ScalarType

# Create a mesh of a cube with dimensions [0, 1] x [0, 1] x [0, 1]
msh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    [N, N, 5],
    CellType.hexahedron,
    ghost_mode=GhostMode.shared_facet,
)

# Define the elasticity parameters
E = 1.0
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
Es = E/(1-nu**2)
Lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(u):
    return sym(grad(u)) 

def sigma(v):
    return 2.0 * mu * epsilon(v) + Lambda * ufl.tr(epsilon(v)) * ufl.Identity(len(v))

V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
Vg = functionspace(msh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
vg = ufl.TestFunction(Vg)


# Apply boundary condition at the bottom
fdim = msh.topology.dim - 1
facets = locate_entities_boundary(msh, dim=2, marker=lambda x: np.isclose(x[2], 0.0))
bc2 = dirichletbc(np.zeros(3, dtype=dtype), locate_dofs_topological(V, entity_dim=2, entities=facets), V=V)

# Bilinear form - assemble ONCE
a = inner(sigma(u), epsilon(v)) * dx

# ============================================================================
# OPTIMIZATION: Assemble stiffness matrix only once
# ============================================================================
print("Assembling stiffness matrix (one-time operation)...")
a_form = form(a)
A = assemble_matrix(a_form, bcs=[bc2])
A.assemble()

# Set up KSP solver for reuse
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("cg")
pc = ksp.getPC()
pc.setType("hypre")
ksp.setFromOptions()
print("Stiffness matrix assembled and solver configured.")

# Get all top surface facets
fdim = msh.topology.dim - 1
locator = lambda x: np.isclose(x[2], 1., atol=1e-5)
top_facets = locate_entities_boundary(msh, dim=fdim, marker=locator)
markers = np.ones(top_facets.shape, dtype=np.int32)
mt = meshtags(msh, fdim, top_facets, markers)
ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

# Get boundary dofs for extraction
boundary_dofs = locate_dofs_geometrical(V, locator)

# Assemble mass matrix
Lm = form(vg * ds(1))
m_vec = assemble_vector(Lm)
m_array = m_vec.array

b_dofs_g = locate_dofs_topological(Vg, fdim, top_facets)
mass_surface = m_array[b_dofs_g]


# ============================================================================
# OPTIMIZATION: Pre-assemble all RHS vectors
# ============================================================================
print(f"Constructing {len(top_facets)} right-hand side vectors...")
force_magnitude = 1.e3

# Create meshtags for efficient facet marking
msh.topology.create_connectivity(fdim, msh.topology.dim)

# Pre-allocate array for all RHS vectors
n_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
all_rhs = []

ten_percent = max(1, len(top_facets) // 10)

start = time.time()
for i, facet_id in enumerate(top_facets):
    # Create meshtags for this single facet
    facet_tag_single = meshtags(msh, fdim, np.array([facet_id], dtype=np.int32), 
                                 np.array([1], dtype=np.int32))
    
    # Define measure for this facet
    ds_single = Measure("ds", domain=msh, subdomain_data=facet_tag_single)
    
    # Apply force on this facet
    T = Constant(msh, default_scalar_type((0.0, 0.0, -1.0 * force_magnitude)))
    L_single = inner(T, v) * ds_single(1)
    
    # Assemble RHS vector
    L_form = form(L_single)
    b = assemble_vector(L_form)
    
    # Scatter reverse to accumulate ghost contributions
    b.scatter_reverse(la.InsertMode.add)
    
    # Apply boundary conditions to the vector's array
    # Signature: set(b, x0=None, alpha=1.0)
    bc2.set(b.array, x0=None, alpha=-1.0)
    
    # Store the underlying PETSc vector
    all_rhs.append(b.petsc_vec.copy())
    
    if i % ten_percent == 0:
        print(f"RHS assembly progress: {i/len(top_facets)*100:3.0f}% (facet {i+1}/{len(top_facets)})")

print(f"--> CPU TIME: RHS assembly: {time.time() - start:.2f} seconds.")
print(f"All {len(top_facets)} RHS vectors assembled.")

# ============================================================================
# OPTIMIZATION: Solve all systems at once using dense matrix
# ============================================================================
print(f"Solving all {len(top_facets)} systems simultaneously using batch solve...")

start = time.time()
# Create dense matrix B containing all RHS vectors
n_systems = len(top_facets)
B = PETSc.Mat().create(comm=msh.comm)
B.setSizes([n_dofs, n_systems])
B.setType('dense')
B.setUp()

# Fill matrix B with all RHS vectors
print("Assembling RHS matrix B...")
for i, b in enumerate(all_rhs):
    b_array = b.getArray()
    for j in range(len(b_array)):
        if abs(b_array[j]) > 1e-14:  # Only set non-zero values for efficiency
            B.setValue(j, i, b_array[j])

B.assemblyBegin()
B.assemblyEnd()
print(f"RHS matrix B assembled: {n_dofs} x {n_systems}")

# Create dense solution matrix X
X = PETSc.Mat().create(comm=msh.comm)
X.setSizes([n_dofs, n_systems])
X.setType('dense')
X.setUp()
print(f"--> CPU TIME: Matrix assembly: {time.time() - start:.2f} seconds.")

# Solve AX = B using matrix-matrix solve
start = time.time()
print("Solving AX = B...")
ksp.setOperators(A)
ksp.matSolve(B, X)
print(f"Batch solve completed for all {n_systems} systems!")
print(f"--> CPU TIME: Solving system: {time.time() - start:.2f} seconds.")

# Extract results into K matrix
print("Extracting deflections...")
K = np.zeros((len(top_facets), len(boundary_dofs)), dtype=default_scalar_type)

for i in range(n_systems):
    # Get column i from solution matrix X
    x_col = X.getDenseArray()[:, i]
    K[i, :] = get_deflection(x_col, boundary_dofs, force_magnitude)
    
    if i % ten_percent == 0:
        print(f"Extraction progress: {i/n_systems*100:3.0f}% (system {i+1}/{n_systems})")

print(f"Completed: extracted results for {n_systems} systems")

# Extract facet center coordinates
msh.topology.create_connectivity(fdim, 0)
facet_to_vertex = msh.topology.connectivity(fdim, 0)
facet_centers = np.zeros((len(top_facets), 3))

for i, facet_id in enumerate(top_facets):
    vertices = facet_to_vertex.links(facet_id)
    facet_centers[i] = np.mean(msh.geometry.x[vertices], axis=0)

# Extract coordinates of the boundary nodes
boundary_coords = np.zeros((len(boundary_dofs), 3))
for i, dof_id in enumerate(boundary_dofs):
    boundary_coords[i] = V.tabulate_dof_coordinates()[dof_id]

# Save K matrix, facet data, and boundary node data
filename = "FlexData_{0}x{0}.npz".format(N)
np.savez(os.path.join(output_directory, filename), 
         K=K.T,
         M=mass_surface,
         facet_ids=top_facets,
         facet_centers=facet_centers,
         boundary_dofs=boundary_dofs,
         boundary_coords=boundary_coords)
print("Successfully saved K matrix to {0}".format(os.path.join(output_directory, filename)))
print(f"  - K matrix shape: {K.shape}")
print(f"  - M matrix shape: {mass_surface.shape}")
print(f"  - Number of facets: {len(top_facets)}")
print(f"  - Number of boundary DOFs: {len(boundary_dofs)}")
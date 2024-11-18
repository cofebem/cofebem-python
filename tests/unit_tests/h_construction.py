from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

# FEniCSx libraries
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    dirichletbc,
    functionspace,
    locate_dofs_geometrical,
)
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, sym

dtype = PETSc.ScalarType

class BEMatrixAssembler:
    def __init__(self, mesh, E=1e9, nu=0.3):
        self.mesh = mesh
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        # Define the vector function space
        self.V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.bcs = []
        self.define_variational_forms()
    
    def epsilon(self, u):
        return sym(grad(u))
    
    def sigma(self, v):
        return 2.0 * self.mu * self.epsilon(v) + self.lmbda * ufl.tr(self.epsilon(v)) * ufl.Identity(len(v))
    
    def define_variational_forms(self):
        self.a = inner(self.sigma(self.u), self.epsilon(self.v)) * dx
        zero_body_force = Constant(self.mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
        self.L = inner(zero_body_force, self.v) * dx  # Zero body force
    
    def apply_dirichlet_bc(self, locator, value):
        dofs = locate_dofs_geometrical(self.V, locator)
        bc = dirichletbc(value, dofs, self.V)
        self.bcs.append(bc)
    
    def assemble_system(self):
        self.problem = LinearProblem(
            self.a,
            self.L,
            bcs=self.bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        self.A = self.problem.A
        self.A.assemble()
        self.solver = PETSc.KSP().create(self.mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType("preonly")
        self.solver.getPC().setType("lu")
        self.solver.setFromOptions()
        self.solver.setUp()
        self.uh = self.problem.u.vector.copy()
        self.rhs = self.problem.b.copy()
    
    def update_and_solve(self, force_value, dof_index):
        self.rhs.zeroEntries()
        self.rhs.setValue(dof_index, force_value)
        self.rhs.assemble()
        self.solver.solve(self.rhs, self.uh)
    
    def get_deflection(self, boundary_dof_indices):
        uh_array = self.uh.array
        displacement_z = uh_array[boundary_dof_indices]
        return displacement_z
    
    def assemble_BEM_matrix(self, boundary_dofs, force_value):
        num_dofs = len(boundary_dofs)
        K = np.zeros((num_dofs, num_dofs), dtype=default_scalar_type)
        
        # Get global dof indices for the z-component
        dofs_z = self.V.sub(2).dofmap.list.array[boundary_dofs]
        
        for i, node in enumerate(boundary_dofs):
            # Global dof index for z-component at node
            dof_index = self.V.sub(2).dofmap.list.array[node]
            self.update_and_solve(force_value, dof_index)
            deflection = self.get_deflection(dofs_z) / force_value
            K[i, :] = deflection
            if i % max(num_dofs // 10, 1) == 0:
                print(f"Progress: {100 * i / num_dofs:.1f}%")
        return K
    
    def save_results(self, K, boundary_coords, filename="FlexData.npz"):
        np.savez(filename, K=K, coords=boundary_coords)
        print(f"Successfully saved K matrix to {filename}")

if __name__ == "__main__":
    # Example usage:

    from dolfinx.mesh import create_unit_cube, CellType

    # Create an arbitrary mesh (e.g., a unit cube)
    mesh = create_unit_cube(MPI.COMM_WORLD, 15, 15, 5, cell_type=CellType.hexahedron)

    # Instantiate the assembler with the mesh and material properties
    assembler = BEMatrixAssembler(mesh, E=1e9, nu=0.3)

    # Apply Dirichlet boundary conditions (e.g., fix the bottom face at z=0)
    def bottom_face(x):
        return np.isclose(x[2], 0.0)

    assembler.apply_dirichlet_bc(locator=bottom_face, value=PETSc.ScalarType((0.0, 0.0, 0.0)))

    # Assemble the system
    assembler.assemble_system()

    # Locate boundary degrees of freedom on the top face (e.g., z=1.0)
    def top_face(x):
        return np.isclose(x[2], 1.0)

    # Get the boundary dofs for the z-component
    boundary_dofs = locate_dofs_geometrical(assembler.V.sub(2), top_face)
    boundary_coords = assembler.V.tabulate_dof_coordinates()[boundary_dofs]

    # Assemble the BEM matrix using a specified force value
    force_value = 1.0  # Apply a unit force
    K = assembler.assemble_BEM_matrix(boundary_dofs, force_value)

    # Save the resulting stiffness matrix and boundary coordinates
    assembler.save_results(K, boundary_coords, filename="FlexData.npz")

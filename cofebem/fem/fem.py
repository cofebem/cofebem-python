from typing import Any, Dict, Optional, Union, Callable, List
import numpy as np
import sympy as sp
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg, bicgstab, gmres
from ..mesh import Mesh
from .function_space import FunctionSpace
from .function import TestFunction, TrialFunction
from .forms import LinearForm, BilinearForm
from .function import Function


class FEM:
    """
    FEM class encapsulates the assembly and solution of the finite element problem.

    Attributes:
        mesh (Mesh): The mesh representing the computational domain.
        function_space (FunctionSpace): The function space for trial and test functions.
        trial_function (TrialFunction): The trial function 'u'.
        test_function (TestFunction): The test function 'v'.
        bilinear_form (BilinearForm): The bilinear form 'a(u, v)' of the problem.
        linear_form (LinearForm): The linear form 'L(v)' of the problem.
        boundary_conditions (Dict[str, Any]): Dictionary of boundary conditions.
        stiffness_matrix (csr_matrix): The assembled global stiffness matrix.
        load_vector (np.ndarray): The assembled global load vector.
        solution (Function): The solution function 'u'.
        num_dofs (int): Total number of degrees of freedom.
        solver_options (Dict[str, Any]): Options for the solver.
        problem_type (str): Type of the problem ('static', 'transient', 'nonlinear').
        verbose (bool): Flag for verbose output.
    """

    def __init__(
        self,
        mesh: Mesh,
        function_space: FunctionSpace,
        bilinear_form: BilinearForm,
        linear_form: LinearForm,
        boundary_conditions: Optional[Dict[str, Any]] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        problem_type: str = "static",
    ) -> None:
        """
        Initialize the FEM class.

        Parameters:
            mesh (Mesh): The mesh representing the computational domain.
            function_space (FunctionSpace): The function space for trial and test functions.
            bilinear_form (BilinearForm): The bilinear form 'a(u, v)'.
            linear_form (LinearForm): The linear form 'L(v)'.
            boundary_conditions (Optional[Dict[str, Any]]): Dictionary of boundary conditions.
            solver_options (Optional[Dict[str, Any]]): Options for the solver.
            problem_type (str): Type of the problem ('static', 'transient', 'nonlinear').
        """
        self.mesh: Mesh = mesh
        self.function_space: FunctionSpace = function_space
        self.trial_function: TrialFunction = bilinear_form.trial_function
        self.test_function: TestFunction = bilinear_form.test_function
        self.bilinear_form: BilinearForm = bilinear_form
        self.linear_form: LinearForm = linear_form
        self.boundary_conditions: Dict[str, Any] = boundary_conditions or {}
        self.problem_type: str = problem_type

        # Solver options
        self.solver_options: Dict[str, Any] = solver_options or {}
        self.solver_type: str = self.solver_options.get("solver_type", "direct")
        self.linear_solver_method: str = self.solver_options.get(
            "linear_solver", "spsolve"
        )
        self.tolerance: float = self.solver_options.get("tolerance", 1e-8)
        self.max_iterations: int = self.solver_options.get("max_iterations", 1000)
        self.verbose: bool = self.solver_options.get("verbose", False)

        # Initialize attributes
        self.stiffness_matrix: Optional[csr_matrix] = None
        self.load_vector: Optional[np.ndarray] = None
        self.solution: Optional[Function] = None
        self.num_dofs: int = self.function_space.num_dofs

    def assemble_system(self) -> None:
        """
        Assemble the global stiffness matrix and load vector.
        """
        if self.verbose:
            print("Assembling the system...")

        # Initialize global stiffness matrix and load vector
        self.stiffness_matrix: lil_matrix = lil_matrix((self.num_dofs, self.num_dofs))
        self.load_vector: np.ndarray = np.zeros(self.num_dofs)

        # Get quadrature points and weights
        quadrature = self.function_space.quadrature
        dof_map = self.function_space.dof_connectivity

        # Assemble element contributions
        for element_index in range(len(dof_map)):
            element_dofs = dof_map[element_index]
            element_coords = self.get_element_coordinates(element_index)

            # Assemble element matrix and vector
            Ke = self.assemble_element_matrix(element_coords, quadrature)
            Fe = self.assemble_element_vector(element_coords, quadrature)

            # Assemble into global system
            for i_local, i_global in enumerate(element_dofs):
                self.load_vector[i_global] += Fe[i_local]
                for j_local, j_global in enumerate(element_dofs):
                    self.stiffness_matrix[i_global, j_global] += Ke[i_local, j_local]

        # Convert stiffness matrix to CSR format
        self.stiffness_matrix = self.stiffness_matrix.tocsr()

        if self.verbose:
            print("System assembly complete.")

    def assemble_element_matrix(
        self, element_coords: np.ndarray, quadrature: Any
    ) -> np.ndarray:
        """
        Assemble the local element stiffness matrix.

        Parameters:
            element_coords (np.ndarray): Coordinates of the element nodes.
            quadrature (QuadratureRule): Quadrature rule for numerical integration.

        Returns:
            Ke (np.ndarray): Local element stiffness matrix.
        """
        num_local_dofs: int = len(element_coords)
        Ke: np.ndarray = np.zeros((num_local_dofs, num_local_dofs))

        for q_point, weight in zip(quadrature.points, quadrature.weights):
            # Evaluate basis functions and derivatives
            N = self.function_space.evaluate_basis_functions(q_point)
            dN_dxi = self.function_space.evaluate_basis_derivatives(q_point)

            # Compute Jacobian and determinant
            J = dN_dxi @ element_coords
            det_J = np.linalg.det(J)
            if det_J <= 0:
                raise ValueError("Jacobian determinant is non-positive.")
            inv_J = np.linalg.inv(J)
            dN_dx = inv_J.T @ dN_dxi

            # Evaluate bilinear form integrand
            integrand = self.evaluate_bilinear_form(N, dN_dx, element_coords, q_point)

            # Accumulate into local stiffness matrix
            Ke += integrand * weight * det_J

        return Ke

    def assemble_element_vector(
        self, element_coords: np.ndarray, quadrature: Any
    ) -> np.ndarray:
        """
        Assemble the local element load vector.

        Parameters:
            element_coords (np.ndarray): Coordinates of the element nodes.
            quadrature (QuadratureRule): Quadrature rule for numerical integration.

        Returns:
            Fe (np.ndarray): Local element load vector.
        """
        num_local_dofs: int = len(element_coords)
        Fe: np.ndarray = np.zeros(num_local_dofs)

        for q_point, weight in zip(quadrature.points, quadrature.weights):
            # Evaluate basis functions
            N = self.function_space.evaluate_basis_functions(q_point)

            # Compute Jacobian and determinant
            dN_dxi = self.function_space.evaluate_basis_derivatives(q_point)
            J = dN_dxi @ element_coords
            det_J = np.linalg.det(J)
            if det_J <= 0:
                raise ValueError("Jacobian determinant is non-positive.")

            # Evaluate linear form integrand
            integrand = self.evaluate_linear_form(N, element_coords, q_point)

            # Accumulate into local load vector
            Fe += integrand * weight * det_J

        return Fe

    def evaluate_bilinear_form(
        self,
        N: np.ndarray,
        dN_dx: np.ndarray,
        element_coords: np.ndarray,
        local_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the bilinear form integrand at a quadrature point.

        Parameters:
            N (np.ndarray): Basis function values at the quadrature point.
            dN_dx (np.ndarray): Basis function derivatives at the quadrature point.
            element_coords (np.ndarray): Coordinates of the element nodes.
            local_coords (np.ndarray): Local coordinates of the quadrature point.

        Returns:
            integrand (np.ndarray): Evaluated integrand for the bilinear form.
        """
        num_local_dofs: int = len(N)
        integrand: np.ndarray = np.zeros((num_local_dofs, num_local_dofs))

        # Map local coordinates to global coordinates
        global_coords = self.function_space.map_local_to_global(
            element_coords, local_coords
        )

        # Prepare substitutions
        substitutions: Dict[str, Any] = {"x": global_coords[0]}
        if self.function_space.mesh.dimension > 1:
            substitutions["y"] = global_coords[1]
        if self.function_space.mesh.dimension > 2:
            substitutions["z"] = global_coords[2]

        # Evaluate coefficients
        coefficients: Dict[str, Any] = {
            k: self.evaluate_coefficient(v, substitutions)
            for k, v in self.bilinear_form.coefficients.items()
        }

        # Compute the integrand
        for i in range(num_local_dofs):
            for j in range(num_local_dofs):
                # Build expressions for u and v
                u = N[j]
                grad_u = dN_dx[:, j]
                v = N[i]
                grad_v = dN_dx[:, i]

                # Prepare further substitutions
                local_subs: Dict[Any, Any] = {
                    self.bilinear_form.u: u,
                    self.bilinear_form.v: v,
                    "grad(u)": sp.Matrix(grad_u),
                    "grad(v)": sp.Matrix(grad_v),
                }
                local_subs.update(coefficients)
                local_subs.update(substitutions)

                # Substitute and evaluate the expression
                expr = self.bilinear_form.expression.subs(local_subs)
                integrand[i, j] += float(expr.evalf())

        return integrand

    def evaluate_linear_form(
        self,
        N: np.ndarray,
        element_coords: np.ndarray,
        local_coords: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the linear form integrand at a quadrature point.

        Parameters:
            N (np.ndarray): Basis function values at the quadrature point.
            element_coords (np.ndarray): Coordinates of the element nodes.
            local_coords (np.ndarray): Local coordinates of the quadrature point.

        Returns:
            integrand (np.ndarray): Evaluated integrand for the linear form.
        """
        num_local_dofs: int = len(N)
        integrand: np.ndarray = np.zeros(num_local_dofs)

        # Map local coordinates to global coordinates
        global_coords = self.function_space.map_local_to_global(
            element_coords, local_coords
        )

        # Prepare substitutions
        substitutions: Dict[str, Any] = {"x": global_coords[0]}
        if self.function_space.mesh.dimension > 1:
            substitutions["y"] = global_coords[1]
        if self.function_space.mesh.dimension > 2:
            substitutions["z"] = global_coords[2]

        # Evaluate coefficients
        coefficients: Dict[str, Any] = {
            k: self.evaluate_coefficient(v, substitutions)
            for k, v in self.linear_form.coefficients.items()
        }

        # Compute the integrand
        for i in range(num_local_dofs):
            v = N[i]
            local_subs: Dict[Any, Any] = {
                self.linear_form.v: v,
            }
            local_subs.update(coefficients)
            local_subs.update(substitutions)

            # Substitute and evaluate the expression
            expr = self.linear_form.expression.subs(local_subs)
            integrand[i] += float(expr.evalf())

        return integrand

    def evaluate_coefficient(
        self, coefficient: Any, substitutions: Dict[str, Any]
    ) -> float:
        """
        Evaluate a coefficient at given substitutions.

        Parameters:
            coefficient (Any): The coefficient to evaluate.
            substitutions (Dict[str, Any]): Substitutions for the variables.

        Returns:
            value (float): Evaluated coefficient.
        """
        if isinstance(coefficient, sp.Expr):
            return float(coefficient.subs(substitutions).evalf())
        elif callable(coefficient):
            return float(coefficient(**substitutions))
        else:
            return float(coefficient)

    def get_element_coordinates(self, element_index: int) -> np.ndarray:
        """
        Get the coordinates of the nodes of an element.

        Parameters:
            element_index (int): Index of the element.

        Returns:
            element_coords (np.ndarray): Coordinates of the element nodes.
        """
        element_type = self.mesh.element_types[0]
        element_nodes = self.mesh.elements[element_type][element_index]
        element_coords = self.mesh.nodes[element_nodes]
        return element_coords

    def apply_boundary_conditions(self) -> None:
        """
        Apply boundary conditions to the global stiffness matrix and load vector.
        """
        if "Dirichlet" in self.boundary_conditions:
            self.apply_dirichlet_boundary_conditions()

        if "Neumann" in self.boundary_conditions:
            self.apply_neumann_boundary_conditions()

    def apply_dirichlet_boundary_conditions(self) -> None:
        """
        Apply Dirichlet boundary conditions.
        """
        dirichlet_bcs = self.boundary_conditions["Dirichlet"]
        nodes = dirichlet_bcs["nodes"]
        values = dirichlet_bcs["values"]

        for node, value in zip(nodes, values):
            # Get DOF indices for the node
            dof = self.function_space.get_node_dof(node)
            if dof is None:
                continue
            # Modify the stiffness matrix and load vector
            self.stiffness_matrix[dof, :] = 0
            self.stiffness_matrix[dof, dof] = 1
            self.load_vector[dof] = value

    def apply_neumann_boundary_conditions(self) -> None:
        """
        Apply Neumann boundary conditions.
        """
        neumann_bcs = self.boundary_conditions["Neumann"]
        # Implementation similar to the assembler; modify load_vector
        # For brevity, this method can be implemented as needed

    def solve(self) -> Function:
        """
        Solve the assembled system.

        Returns:
            solution (Function): The solution function 'u'.
        """
        if self.verbose:
            print("Solving the system...")

        K = self.stiffness_matrix
        f = self.load_vector

        if self.solver_type == "direct":
            if self.linear_solver_method == "spsolve":
                solution_vector = spsolve(K, f)
            else:
                raise ValueError(
                    f"Unknown linear solver method: {self.linear_solver_method}"
                )
        elif self.solver_type == "iterative":
            solution_vector = self.solve_linear_system_iterative(K, f)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        # Store the solution in a Function
        self.solution = Function(self.function_space, solution_vector)

        if self.verbose:
            print("System solved.")

        return self.solution

    def solve_linear_system_iterative(self, K: csr_matrix, f: np.ndarray) -> np.ndarray:
        """
        Solve the linear system using an iterative solver.

        Parameters:
            K (csr_matrix): Global stiffness matrix.
            f (np.ndarray): Global load vector.

        Returns:
            solution (np.ndarray): Solution vector.
        """
        method = self.linear_solver_method
        tol = self.tolerance
        maxiter = self.max_iterations

        if self.verbose:
            print(
                f"Using iterative solver '{method}' with tolerance {tol} and max iterations {maxiter}."
            )

        if method == "cg":
            solution, info = cg(K, f, tol=tol, maxiter=maxiter)
        elif method == "bicgstab":
            solution, info = bicgstab(K, f, tol=tol, maxiter=maxiter)
        elif method == "gmres":
            solution, info = gmres(K, f, tol=tol, maxiter=maxiter)
        else:
            raise ValueError(f"Unknown iterative solver method: {method}")

        if info != 0:
            raise RuntimeError(f"Iterative solver did not converge. Info: {info}")

        return solution

    def run(self) -> Function:
        """
        Run the entire problem setup, assembly, and solution process.

        Returns:
            solution (Function): The solution function 'u'.
        """
        self.assemble_system()
        self.apply_boundary_conditions()
        return self.solve()

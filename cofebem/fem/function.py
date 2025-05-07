import numpy as np
from .function_space import FunctionSpace
from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Union, Any, Callable


class Function:
    """
    Base class for functions in the finite element function space.

    Attributes:
        function_space (FunctionSpace): The function space to which the function belongs.
        coefficients (np.ndarray): Coefficients of the function at DOFs.
        time (float): Current time associated with the function (for time-dependent problems).
        _cache (Dict[Any, Any]): Cache for storing computed values to avoid redundant computations.
    """

    def __init__(
        self,
        function_space: FunctionSpace,
        coefficients: Optional[np.ndarray] = None,
        time: float = 0.0,
    ) -> None:
        """
        Initialize the Function.

        Parameters:
            function_space (FunctionSpace): The function space to which the function belongs.
            coefficients (np.ndarray, optional): Coefficients of the function at DOFs.
            time (float, optional): Current time associated with the function.
        """
        self.function_space: FunctionSpace = function_space
        self.time: float = time
        if coefficients is None:
            self.coefficients: np.ndarray = np.zeros(self.function_space.num_dofs)
        else:
            if len(coefficients) != self.function_space.num_dofs:
                raise ValueError(
                    "Coefficient array size does not match the number of DOFs."
                )
            self.coefficients = coefficients
        self._cache: Dict[Any, Any] = {}

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at given global coordinates.

        Parameters:
            coords (np.ndarray): Array of coordinates where the function is evaluated.

        Returns:
            values (np.ndarray): Function values at the given coordinates.
        """
        # Check if the result is cached
        cache_key = ("evaluate", tuple(coords))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Find the element containing the point
        element_index = self.find_containing_element(coords)
        if element_index is None:
            raise ValueError("Point is outside the mesh.")

        # Get element information
        element_type = self.function_space.mesh.element_types[0]
        element_nodes = self.function_space.mesh.elements[element_type][element_index]
        element_coords = self.function_space.mesh.nodes[element_nodes]
        element_dofs = self.function_space.dof_connectivity[element_index]

        # Map global coordinates to local coordinates
        local_coords = self.map_global_to_local(element_coords, coords)

        # Evaluate basis functions at local coordinates
        N = self.function_space.evaluate_basis_functions(local_coords)

        # Compute the function value
        if self.function_space.num_components == 1:
            values = N @ self.coefficients[element_dofs]
        else:
            values = np.zeros(self.function_space.num_components)
            for comp in range(self.function_space.num_components):
                comp_dofs = element_dofs[comp :: self.function_space.num_components]
                values[comp] = N @ self.coefficients[comp_dofs]

        # Cache the result
        self._cache[cache_key] = values
        return values

    def evaluate_gradient(self, coords: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the function at given global coordinates.

        Parameters:
            coords (np.ndarray): Global coordinates where the gradient is evaluated.

        Returns:
            gradient (np.ndarray): Gradient of the function at the given coordinates.
        """
        # Check if the result is cached
        cache_key = ("evaluate_gradient", tuple(coords))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Find the element containing the point
        element_index = self.find_containing_element(coords)
        if element_index is None:
            raise ValueError("Point is outside the mesh.")

        # Get element information
        element_type = self.function_space.mesh.element_types[0]
        element_nodes = self.function_space.mesh.elements[element_type][element_index]
        element_coords = self.function_space.mesh.nodes[element_nodes]
        element_dofs = self.function_space.dof_connectivity[element_index]

        # Map global coordinates to local coordinates
        local_coords = self.map_global_to_local(element_coords, coords)

        # Evaluate basis function derivatives at local coordinates
        dN_dxi = self.function_space.evaluate_basis_derivatives(local_coords)

        # Compute the Jacobian matrix
        J = dN_dxi @ element_coords

        # Compute the inverse of the Jacobian
        inv_J = np.linalg.inv(J)

        # Transform derivatives to global coordinates
        dN_dx = inv_J.T @ dN_dxi

        # Compute the gradient
        if self.function_space.num_components == 1:
            gradient = dN_dx @ self.coefficients[element_dofs]
        else:
            gradient = np.zeros(
                (self.function_space.mesh.dimension, self.function_space.num_components)
            )
            for comp in range(self.function_space.num_components):
                comp_dofs = element_dofs[comp :: self.function_space.num_components]
                gradient[:, comp] = dN_dx @ self.coefficients[comp_dofs]

        # Cache the result
        self._cache[cache_key] = gradient
        return gradient

    def find_containing_element(self, coords: np.ndarray) -> Optional[int]:
        """
        Find the element that contains the given point.

        Parameters:
            coords (np.ndarray): Coordinates of the point.

        Returns:
            element_index (Optional[int]): Index of the containing element, or None if not found.
        """
        # Implement an efficient search method (e.g., spatial tree) for large meshes
        for element_type in self.function_space.mesh.element_types:
            connectivity = self.function_space.mesh.elements[element_type]
            for idx, element_nodes in enumerate(connectivity):
                element_coords = self.function_space.mesh.nodes[element_nodes]
                if self.is_point_in_element(coords, element_coords):
                    return idx
        return None

    def is_point_in_element(
        self, point: np.ndarray, element_coords: np.ndarray
    ) -> bool:
        """
        Check if a point is inside an element.

        Parameters:
            point (np.ndarray): Coordinates of the point.
            element_coords (np.ndarray): Coordinates of the element nodes.

        Returns:
            is_inside (bool): True if the point is inside the element.
        """
        # Implement for 2D triangular elements
        if self.function_space.mesh.dimension == 2:
            from matplotlib.path import Path

            path = Path(element_coords[:, :2])
            return path.contains_point(point[:2])
        else:
            raise NotImplementedError(
                "Point-in-element check not implemented for this dimension."
            )

    def map_global_to_local(
        self, element_coords: np.ndarray, global_coords: np.ndarray
    ) -> np.ndarray:
        """
        Map global coordinates to local coordinates in the reference element.

        Parameters:
            element_coords (np.ndarray): Coordinates of the element nodes.
            global_coords (np.ndarray): Global coordinates to map.

        Returns:
            local_coords (np.ndarray): Local coordinates in the reference element.
        """
        # Simplified implementation for linear elements
        if self.function_space.mesh.dimension == 2 and self.function_space.degree == 1:
            # Use barycentric coordinates for triangles
            A = np.array(
                [
                    [element_coords[0, 0], element_coords[1, 0], element_coords[2, 0]],
                    [element_coords[0, 1], element_coords[1, 1], element_coords[2, 1]],
                    [1, 1, 1],
                ]
            )
            b = np.array([global_coords[0], global_coords[1], 1])
            xi_eta = np.linalg.solve(A, b)
            return xi_eta[:2]
        else:
            raise NotImplementedError(
                "Global to local mapping not implemented for this element type."
            )

    def set_coefficients(self, coefficients: np.ndarray) -> None:
        """
        Set the coefficients of the function.

        Parameters:
            coefficients (np.ndarray): Array of coefficients at DOFs.
        """
        if len(coefficients) != self.function_space.num_dofs:
            raise ValueError(
                "Coefficient array size does not match the number of DOFs."
            )
        self.coefficients = coefficients
        # Invalidate cache
        self._cache.clear()

    def get_coefficients(self) -> np.ndarray:
        """
        Get the coefficients of the function.

        Returns:
            coefficients (np.ndarray): Array of coefficients at DOFs.
        """
        return self.coefficients

    def copy(self):
        """
        Create a shallow copy of the function.

        Returns:
            Function: A copy of the function.
        """
        return Function(self.function_space, self.coefficients.copy(), self.time)

    def deepcopy(self):
        """
        Create a deep copy of the function.

        Returns:
            Function: A deep copy of the function.
        """
        return deepcopy(self)

    def plot(self) -> None:
        """
        Plot the function over the mesh.

        Note:
            Requires PyVista or Matplotlib for visualization.
        """
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "PyVista is required for plotting. Install it with 'pip install pyvista'."
            )

        # Create a mesh for plotting
        mesh = self.function_space.mesh
        points = mesh.nodes
        element_type = mesh.element_types[0]
        connectivity = mesh.elements[element_type]

        # Prepare cell arrays
        num_cells = connectivity.shape[0]
        num_points_per_cell = connectivity.shape[1]
        cells = np.hstack(
            [np.full((num_cells, 1), num_points_per_cell), connectivity]
        ).flatten()
        cell_types = np.full(num_cells, pv.CellType.TRIANGLE, dtype=np.uint8)

        # Create PyVista UnstructuredGrid
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        # Get function values at nodes
        dof_values = self.get_dof_values()
        grid.point_data["values"] = dof_values

        # Plot the function
        grid.plot(scalars="values", show_edges=True)

    def get_dof_values(self) -> np.ndarray:
        """
        Get the function values at the degrees of freedom.

        Returns:
            dof_values (np.ndarray): Function values at DOFs.
        """
        return self.coefficients

    def __add__(self, other: "Function") -> "Function":
        """
        Add two functions.

        Parameters:
            other (Function): Another function to add.

        Returns:
            Function: The sum of the two functions.
        """
        if self.function_space != other.function_space:
            raise ValueError(
                "Functions must be in the same function space to be added."
            )
        new_coefficients = self.coefficients + other.coefficients
        return Function(self.function_space, new_coefficients, self.time)

    def __sub__(self, other: "Function") -> "Function":
        """
        Subtract another function from this function.

        Parameters:
            other (Function): Another function to subtract.

        Returns:
            Function: The difference of the two functions.
        """
        if self.function_space != other.function_space:
            raise ValueError(
                "Functions must be in the same function space to be subtracted."
            )
        new_coefficients = self.coefficients - other.coefficients
        return Function(self.function_space, new_coefficients, self.time)

    def __mul__(self, scalar: float) -> "Function":
        """
        Multiply the function by a scalar.

        Parameters:
            scalar (float): Scalar value to multiply.

        Returns:
            Function: The scaled function.
        """
        new_coefficients = self.coefficients * scalar
        return Function(self.function_space, new_coefficients, self.time)

    def __truediv__(self, scalar: float) -> "Function":
        """
        Divide the function by a scalar.

        Parameters:
            scalar (float): Scalar value to divide.

        Returns:
            Function: The scaled function.
        """
        new_coefficients = self.coefficients / scalar
        return Function(self.function_space, new_coefficients, self.time)

    def compute_L2_norm(self) -> float:
        """
        Compute the L2 norm of the function over the domain.

        Returns:
            norm (float): The L2 norm of the function.
        """
        norm_squared = 0.0
        for element_index in range(len(self.function_space.dof_connectivity)):
            # Get element information
            element_dofs = self.function_space.dof_connectivity[element_index]
            element_coefficients = self.coefficients[element_dofs]
            element = self.function_space.mesh.elements[
                self.function_space.mesh.element_types[0]
            ][element_index]
            element_coords = self.function_space.mesh.nodes[element]

            # Perform numerical integration over the element
            for q_point, weight in zip(
                self.function_space.quadrature.points,
                self.function_space.quadrature.weights,
            ):
                N = self.function_space.evaluate_basis_functions(q_point)
                value = N @ element_coefficients
                det_J = self.element_jacobian_determinant(element_coords, q_point)
                norm_squared += (value @ value) * weight * det_J
        return np.sqrt(norm_squared)

    def compute_H1_norm(self) -> float:
        """
        Compute the H1 norm of the function over the domain.

        Returns:
            norm (float): The H1 norm of the function.
        """
        norm_squared = 0.0
        for element_index in range(len(self.function_space.dof_connectivity)):
            # Get element information
            element_dofs = self.function_space.dof_connectivity[element_index]
            element_coefficients = self.coefficients[element_dofs]
            element = self.function_space.mesh.elements[
                self.function_space.mesh.element_types[0]
            ][element_index]
            element_coords = self.function_space.mesh.nodes[element]

            # Perform numerical integration over the element
            for q_point, weight in zip(
                self.function_space.quadrature.points,
                self.function_space.quadrature.weights,
            ):
                # Function value contribution
                N = self.function_space.evaluate_basis_functions(q_point)
                value = N @ element_coefficients
                det_J = self.element_jacobian_determinant(element_coords, q_point)
                norm_squared += (value @ value) * weight * det_J

                # Gradient contribution
                dN_dxi = self.function_space.evaluate_basis_derivatives(q_point)
                J = dN_dxi @ element_coords
                inv_J = np.linalg.inv(J)
                dN_dx = inv_J.T @ dN_dxi
                gradient = dN_dx @ element_coefficients
                norm_squared += (gradient @ gradient) * weight * det_J
        return np.sqrt(norm_squared)

    def element_jacobian_determinant(
        self, element_coords: np.ndarray, local_coords: np.ndarray
    ) -> float:
        """
        Compute the determinant of the Jacobian matrix for an element.

        Parameters:
            element_coords (np.ndarray): Coordinates of the element nodes.
            local_coords (np.ndarray): Local coordinates in the reference element.

        Returns:
            det_J (float): Determinant of the Jacobian matrix.
        """
        dN_dxi = self.function_space.evaluate_basis_derivatives(local_coords)
        J = dN_dxi @ element_coords
        return abs(np.linalg.det(J))

    def compute_error(self, exact_solution: Callable[..., float]) -> float:
        """
        Compute the error between the function and an exact solution.

        Parameters:
            exact_solution (Callable): The exact solution function.

        Returns:
            error (float): The L2 norm of the error.
        """
        error_squared = 0.0
        for element_index in range(len(self.function_space.dof_connectivity)):
            # Get element information
            element_dofs = self.function_space.dof_connectivity[element_index]
            element_coefficients = self.coefficients[element_dofs]
            element = self.function_space.mesh.elements[
                self.function_space.mesh.element_types[0]
            ][element_index]
            element_coords = self.function_space.mesh.nodes[element]

            # Perform numerical integration over the element
            for q_point, weight in zip(
                self.function_space.quadrature.points,
                self.function_space.quadrature.weights,
            ):
                N = self.function_space.evaluate_basis_functions(q_point)
                approx_value = N @ element_coefficients
                global_coords = self.function_space.map_local_to_global(
                    element_coords, q_point
                )
                exact_value = exact_solution(*global_coords)
                diff = approx_value - exact_value
                det_J = self.element_jacobian_determinant(element_coords, q_point)
                error_squared += (diff @ diff) * weight * det_J
        return np.sqrt(error_squared)

    def update(self, coefficients: np.ndarray, time: Optional[float] = None) -> None:
        """
        Update the function's coefficients and time.

        Parameters:
            coefficients (np.ndarray): New coefficients.
            time (float, optional): New time value.
        """
        self.set_coefficients(coefficients)
        if time is not None:
            self.time = time

    def save(self, filename: str) -> None:
        """
        Save the function's coefficients to a file.

        Parameters:
            filename (str): Path to the file where coefficients will be saved.
        """
        np.save(filename, self.coefficients)

    def load(self, filename: str) -> None:
        """
        Load function coefficients from a file.

        Parameters:
            filename (str): Path to the file from which coefficients will be loaded.
        """
        coefficients = np.load(filename)
        self.set_coefficients(coefficients)

    def clear_cache(self) -> None:
        """
        Clear the cache of computed values.
        """
        self._cache.clear()


class TestFunction(Function):
    """
    Class representing a test function 'v'.
    """

    def __init__(self, function_space):
        super().__init__(function_space)
        # Additional attributes or methods specific to test functions can be added here


class TrialFunction(Function):
    """
    Class representing a trial function 'u'.
    """

    def __init__(self, function_space):
        super().__init__(function_space)
        # Additional attributes or methods specific to trial functions can be added here

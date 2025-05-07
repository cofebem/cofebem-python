import numpy as np
from scipy.special import comb
from typing import Optional, Dict, List, Tuple, Union, Any, Callable
from cofebem.mesh.mesh import Mesh


class FunctionSpace:
    """
    FunctionSpace class for defining finite element function spaces.

    Attributes:
        mesh (Mesh): The mesh associated with the function space.
        element_type (str): Type of finite element (e.g., 'Lagrange', 'Hermite').
        degree (int): Polynomial degree of the basis functions.
        continuity (str): Continuity of the basis functions (e.g., 'C0').
        num_components (int): Number of components of the field (1 for scalar, >1 for vector fields).
        num_dofs (int): Total number of degrees of freedom.
        dof_indices (Dict[Tuple[int, int], int]): Mapping from mesh entities to DOF indices.
        dof_coordinates (np.ndarray): Coordinates of the DOFs.
        quadrature (QuadratureRule): Quadrature rule for numerical integration.
        reference_element (ReferenceElement): Contains shape functions and derivatives.
        dof_connectivity (List[List[int]]): List mapping elements to DOF indices.
        node_to_dof (Dict[Tuple[int, int], int]): Mapping from node indices to DOF indices.
        dof_to_node (Dict[int, Tuple[int, int]]): Mapping from DOF indices to node indices.
    """

    def __init__(
        self,
        mesh: Mesh,
        element_type: str = "Lagrange",
        degree: int = 1,
        continuity: str = "C0",
        num_components: int = 1,
    ) -> None:
        """
        Initialize the FunctionSpace.

        Parameters:
            mesh (Mesh): The mesh associated with the function space.
            element_type (str, optional):nowolynomial degree of the basis functions (default 1).
            continuity (str, optional): Continuity of the basis functions (default 'C0').
            num_components (int, optional): Number of components of the field (default 1).
        """
        self.mesh: Mesh = mesh
        self.element_type: str = element_type
        self.degree: int = degree
        self.continuity: str = continuity
        self.num_components: int = num_components

        self.num_dofs: int = 0
        self.dof_indices: Dict[Tuple[int, int], int] = {}
        self.dof_coordinates: Optional[np.ndarray] = None
        self.quadrature: Optional["QuadratureRule"] = None
        self.reference_element: Optional["ReferenceElement"] = None
        self.dof_connectivity: List[List[int]] = []
        self.node_to_dof: Dict[Tuple[int, int], int] = {}
        self.dof_to_node: Dict[int, Tuple[int, int]] = {}

        self.setup_function_space()

    def setup_function_space(self) -> None:
        """
        Set up the function space by assigning DOFs and initializing required data structures.
        """
        # Assign DOFs to mesh nodes
        self.assign_dofs()

        # Build DOF connectivity
        self.build_dof_connectivity()

        # Initialize quadrature rule
        self.initialize_quadrature()

        # Initialize reference element
        self.initialize_reference_element()

    def assign_dofs(self) -> None:
        """
        Assign degrees of freedom to mesh nodes.

        Populates:
            self.num_dofs: Total number of DOFs.
            self.node_to_dof: Mapping from node indices to DOF indices.
            self.dof_to_node: Mapping from DOF indices to node indices.
            self.dof_coordinates: Coordinates of the DOFs.
        """
        num_nodes: int = self.mesh.nodes.shape[0]
        self.num_dofs = num_nodes * self.num_components
        self.dof_coordinates = np.zeros((self.num_dofs, self.mesh.dimension))

        dof_counter: int = 0
        for node_index in range(num_nodes):
            node_coords: np.ndarray = self.mesh.nodes[node_index]
            for comp in range(self.num_components):
                dof_index: int = dof_counter
                self.node_to_dof[(node_index, comp)] = dof_index
                self.dof_to_node[dof_index] = (node_index, comp)
                self.dof_coordinates[dof_index] = node_coords
                dof_counter += 1

        if self.num_dofs != dof_counter:
            raise ValueError("Mismatch in the number of DOFs assigned.")

    def build_dof_connectivity(self) -> None:
        """
        Build the DOF connectivity mapping elements to DOF indices.

        Populates:
            self.dof_connectivity: List of DOF indices for each element.
        """
        self.dof_connectivity = []
        element_types: List[str] = self.mesh.element_types

        for element_type in element_types:
            connectivity: np.ndarray = self.mesh.elements[element_type]
            num_elements: int = connectivity.shape[0]
            for elem_index in range(num_elements):
                element_nodes: np.ndarray = connectivity[elem_index]
                element_dofs: List[int] = []
                for node_index in element_nodes:
                    for comp in range(self.num_components):
                        dof_index = self.node_to_dof[(node_index, comp)]
                        element_dofs.append(dof_index)
                self.dof_connectivity.append(element_dofs)

    def initialize_quadrature(self) -> None:
        """
        Initialize the quadrature rule for numerical integration.

        Populates:
            self.quadrature: QuadratureRule object.
        """
        self.quadrature = QuadratureRule(self.mesh.dimension, self.degree + 1)

    def initialize_reference_element(self) -> None:
        """
        Initialize the reference element.

        Populates:
            self.reference_element: ReferenceElement object.
        """
        self.reference_element = ReferenceElement(
            self.mesh.dimension, self.element_type, self.degree
        )

    def evaluate_basis_functions(self, local_coords: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions at given local coordinates.

        Parameters:
            local_coords (np.ndarray): Local coordinates where to evaluate basis functions.

        Returns:
            N (np.ndarray): Values of basis functions at local_coords.
        """
        return self.reference_element.evaluate_shape_functions(local_coords)

    def evaluate_basis_derivatives(self, local_coords: np.ndarray) -> np.ndarray:
        """
        Evaluate basis function derivatives at given local coordinates.

        Parameters:
            local_coords (np.ndarray): Local coordinates where to evaluate basis derivatives.

        Returns:
            dN_dxi (np.ndarray): Derivatives of basis functions at local_coords.
        """
        return self.reference_element.evaluate_shape_function_derivatives(local_coords)

    def map_local_to_global(
        self, element_coords: np.ndarray, local_coords: np.ndarray
    ) -> np.ndarray:
        """
        Map local coordinates to global coordinates.

        Parameters:
            element_coords (np.ndarray): Coordinates of the element's nodes.
            local_coords (np.ndarray): Local coordinates.

        Returns:
            global_coords (np.ndarray): Mapped global coordinates.
        """
        N: np.ndarray = self.evaluate_basis_functions(local_coords)
        global_coords: np.ndarray = N @ element_coords
        return global_coords

    def get_node_dof(self, node_index: int, component: int = 0) -> int:
        """
        Get the DOF index corresponding to a given node and component.

        Parameters:
            node_index (int): Index of the node.
            component (int, optional): Component index (default 0).

        Returns:
            dof_index (int): Corresponding DOF index.
        """
        return self.node_to_dof[(node_index, component)]

    def interpolate(
        self, function_values: Union[Callable[[np.ndarray], float], np.ndarray]
    ) -> np.ndarray:
        """
        Interpolate function values into the function space.

        Parameters:
            function_values (Callable or np.ndarray): Function to interpolate or array of values at nodes.

        Returns:
            coefficients (np.ndarray): Coefficients of the interpolated function in the function space.
        """
        if callable(function_values):
            # Evaluate function at DOF coordinates
            values = np.array(
                [function_values(coord) for coord in self.dof_coordinates]
            )
        elif isinstance(function_values, np.ndarray):
            if function_values.shape[0] != self.num_dofs:
                raise ValueError("Input array size does not match the number of DOFs.")
            values = function_values
        else:
            raise TypeError("function_values must be a callable or an ndarray.")

        return values


class QuadratureRule:
    """
    QuadratureRule class for numerical integration over reference elements.

    Attributes:
        dimension (int): Spatial dimension.
        order (int): Order of the quadrature rule.
        points (np.ndarray): Quadrature points in reference coordinates.
        weights (np.ndarray): Corresponding quadrature weights.
    """

    def __init__(self, dimension: int, order: int) -> None:
        """
        Initialize the QuadratureRule.

        Parameters:
            dimension (int): Spatial dimension.
            order (int): Order of the quadrature rule.
        """
        self.dimension: int = dimension
        self.order: int = order
        self.points: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

        self.setup_quadrature()

    def setup_quadrature(self) -> None:
        """
        Set up quadrature points and weights based on dimension and order.
        """
        if self.dimension == 1:
            self.points, self.weights = np.polynomial.legendre.leggauss(self.order)
            # Map from [-1, 1] to [0, 1]
            self.points = 0.5 * (self.points + 1)
            self.weights *= 0.5
        elif self.dimension == 2:
            # Use tensor product of 1D quadrature rules
            points_1d, weights_1d = np.polynomial.legendre.leggauss(self.order)
            # Map from [-1, 1] to [0, 1]
            points_1d = 0.5 * (points_1d + 1)
            weights_1d *= 0.5
            self.points = np.array([[x, y] for x in points_1d for y in points_1d])
            self.weights = np.array([wx * wy for wx in weights_1d for wy in weights_1d])
        elif self.dimension == 3:
            # Use tensor product of 1D quadrature rules
            points_1d, weights_1d = np.polynomial.legendre.leggauss(self.order)
            # Map from [-1, 1] to [0, 1]
            points_1d = 0.5 * (points_1d + 1)
            weights_1d *= 0.5
            self.points = np.array(
                [[x, y, z] for x in points_1d for y in points_1d for z in points_1d]
            )
            self.weights = np.array(
                [
                    wx * wy * wz
                    for wx in weights_1d
                    for wy in weights_1d
                    for wz in weights_1d
                ]
            )
        else:
            raise NotImplementedError(
                f"Quadrature not implemented for dimension {self.dimension}."
            )


class ReferenceElement:
    """
    ReferenceElement class containing shape functions and derivatives.

    Attributes:
        dimension (int): Spatial dimension.
        element_type (str): Type of finite element (e.g., 'Lagrange').
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
    """

    def __init__(self, dimension: int, element_type: str, degree: int) -> None:
        """
        Initialize the ReferenceElement.

        Parameters:
            dimension (int): Spatial dimension.
            element_type (str): Type of finite element.
            degree (int): Polynomial degree.
        """
        self.dimension: int = dimension
        self.element_type: str = element_type
        self.degree: int = degree
        self.nodes: Optional[np.ndarray] = None

        self.setup_reference_element()

    def setup_reference_element(self) -> None:
        """
        Set up the reference element nodes and related data.
        """
        if self.element_type == "Lagrange":
            if self.dimension == 1:
                self.nodes = np.linspace(0, 1, self.degree + 1)
            elif self.dimension == 2:
                num_nodes = (self.degree + 1) ** 2
                self.nodes = np.zeros((num_nodes, 2))
                index = 0
                for i in range(self.degree + 1):
                    for j in range(self.degree + 1):
                        self.nodes[index] = [i / self.degree, j / self.degree]
                        index += 1
            elif self.dimension == 3:
                num_nodes = (self.degree + 1) ** 3
                self.nodes = np.zeros((num_nodes, 3))
                index = 0
                for i in range(self.degree + 1):
                    for j in range(self.degree + 1):
                        for k in range(self.degree + 1):
                            self.nodes[index] = [
                                i / self.degree,
                                j / self.degree,
                                k / self.degree,
                            ]
                            index += 1
            else:
                raise NotImplementedError(
                    f"Reference element not implemented for dimension {self.dimension}."
                )
        else:
            raise NotImplementedError(
                f"Element type '{self.element_type}' not implemented."
            )

    def evaluate_shape_functions(self, local_coords: np.ndarray) -> np.ndarray:
        """
        Evaluate shape functions at given local coordinates.

        Parameters:
            local_coords (np.ndarray): Local coordinates where to evaluate shape functions.

        Returns:
            N (np.ndarray): Values of shape functions at local_coords.
        """
        if self.element_type == "Lagrange":
            N = lagrange_shape_functions(
                self.dimension, self.degree, self.nodes, local_coords
            )
            return N
        else:
            raise NotImplementedError(
                f"Shape functions not implemented for element type '{self.element_type}'."
            )

    def evaluate_shape_function_derivatives(
        self, local_coords: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate shape function derivatives at given local coordinates.

        Parameters:
            local_coords (np.ndarray): Local coordinates where to evaluate shape function derivatives.

        Returns:
            dN_dxi (np.ndarray): Derivatives of shape functions at local_coords.
        """
        if self.element_type == "Lagrange":
            dN_dxi = lagrange_shape_function_derivatives(
                self.dimension, self.degree, self.nodes, local_coords
            )
            return dN_dxi
        else:
            raise NotImplementedError(
                f"Shape function derivatives not implemented for element type '{self.element_type}'."
            )


def lagrange_shape_functions(
    dimension: int, degree: int, nodes: np.ndarray, local_coords: np.ndarray
) -> np.ndarray:
    """
    Compute Lagrange shape functions at given local coordinates.

    Parameters:
        dimension (int): Spatial dimension.
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        local_coords (np.ndarray): Local coordinates where to evaluate shape functions.

    Returns:
        N (np.ndarray): Values of shape functions at local_coords.
    """
    num_nodes = nodes.shape[0]
    N = np.zeros(num_nodes)
    if dimension == 1:
        xi = local_coords[0]
        for i in range(num_nodes):
            N[i] = lagrange_basis_1d(degree, nodes[:, 0], xi, i)
        return N
    elif dimension == 2:
        xi, eta = local_coords
        for i in range(num_nodes):
            N[i] = lagrange_basis_2d(degree, nodes, xi, eta, i)
        return N
    elif dimension == 3:
        xi, eta, zeta = local_coords
        for i in range(num_nodes):
            N[i] = lagrange_basis_3d(degree, nodes, xi, eta, zeta, i)
        return N
    else:
        raise NotImplementedError(
            f"Lagrange shape functions not implemented for dimension {dimension}."
        )


def lagrange_shape_function_derivatives(
    dimension: int, degree: int, nodes: np.ndarray, local_coords: np.ndarray
) -> np.ndarray:
    """
    Compute derivatives of Lagrange shape functions at given local coordinates.

    Parameters:
        dimension (int): Spatial dimension.
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        local_coords (np.ndarray): Local coordinates where to evaluate derivatives.

    Returns:
        dN_dxi (np.ndarray): Derivatives of shape functions at local_coords.
    """
    num_nodes = nodes.shape[0]
    if dimension == 1:
        dN_dxi = np.zeros((1, num_nodes))
        xi = local_coords[0]
        for i in range(num_nodes):
            dN_dxi[0, i] = lagrange_basis_derivative_1d(degree, nodes[:, 0], xi, i)
        return dN_dxi
    elif dimension == 2:
        dN_dxi = np.zeros((2, num_nodes))
        xi, eta = local_coords
        for i in range(num_nodes):
            dN_dxi[:, i] = lagrange_basis_derivative_2d(degree, nodes, xi, eta, i)
        return dN_dxi
    elif dimension == 3:
        dN_dxi = np.zeros((3, num_nodes))
        xi, eta, zeta = local_coords
        for i in range(num_nodes):
            dN_dxi[:, i] = lagrange_basis_derivative_3d(degree, nodes, xi, eta, zeta, i)
        return dN_dxi
    else:
        raise NotImplementedError(
            f"Lagrange shape function derivatives not implemented for dimension {dimension}."
        )


def lagrange_basis_1d(degree: int, nodes: np.ndarray, xi: float, i: int) -> float:
    """
    Compute 1D Lagrange basis function at xi.

    Parameters:
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        xi (float): Local coordinate.
        i (int): Basis function index.

    Returns:
        N_i (float): Value of the i-th basis function at xi.
    """
    N_i = 1.0
    xi_i = nodes[i]
    for j in range(len(nodes)):
        if j != i:
            xi_j = nodes[j]
            N_i *= (xi - xi_j) / (xi_i - xi_j)
    return N_i


def lagrange_basis_derivative_1d(
    degree: int, nodes: np.ndarray, xi: float, i: int
) -> float:
    """
    Compute derivative of 1D Lagrange basis function at xi.

    Parameters:
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        xi (float): Local coordinate.
        i (int): Basis function index.

    Returns:
        dN_i_dxi (float): Derivative of the i-th basis function at xi.
    """
    dN_i_dxi = 0.0
    xi_i = nodes[i]
    for j in range(len(nodes)):
        if j != i:
            xi_j = nodes[j]
            product = 1.0 / (xi_i - xi_j)
            for k in range(len(nodes)):
                if k != i and k != j:
                    xi_k = nodes[k]
                    product *= (xi - xi_k) / (xi_i - xi_k)
            dN_i_dxi += product
    return dN_i_dxi


def lagrange_basis_2d(
    degree: int, nodes: np.ndarray, xi: float, eta: float, i: int
) -> float:
    """
    Compute 2D Lagrange basis function at (xi, eta).

    Parameters:
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        xi, eta (float): Local coordinates.
        i (int): Basis function index.

    Returns:
        N_i (float): Value of the i-th basis function at (xi, eta).
    """
    xi_i, eta_i = nodes[i]
    N_i = 1.0
    for j in range(len(nodes)):
        if j != i:
            xi_j, eta_j = nodes[j]
            N_i *= ((xi - xi_j) / (xi_i - xi_j)) if xi_i != xi_j else 1.0
            N_i *= ((eta - eta_j) / (eta_i - eta_j)) if eta_i != eta_j else 1.0
    return N_i


def lagrange_basis_derivative_2d(
    degree: int, nodes: np.ndarray, xi: float, eta: float, i: int
) -> np.ndarray:
    """
    Compute derivative of 2D Lagrange basis function at (xi, eta).

    Parameters:
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        xi, eta (float): Local coordinates.
        i (int): Basis function index.

    Returns:
        dN_i_dxi (np.ndarray): Derivatives of the i-th basis function at (xi, eta).
    """
    # Placeholder implementation
    dN_i_dxi = np.zeros(2)
    # Compute partial derivatives with respect to xi and eta
    # This can be implemented using similar logic to the 1D case
    return dN_i_dxi


def lagrange_basis_3d(
    degree: int, nodes: np.ndarray, xi: float, eta: float, zeta: float, i: int
) -> float:
    """
    Compute 3D Lagrange basis function at (xi, eta, zeta).

    Parameters:
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        xi, eta, zeta (float): Local coordinates.
        i (int): Basis function index.

    Returns:
        N_i (float): Value of the i-th basis function at (xi, eta, zeta).
    """
    xi_i, eta_i, zeta_i = nodes[i]
    N_i = 1.0
    for j in range(len(nodes)):
        if j != i:
            xi_j, eta_j, zeta_j = nodes[j]
            N_i *= ((xi - xi_j) / (xi_i - xi_j)) if xi_i != xi_j else 1.0
            N_i *= ((eta - eta_j) / (eta_i - eta_j)) if eta_i != eta_j else 1.0
            N_i *= ((zeta - zeta_j) / (zeta_i - zeta_j)) if zeta_i != zeta_j else 1.0
    return N_i


def lagrange_basis_derivative_3d(
    degree: int, nodes: np.ndarray, xi: float, eta: float, zeta: float, i: int
) -> np.ndarray:
    """
    Compute derivative of 3D Lagrange basis function at (xi, eta, zeta).

    Parameters:
        degree (int): Polynomial degree.
        nodes (np.ndarray): Reference coordinates of element nodes.
        xi, eta, zeta (float): Local coordinates.
        i (int): Basis function index.

    Returns:
        dN_i_dxi (np.ndarray): Derivatives of the i-th basis function at (xi, eta, zeta).
    """
    # Placeholder implementation
    dN_i_dxi = np.zeros(3)
    # Compute partial derivatives with respect to xi, eta, and zeta
    # This can be implemented using similar logic to the 1D case
    return dN_i_dxi

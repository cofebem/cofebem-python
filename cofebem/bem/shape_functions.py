import numpy as np


def shape_functions(xi, eta):
    """
    Standard bilinear shape functions on [-1,1]x[-1,1] for a quadrilateral.
    """
    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)
    return np.array([N1, N2, N3, N4])


def shape_function_derivatives(xi, eta):
    """
    Derivatives of the bilinear shape functions with respect to xi and eta.
    Returns an array of shape (4, 2): for each node, [dN/dxi, dN/deta].
    """
    dN1_dxi = -0.25 * (1 - eta)
    dN1_deta = -0.25 * (1 - xi)
    dN2_dxi = 0.25 * (1 - eta)
    dN2_deta = -0.25 * (1 + xi)
    dN3_dxi = 0.25 * (1 + eta)
    dN3_deta = 0.25 * (1 + xi)
    dN4_dxi = -0.25 * (1 + eta)
    dN4_deta = 0.25 * (1 - xi)
    return np.array(
        [
            [dN1_dxi, dN1_deta],
            [dN2_dxi, dN2_deta],
            [dN3_dxi, dN3_deta],
            [dN4_dxi, dN4_deta],
        ]
    )


def map_to_physical_3d(element, xi, eta):
    """
    Maps the parametric coordinates (xi, eta) in [-1,1]^2 to the physical space.
    'element' is a (4,3) array with the coordinates of the 4 nodes.
    """
    N_vals = shape_functions(xi, eta)  # shape (4,)
    point = np.dot(N_vals, element)  # shape (3,)
    return point


def jacobian_determinant_3d(element, xi, eta):
    """
    Computes the surface Jacobian (area scaling factor) for a 3D quadrilateral element.
    'element' is a (4,3) array.
    The Jacobian matrix J is 3x2: partial derivatives of the mapping with respect to xi and eta.
    The area element is the norm of the cross product of the columns of J.
    """
    dN = shape_function_derivatives(xi, eta)  # shape (4,2)
    J = np.zeros((3, 2))
    for i in range(4):
        J[:, 0] += dN[i, 0] * element[i, :]  # derivative with respect to xi
        J[:, 1] += dN[i, 1] * element[i, :]  # derivative with respect to eta
    cross_prod = np.cross(J[:, 0], J[:, 1])
    return np.linalg.norm(cross_prod)

import numpy as np
import math


# def tri_shapes_function(xi1, xi2):
#     N1 = 1 - xi1 - xi2
#     N2 = xi1
#     N3 = xi2
#     return np.array([N1, N2, N3])


# def polar_tri_shape_functions(alpha):
#     N1 = -math.cos(alpha) + math.sin(alpha)
#     N2 = math.cos(alpha)
#     N3 = math.sin(alpha)
#     return np.array([N1, N2, N3])


# def tri_shape_function_derivatives(xi1, xi2):
#     dN1_dxi1 = -1
#     dN1_dxi2 = -1
#     dN2_dxi1 = 1
#     dN2_dxi2 = 0
#     dN3_dxi1 = 0
#     dN3_dxi2 = 1
#     return np.array(
#         [
#             [dN1_dxi1, dN1_dxi2],
#             [dN2_dxi1, dN2_dxi2],
#             [dN3_dxi1, dN3_dxi2],
#         ]
#     )


def shape_functions(xi1, xi2):
    N1 = 0.25 * (1 - xi1) * (1 - xi2)
    N2 = 0.25 * (1 + xi1) * (1 - xi2)
    N3 = 0.25 * (1 + xi1) * (1 + xi2)
    N4 = 0.25 * (1 - xi1) * (1 + xi2)
    return np.array([N1, N2, N3, N4])


def shape_function_derivatives(xi1, xi2):
    dN1_dxi1 = -0.25 * (1 - xi2)
    dN1_dxi2 = -0.25 * (1 - xi1)
    dN2_dxi1 = 0.25 * (1 - xi2)
    dN2_dxi2 = -0.25 * (1 + xi1)
    dN3_dxi1 = 0.25 * (1 + xi2)
    dN3_dxi2 = 0.25 * (1 + xi1)
    dN4_dxi1 = -0.25 * (1 + xi2)
    dN4_dxi2 = 0.25 * (1 - xi1)
    return np.array(
        [
            [dN1_dxi1, dN1_dxi2],
            [dN2_dxi1, dN2_dxi2],
            [dN3_dxi1, dN3_dxi2],
            [dN4_dxi1, dN4_dxi2],
        ]
    )


def map_to_physical_3d(element, xi1, xi2):
    N_vals = shape_functions(xi1, xi2)
    point = np.dot(N_vals, element)
    return point


def jacobian_determinant_3d(element, xi1, xi2):
    dN = shape_function_derivatives(xi1, xi2)  # shape (4,2)
    J = np.zeros((3, 2))
    for i in range(4):
        J[:, 0] += dN[i, 0] * element[i, :]
        J[:, 1] += dN[i, 1] * element[i, :]
    cross_prod = np.cross(J[:, 0], J[:, 1])
    return np.linalg.norm(cross_prod)


def integrate(
    kernel, x_c, normal, nodes, mu, nu, singular, n_gauss, local_index
):  # , regularity="reg"):

    # element = np.array(nodes).reshape((4, 3))
    element = nodes
    # match regularity:
    #     case "reg":
    #         gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(n_gauss)
    #         result = np.zeros((3, 3))
    #         for i in range(n_gauss):
    #             for j in range(n_gauss):
    #                 xi1 = gauss_nodes[i]
    #                 xi2 = gauss_nodes[j]
    #                 weight = gauss_weights[i] * gauss_weights[j]
    #                 N_vals = shape_functions(xi1, xi2)
    #                 N_local = N_vals[local_index]
    #                 y = map_to_physical_3d(element, xi1, xi2)
    #                 J_geo = jacobian_determinant_3d(element, xi1, xi2)
    #                 total_weight = weight * J_geo
    #                 result += kernel(x_c, y, normal, mu, nu) * (N_local * total_weight)
    #         return result
    #     case "near_sing":  # PART or Cubic(Telles) Transformatiom
    #         pass
    #     case "weak_sing":
    #         pass  # Duffy or Bonnet transformation
    #     case "strong_sing":
    #         pass  # CPV with Guggiani's method

    if not singular:
        gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(n_gauss)
        result = np.zeros((3, 3))
        for i in range(n_gauss):
            for j in range(n_gauss):
                xi1 = gauss_nodes[i]
                xi2 = gauss_nodes[j]
                weight = gauss_weights[i] * gauss_weights[j]
                N_vals = shape_functions(xi1, xi2)
                N_local = N_vals[local_index]
                y = map_to_physical_3d(element, xi1, xi2)
                J_geo = jacobian_determinant_3d(element, xi1, xi2)
                total_weight = weight * J_geo
                result += kernel(x_c, y, normal, mu, nu) * (N_local * total_weight)
        return result
    else:
        #  xi1 = -1 + ρ cos α,   xi2 = -1 + ρ sin α, with α ∈ [0, π/2] and ρ in [0, ρ_max(α)].
        result = np.zeros((3, 3))
        # Gauss quadrature in α over [0, π/2]
        # alpha_nodes, alpha_weights = np.polynomial.legendre.leggauss(n_gauss)
        # alpha_nodes = 0.5 * (alpha_nodes + 1) * (math.pi / 2)
        # alpha_weights = 0.5 * (math.pi / 2) * alpha_weights
        # for i in range(n_gauss):
        #     alpha = alpha_nodes[i]
        #     w_alpha = alpha_weights[i]
        #     # ρ_max = min(2/cosα, 2/sinα).
        #     rho_max = min(2.0 / np.cos(alpha), 2.0 / np.sin(alpha))
        #     # Gauss quadrature in ρ over [0, rho_max]
        #     rho_nodes, rho_weights = np.polynomial.legendre.leggauss(n_gauss)
        #     rho_nodes = 0.5 * (rho_nodes + 1) * rho_max
        #     rho_weights = 0.5 * rho_max * rho_weights
        #     for j in range(n_gauss):
        #         rho = rho_nodes[j]
        #         w_rho = rho_weights[j]
        #         xi1 = -1 + rho * np.cos(alpha)
        #         xi2 = -1 + rho * np.sin(alpha)
        #         N_vals = shape_functions(xi1, xi2)
        #         N_local = N_vals[local_index]
        #         y = map_to_physical_3d(element, xi1, xi2)
        #         J_geo = jacobian_determinant_3d(element, xi1, xi2)
        #         total_weight = w_alpha * w_rho * (rho * J_geo)
        #         result += kernel(x_c, y, normal, mu, nu) * (N_local * total_weight)
        return result

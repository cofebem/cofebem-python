import numpy as np
from abc import ABC, abstractmethod


class FundamentalSolution(ABC):
    def __init__(self, mat_params):
        self.mat_params = mat_params

    @abstractmethod
    def G(self, x, y, normal):
        pass

    @abstractmethod
    def H(self, x, y, normal):
        pass


class Kelvin(FundamentalSolution):
    def __init__(self, mat_params):
        super().__init__(mat_params)

    def G(self, x, y, normal):
        x = np.asarray(x)
        y = np.asarray(y)
        r = x - y
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            raise ValueError(
                "Singularity encountered: field and source points coincide."
            )

        I = np.eye(3)

        factor = 1.0 / (16 * np.pi * self.mat_params.mu * (1 - self.mat_params.nu))
        G = factor * (
            (3 - 4 * self.mat_params.nu) * I / r_norm + np.outer(r, r) / r_norm**3
        )
        return G

    def H(self, x, y, normal):
        x = np.asarray(x)
        y = np.asarray(y)
        normal = np.asarray(normal)
        r = x - y
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-12:
            raise ValueError(
                "Singularity encountered: field and source points coincide."
            )

        r_dot_n = np.dot(r, normal)
        I = np.eye(3)
        factor = -1.0 / (8 * np.pi * (1 - self.mat_params.nu))

        term1 = (1 - 2 * self.mat_params.nu) * I * (r_dot_n) / r_norm**3
        term2 = 3 * np.outer(r, r) * (r_dot_n) / r_norm**5
        term3 = np.outer(r, normal) / r_norm**3

        H = factor * (term1 + term2 - term3)
        return H


class Mindlin(FundamentalSolution):
    def __init__(self, mat_params):
        super().__init__(mat_params)

    def G(self, x, y, normal):
        pass

    def H(self, x, y, normal):
        pass


def kelvin_G(x, y, normal, mu, nu):
    x = np.asarray(x)
    y = np.asarray(y)
    r = x - y
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: field and source points coincide.")

    I = np.eye(3)

    factor = 1.0 / (16 * np.pi * mu * (1 - nu))
    G = factor * ((3 - 4 * nu) * I / r_norm + np.outer(r, r) / r_norm**3)
    return G


# def kelvin_H(x, y, normal, mu, nu):
#     x = np.asarray(x)
#     y = np.asarray(y)
#     normal = np.asarray(normal)
#     r = x - y
#     r_norm = np.linalg.norm(r)
#     if r_norm < 1e-12:
#         raise ValueError("Singularity encountered: field and source points coincide.")

#     r_dot_n = np.dot(r, normal)
#     I = np.eye(3)
#     factor = -1.0 / (8 * np.pi * (1 - nu))

#     term1 = -(1 - 2 * nu) * I * (r_dot_n) / r_norm**3
#     term2 = -3 * np.outer(r, r) * (r_dot_n) / r_norm**5
#     term3 = (1 - 2 * nu) * (np.outer(r, normal) - np.outer(normal, r)) / r_norm**3

#     H = factor * (term1 + term2 + term3)
#     return H


def kelvin_H(x, y, normal, mu, nu):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    normal = np.asarray(normal, dtype=float)

    r = x - y
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: x and y coincide (r = 0).")

    r_dot_n = np.dot(r, normal)

    I = np.eye(3)

    outer_rr = np.outer(r, r)

    outer_nr = np.outer(normal, r)
    outer_rn = np.outer(r, normal)

    term1 = (r_dot_n / r_norm) * ((1 - 2 * nu) * I + 3 * outer_rr)

    term2 = (1 - 2 * nu) * (outer_nr - outer_rn)

    factor = 1 / (8 * np.pi * (1 - nu) * r_norm**2)

    H = factor * (term1 + term2)

    return H

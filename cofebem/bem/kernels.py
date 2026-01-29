import numpy as np


def kelvin_G(x, y, normal, mu, nu):
    x = np.asarray(x)
    y = np.asarray(y)
    r = y - x
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: x and y coincide (r = 0).")

    I = np.eye(3)

    factor = 1.0 / (16 * np.pi * mu * (1 - nu) * r_norm)

    G = factor * ((3 - 4 * nu) * I + np.outer(r, r) / r_norm**2)

    return G


def kelvin_H(x, y, normal, mu, nu):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    normal = np.asarray(normal, dtype=float)

    r = y - x
    r_norm = np.linalg.norm(r)

    if r_norm < 1e-12:
        raise ValueError("Singularity encountered: x and y coincide (r = 0).")

    I = np.eye(3)

    factor = -1 / (8 * np.pi * (1 - nu) * r_norm**2)
    term1 = (
        np.dot(r, normal) * ((1 - 2 * nu) * I + 3 * np.outer(r, r) / r_norm**2) / r_norm
    )
    term2 = (1 - 2 * nu) * (np.outer(r, normal) - np.outer(normal, r)) / r_norm

    H = factor * (term1 - term2)

    return H

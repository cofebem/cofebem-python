from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class QuadratureRule:

    points: np.ndarray
    weights: np.ndarray
    degree: int
    domain: Literal["triangle", "quad"]
    family: str
    name: str

    def __post_init__(self) -> None:
        if self.points.ndim != 2:
            raise ValueError("points must have shape (n_points, dim)")
        if self.weights.ndim != 1:
            raise ValueError("weights must have shape (n_points,)")
        if self.points.shape[0] != self.weights.shape[0]:
            raise ValueError(
                "points and weights must have the same number of rows / entries"
            )
        if self.domain not in ("triangle", "quad"):
            raise ValueError("domain must be 'triangle' or 'quad'")


def _triangle_dunavant_rules() -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Return a dictionary mapping degree -> (points, weights) for T_hat.

    - points: (n_points, 2) array with (xi, eta) in reference triangle
    - weights: (n_points,) array, sum(weights) = 1/2

    Degrees implemented: 1, 2, 3.
    For these degrees, the symmetric Dunavant and Lyness–Jespersen rules coincide.
    """
    rules: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Helper: convert barycentric (λ1, λ2, λ3) -> (xi, eta)
    # with vertices V1=(0,0), V2=(1,0), V3=(0,1):
    #   x = λ2, y = λ3
    def bary_to_xy(l1: float, l2: float, l3: float) -> Tuple[float, float]:
        return float(l2), float(l3)

    # Degree 1, order 1: centroid rule
    # Barycentric: (1/3, 1/3, 1/3)
    # Single weight chosen so that sum w_i = area(T_hat) = 1/2.
    l1 = 1.0 / 3.0
    centroid = np.array([bary_to_xy(l1, l1, l1)])  # shape (1, 2)
    w1 = np.array([0.5])  # area of reference triangle
    rules[1] = (centroid, w1)

    # Degree 2, order 3: 3-point symmetric rule
    #
    # Barycentric points:
    #   (2/3, 1/6, 1/6)
    #   (1/6, 2/3, 1/6)
    #   (1/6, 1/6, 2/3)
    #
    # All weights equal, chosen so that sum w_i = 1/2.
    a = 1.0 / 6.0
    b = 2.0 / 3.0
    bary_pts_deg2 = [
        (b, a, a),
        (a, b, a),
        (a, a, b),
    ]
    pts_deg2 = np.array([bary_to_xy(L1, L2, L3) for (L1, L2, L3) in bary_pts_deg2])
    w_deg2 = np.full(3, 1.0 / 6.0)  # 3 * (1/6) = 1/2
    rules[2] = (pts_deg2, w_deg2)

    # Degree 3, order 4: 4-point symmetric rule
    #
    # Barycentric:
    #   1 point  : (1/3, 1/3, 1/3)          with weight w0 = -27/96
    #   3 points : (3/5, 1/5, 1/5) + perms  with weight w1 = 25/96 each
    #
    # Check: w0 + 3*w1 = -27/96 + 3*(25/96) = 48/96 = 1/2 (area).
    a = 3.0 / 5.0
    b = 1.0 / 5.0
    w0 = -27.0 / 96.0
    w1_tri = 25.0 / 96.0

    bary_pts_deg3 = [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),  # centroid
        (a, b, b),
        (b, a, b),
        (b, b, a),
    ]
    pts_deg3 = np.array([bary_to_xy(L1, L2, L3) for (L1, L2, L3) in bary_pts_deg3])
    w_deg3 = np.array([w0, w1_tri, w1_tri, w1_tri])
    rules[3] = (pts_deg3, w_deg3)

    return rules


_TRIANGLE_DUNAVANT = _triangle_dunavant_rules()

# We treat Lyness–Jespersen as identical to Dunavant for the
# implemented degrees (1–3), since the symmetric rules coincide.
_TRIANGLE_LYNESS_JESPERSEN = _TRIANGLE_DUNAVANT


def get_triangle_rule(
    degree: int,
    family: Literal["dunavant", "lyness_jespersen"] = "dunavant",
) -> QuadratureRule:
    """
    Return a symmetric quadrature rule on the reference triangle T_hat.

    Parameters
    ----------
    degree : int
        Polynomial degree of exactness (total degree).
        Implemented: 1, 2, 3.

    family : {"dunavant", "lyness_jespersen"}
        Which family of symmetric rules to use. For degrees 1–3, the
        actual rules are identical.

    Returns
    -------
    QuadratureRule
        Quadrature rule on T_hat.

    Notes
    -----
    - Points are given in (xi, eta) on T_hat:
        V1 = (0,0), V2 = (1,0), V3 = (0,1).
    - Weights sum to 1/2 (area of T_hat).
    - You can extend this function by adding more entries to the
      `_TRIANGLE_DUNAVANT` dictionary with higher-degree rules.
    """
    if family == "dunavant":
        table = _TRIANGLE_DUNAVANT
    elif family == "lyness_jespersen":
        table = _TRIANGLE_LYNESS_JESPERSEN
    else:
        raise ValueError(f"Unknown triangle quadrature family '{family}'")

    if degree not in table:
        available = sorted(table.keys())
        raise ValueError(
            f"No triangle rule of degree {degree} "
            f"for family '{family}'. Available degrees: {available}"
        )

    pts, wts = table[degree]
    return QuadratureRule(
        points=pts.copy(),
        weights=wts.copy(),
        degree=degree,
        domain="triangle",
        family=family,
        name=f"{family.capitalize()} degree {degree}, order {len(wts)}",
    )


# ---------------------------------------------------------------------------
# Quadrilateral rules: tensor-product Gauss–Legendre on Q_hat = [-1,1]^2
# ---------------------------------------------------------------------------


def _gauss_legendre_1d(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 1D Gauss–Legendre points and weights on [-1, 1].

    Parameters
    ----------
    order : int
        Number of points (order >= 1).

    Returns
    -------
    x : (order,) ndarray
        Points in [-1,1].
    w : (order,) ndarray
        Weights, sum(w) = 2.
    """
    if order < 1:
        raise ValueError("Gauss–Legendre order must be >= 1")

    # numpy's leggauss gives points and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(order)
    return x, w


def get_quad_rule(
    order: int | Tuple[int, int],
    family: Literal["gauss_legendre"] = "gauss_legendre",
) -> QuadratureRule:
    """
    Return a tensor-product quadrature rule on Q_hat = [-1,1]^2.

    Parameters
    ----------
    order : int or (int, int)
        If int:
            order in both directions (nx = ny = order).
        If tuple (nx, ny):
            separate orders in x and y directions.

        For Gauss–Legendre, the degree of exactness in each variable is:
            degree_x = 2*nx - 1
            degree_y = 2*ny - 1

    family : {"gauss_legendre"}
        Quadrature family. Currently only tensor-product Gauss–Legendre
        is implemented.

    Returns
    -------
    QuadratureRule
        Quadrature rule on Q_hat.

    Notes
    -----
    - Points are (xi, eta) in [-1,1]^2.
    - Weights sum to area(Q_hat) = 4.
    """
    if family != "gauss_legendre":
        raise ValueError(f"Unknown quadrilateral quadrature family '{family}'")

    if isinstance(order, int):
        nx = ny = order
    else:
        nx, ny = order
    if nx < 1 or ny < 1:
        raise ValueError("Quadrature orders nx, ny must be >= 1")

    x1d, w1d = _gauss_legendre_1d(nx)
    y1d, v1d = _gauss_legendre_1d(ny)

    # Tensor product
    X, Y = np.meshgrid(x1d, y1d, indexing="xy")  # shape (ny, nx)
    W = np.outer(v1d, w1d)  # shape (ny, nx), sum(W) = 4

    points = np.column_stack([X.ravel(), Y.ravel()])
    weights = W.ravel()

    degree = min(2 * nx - 1, 2 * ny - 1)
    return QuadratureRule(
        points=points,
        weights=weights,
        degree=degree,
        domain="quad",
        family=family,
        name=f"Tensor-product Gauss–Legendre ({nx}x{ny})",
    )


# ---------------------------------------------------------------------------
# Convenience wrappers / factory
# ---------------------------------------------------------------------------


def get_rule(
    domain: Literal["triangle", "quad"],
    *,
    degree: Optional[int] = None,
    order: Optional[int | Tuple[int, int]] = None,
    family: Optional[str] = None,
) -> QuadratureRule:
    """
    Generic factory for quadrature rules.

    Parameters
    ----------
    domain : {"triangle", "quad"}
        Reference domain.

    degree : int, optional
        Desired polynomial degree (triangle). Required if domain="triangle".

    order : int or (int, int), optional
        Quadrature order (quad). Required if domain="quad".

    family : str, optional
        Family name:
            - triangle: "dunavant" or "lyness_jespersen"
            - quad: "gauss_legendre"
          Defaults:
            - triangle: "dunavant"
            - quad: "gauss_legendre"

    Returns
    -------
    QuadratureRule
    """
    if domain == "triangle":
        if degree is None:
            raise ValueError("degree must be provided for triangle rules")
        fam = family or "dunavant"
        if fam not in ("dunavant", "lyness_jespersen"):
            raise ValueError("Triangle family must be 'dunavant' or 'lyness_jespersen'")
        return get_triangle_rule(degree=degree, family=fam)  # type: ignore[arg-type]

    elif domain == "quad":
        if order is None:
            raise ValueError("order must be provided for quadrilateral rules")
        fam = family or "gauss_legendre"
        if fam != "gauss_legendre":
            raise ValueError("Quadrilateral family must be 'gauss_legendre'")
        return get_quad_rule(order=order, family="gauss_legendre")

    else:
        raise ValueError("domain must be 'triangle' or 'quad'")

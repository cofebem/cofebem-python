from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional
import numpy as np


class Element(ABC):
    """
    Abstract base for BEM surface elements.

    Derived classes must implement:
      - element_type (property)
      - shape_functions(xi, eta)
      - shape_function_derivatives(xi, eta)
      - reference_center()
      - compute_area()  # exact or preferred evaluation for this element
    """

    __slots__ = (
        "_id",
        "_nodes",
        "_coords",
        "_order",
        "_dim",
        "_center",
        "_normal",
        "_area",
    )

    def __init__(
        self, id: int, nodes: Sequence[int], coords: np.ndarray, order: int
    ) -> None:
        self._id = int(id)
        self._nodes = np.asarray(nodes, dtype=np.int32)
        self._coords = np.asarray(coords, dtype=float)
        if self._coords.ndim != 2:
            raise ValueError("coords must be a (n_nodes, dim) array")
        if self._nodes.ndim != 1:
            raise ValueError("nodes must be a 1D array of indices")
        if self._coords.shape[0] != self._nodes.shape[0]:
            raise ValueError("nodes and coords must have the same length")
        self._order = int(order)
        self._dim = int(self._coords.shape[1])

        # lazy-computed geometric quantities
        self._center: Optional[np.ndarray] = None
        self._normal: Optional[np.ndarray] = None
        self._area: Optional[float] = None

    @property
    def id(self) -> int:
        return self._id

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def order(self) -> int:
        return self._order

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_nodes(self) -> int:
        return self._nodes.size

    @property
    @abstractmethod
    def element_type(self) -> str:
        """Human-readable type tag: 'triangle' or 'quad'."""
        ...

    @abstractmethod
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Return N(xi, eta) with shape (n_nodes,)."""
        ...

    @abstractmethod
    def shape_function_derivatives(
        self, xi: float, eta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dN/dxi, dN/deta), each shape (n_nodes,)."""
        ...

    @abstractmethod
    def reference_center(self) -> Tuple[float, float]:
        """Return a canonical reference point (xi_c, eta_c) (e.g., centroid)."""
        ...

    @abstractmethod
    def compute_area(self) -> float:
        """Return the element area/measure (exact or preferred for this type)."""
        ...

    def map_local_to_global(self, xi: float, eta: float) -> np.ndarray:
        """
        x(xi,eta) = sum_i N_i(xi,eta) * X_i
        Returns shape (dim,).
        """
        N = self.shape_functions(xi, eta)
        return (N[:, None] * self._coords).sum(axis=0)

    def jacobian_matrix(self, xi: float, eta: float) -> np.ndarray:
        """
        J = [dx/dxi, dx/deta] with shape (dim, 2).
        Uses dN and coords; valid for any element with provided dN.
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        dx_dxi = (dN_dxi[:, None] * self._coords).sum(axis=0)
        dx_deta = (dN_deta[:, None] * self._coords).sum(axis=0)
        return np.column_stack((dx_dxi, dx_deta))  # (dim, 2)

    def det_jacobian(self, xi: float, eta: float) -> float:
        """
        Surface metric:
          - In R^3: || dx/dxi x dx/deta ||
          - In R^2: | det([dx/dxi, dx/deta]) |
        """
        J = self.jacobian_matrix(xi, eta)
        if self._dim == 3:
            a, b = J[:, 0], J[:, 1]
            return float(np.linalg.norm(np.cross(a, b)))
        elif self._dim == 2:
            return float(abs(np.linalg.det(J)))
        else:
            raise ValueError("Unsupported geometric dimension for det_jacobian")

    @property
    def center(self) -> np.ndarray:
        if self._center is None:
            # Default center: map reference center
            xi_c, eta_c = self.reference_center()
            self._center = self.map_local_to_global(xi_c, eta_c)
        return self._center

    @property
    def normal(self) -> np.ndarray:
        if self._normal is None:
            if self._dim == 3:
                xi_c, eta_c = self.reference_center()
                J = self.jacobian_matrix(xi_c, eta_c)
                n = np.cross(J[:, 0], J[:, 1])
                norm = float(np.linalg.norm(n))
                if norm == 0.0:
                    # fallback using two polygon edges
                    e1 = self._coords[1] - self._coords[0]
                    e2 = self._coords[-1] - self._coords[0]
                    n = np.cross(e1, e2)
                    norm = float(np.linalg.norm(n))
                self._normal = n / (norm if norm != 0.0 else 1.0)
            elif self._dim == 2:
                # convention: out-of-plane normal for 2D embeddings (rare in BEM)
                self._normal = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                self._normal = np.zeros(self._dim, dtype=float)
        return self._normal

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = float(self.compute_area())
        return self._area

    # --------------------- utilities & mutation -------------------------------

    def flip_normal(self) -> None:
        """Flip stored normal (useful to enforce outward orientation)."""
        if self._normal is not None:
            self._normal = -self._normal

    def invalidate_cache(self) -> None:
        """Invalidate lazy-computed fields after coords update."""
        self._center = None
        self._normal = None
        self._area = None

    def set_coords(self, coords: np.ndarray) -> None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != self._coords.shape:
            raise ValueError("New coords must have the same shape as current coords")
        self._coords = coords
        self.invalidate_cache()

    def set_nodes(self, nodes: Sequence[int]) -> None:
        nodes = np.asarray(nodes, dtype=np.int32)
        if nodes.shape != self._nodes.shape:
            raise ValueError(
                "New nodes must have the same shape/length as current nodes"
            )
        self._nodes = nodes
        # no need to invalidate geometry: geometry follows coords, not indices

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "type": self.element_type,
            "order": self._order,
            "nodes": self._nodes.tolist(),
            "coords": self._coords.tolist(),
            "dim": self._dim,
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "area": self.area,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self._id}, type='{self.element_type}', "
            f"order=P{self._order}, n_nodes={self.n_nodes}, dim={self._dim})"
        )


class TriP0(Element):
    """
    3-node triangle with *linear geometry*; intended for P0 (constant) field interpolation.
    Geometry mapping uses the standard linear (P1) triangle shape functions.
    """

    __slots__ = ()

    @property
    def element_type(self) -> str:
        return "triangle"

    def __init__(
        self, id: int, nodes: Sequence[int], coords: np.ndarray, order: int = 0
    ) -> None:
        super().__init__(id, nodes, coords, order)
        if self.n_nodes != 3:
            raise ValueError("TriP0 expects exactly 3 nodes (triangle vertices).")

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        # Linear (P1) geometry
        return np.array([1.0 - xi - eta, xi, eta], dtype=float)

    def shape_function_derivatives(
        self, xi: float, eta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # dN/dxi and dN/deta for linear triangle (independent of (xi,eta))
        dN_dxi = np.array([-1.0, 1.0, 0.0], dtype=float)
        dN_deta = np.array([-1.0, 0.0, 1.0], dtype=float)
        return dN_dxi, dN_deta

    def reference_center(self) -> Tuple[float, float]:
        # Centroid in reference triangle
        return (1.0 / 3.0, 1.0 / 3.0)

    def compute_area(self) -> float:
        x1, x2, x3 = self._coords
        if self._dim == 3:
            return 0.5 * float(np.linalg.norm(np.cross(x2 - x1, x3 - x1)))
        elif self._dim == 2:
            E = np.column_stack((x2 - x1, x3 - x1))  # (2,2)
            return 0.5 * float(abs(np.linalg.det(E)))
        else:
            raise ValueError("TriP0 supports 2D or 3D embedded coordinates only.")


class TriP1(Element):
    """
    3-node linear (P1) triangle for both geometry mapping and (typically) P1 field interpolation.
    """

    __slots__ = ()

    @property
    def element_type(self) -> str:
        return "triangle"

    def __init__(
        self, id: int, nodes: Sequence[int], coords: np.ndarray, order: int = 1
    ) -> None:
        super().__init__(id, nodes, coords, order)
        if self.n_nodes != 3:
            raise ValueError("TriP1 expects exactly 3 nodes (triangle vertices).")

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        return np.array([1.0 - xi - eta, xi, eta], dtype=float)

    def shape_function_derivatives(
        self, xi: float, eta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        dN_dxi = np.array([-1.0, 1.0, 0.0], dtype=float)
        dN_deta = np.array([-1.0, 0.0, 1.0], dtype=float)
        return dN_dxi, dN_deta

    def reference_center(self) -> Tuple[float, float]:
        return (1.0 / 3.0, 1.0 / 3.0)

    def compute_area(self) -> float:
        x1, x2, x3 = self._coords
        if self._dim == 3:
            return 0.5 * float(np.linalg.norm(np.cross(x2 - x1, x3 - x1)))
        elif self._dim == 2:
            E = np.column_stack((x2 - x1, x3 - x1))
            return 0.5 * float(abs(np.linalg.det(E)))
        else:
            raise ValueError("TriP1 supports 2D or 3D embedded coordinates only.")


class QuadP0(Element):
    """
    4-node bilinear quadrilateral with linear/bilinear geometry mapping.
    Intended for P0 (constant) field interpolation at assembly time.
    Reference domain: (xi, eta) in [-1, 1] x [-1, 1]
    Node order (conventional): 1:(-1,-1), 2:(+1,-1), 3:(+1,+1), 4:(-1,+1)
    """

    __slots__ = ()

    @property
    def element_type(self) -> str:
        return "quadrilateral"

    def __init__(
        self, id: int, nodes: Sequence[int], coords: np.ndarray, order: int = 0
    ) -> None:
        super().__init__(id, nodes, coords, order)
        if self.n_nodes != 4:
            raise ValueError("QuadP0 expects exactly 4 nodes (quad vertices).")

    # ---- bilinear geometry shape functions (used by base class mapping) ----
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        # N1..N4 for bilinear quad on [-1,1]^2
        return 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),  # N1
                (1 + xi) * (1 - eta),  # N2
                (1 + xi) * (1 + eta),  # N3
                (1 - xi) * (1 + eta),  # N4
            ],
            dtype=float,
        )

    def shape_function_derivatives(
        self, xi: float, eta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # dN/dxi and dN/deta for bilinear quad
        dN_dxi = 0.25 * np.array(
            [
                -(1 - eta),  # dN1/dxi
                +(1 - eta),  # dN2/dxi
                +(1 + eta),  # dN3/dxi
                -(1 + eta),  # dN4/dxi
            ],
            dtype=float,
        )
        dN_deta = 0.25 * np.array(
            [
                -(1 - xi),  # dN1/deta
                -(1 + xi),  # dN2/deta
                +(1 + xi),  # dN3/deta
                +(1 - xi),  # dN4/deta
            ],
            dtype=float,
        )
        return dN_dxi, dN_deta

    def reference_center(self) -> Tuple[float, float]:
        # Center of the square [-1,1]^2
        return (0.0, 0.0)

    def compute_area(self) -> float:
        """
        Exact for planar quads; good approximation for warped quads.
        Uses 2x2 Gauss on [-1,1]^2: sum |Jsurf| * w.
        """
        # 2x2 Gauss points and weights
        gp = 1.0 / np.sqrt(3.0)
        gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        w = 1.0  # each weight is 1 for 2x2 on [-1,1]^2
        A = 0.0
        for xi, eta in gauss:
            A += self.det_jacobian(xi, eta) * w
        return float(A)


class QuadP1(Element):
    """
    4-node bilinear quadrilateral (P1-like field; i.e., bilinear basis).
    Geometry mapping is bilinear; field interpolation is typically bilinear at assembly.
    Reference domain: (xi, eta) in [-1, 1] x [-1, 1]
    """

    __slots__ = ()

    @property
    def element_type(self) -> str:
        return "quadrilateral"

    def __init__(
        self, id: int, nodes: Sequence[int], coords: np.ndarray, order: int = 1
    ) -> None:
        super().__init__(id, nodes, coords, order)
        if self.n_nodes != 4:
            raise ValueError("QuadP1 expects exactly 4 nodes (quad vertices).")

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        return 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),  # N1
                (1 + xi) * (1 - eta),  # N2
                (1 + xi) * (1 + eta),  # N3
                (1 - xi) * (1 + eta),  # N4
            ],
            dtype=float,
        )

    def shape_function_derivatives(
        self, xi: float, eta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        dN_dxi = 0.25 * np.array(
            [-(1 - eta), +(1 - eta), +(1 + eta), -(1 + eta)], dtype=float
        )
        dN_deta = 0.25 * np.array(
            [-(1 - xi), -(1 + xi), +(1 + xi), +(1 - xi)], dtype=float
        )
        return dN_dxi, dN_deta

    def reference_center(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def compute_area(self) -> float:
        gp = 1.0 / np.sqrt(3.0)
        gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        A = 0.0
        for xi, eta in gauss:
            A += self.det_jacobian(xi, eta)
        return float(A)

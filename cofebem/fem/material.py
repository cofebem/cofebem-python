from __future__ import annotations
from typing import Optional, Any, Mapping, Dict


class Material:
    __slots__ = ("_E", "_nu", "_rho", "_sigma_y", "_name")

    def __init__(
        self,
        E: float,
        nu: float,
        rho: Optional[float] = None,
        sigma_y: Optional[float] = None,
        name: Optional[str] = "Material",
    ) -> None:

        self.validate(E, nu, rho, sigma_y, name)
        self._E: float = float(E)
        self._nu: float = float(nu)
        self._rho: Optional[float] = float(rho) if rho is not None else None
        self._sigma_y: Optional[float] = float(sigma_y) if sigma_y is not None else None
        self._name: str = str(name) if name is not None else "Material"

    def __repr__(self) -> str:
        return (
            f"Material(name={self.name}, rho={self.rho}, E={self.E}, "
            f"nu={self.nu}, sigma_y={self.sigma_y})"
        )

    @staticmethod
    def validate(
        E: float,
        nu: float,
        rho: Optional[float],
        sigma_y: Optional[float],
        name: Optional[str],
    ) -> None:
        assert (
            isinstance(E, (int, float)) and E > 0
        ), "Young modulus E must be positive."
        assert isinstance(nu, (int, float)) and (
            -1.0 < nu < 0.5
        ), "Poisson's ratio must be in (-1, 0.5)."
        if rho is not None:
            assert (
                isinstance(rho, (int, float)) and rho > 0
            ), "Material density rho must be positive."
        if sigma_y is not None:
            assert (
                isinstance(sigma_y, (int, float)) and sigma_y > 0
            ), "Yield stress sigma_y must be positive."
        assert name is None or isinstance(
            name, str
        ), "The name variable must be a string"

    @property
    def E(self) -> float:
        return self._E

    @E.setter
    def E(self, value: float) -> None:
        assert (
            isinstance(value, (int, float)) and value > 0
        ), "The value of E is invalid"
        self._E = float(value)

    @property
    def nu(self) -> float:
        return self._nu

    @nu.setter
    def nu(self, value: float) -> None:
        assert isinstance(value, (int, float)) and (
            -1.0 < value < 0.5
        ), "The value of nu is invalid"
        self._nu = float(value)

    @property
    def rho(self) -> Optional[float]:
        return self._rho

    @rho.setter
    def rho(self, value: Optional[float]) -> None:
        if value is not None:
            assert (
                isinstance(value, (int, float)) and value > 0
            ), "The value of rho is invalid"
            self._rho = float(value)
        else:
            self._rho = None

    @property
    def sigma_y(self) -> Optional[float]:
        return self._sigma_y

    @sigma_y.setter
    def sigma_y(self, value: Optional[float]) -> None:
        if value is not None:
            assert (
                isinstance(value, (int, float)) and value > 0
            ), "The value of sigma_y is invalid"
            self._sigma_y = float(value)
        else:
            self._sigma_y = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        assert isinstance(value, str), "The value of name is invalid"
        self._name = value

    @property
    def lmbda(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def K(self) -> float:
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))

    def to_dict(self) -> Dict[str, Optional[float | str]]:
        return {
            "name": self.name,
            "rho": self.rho,
            "E": self.E,
            "nu": self.nu,
            "sigma_y": self.sigma_y,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Material:
        return cls(
            E=float(data["E"]),
            nu=float(data["nu"]),
            rho=(float(data["rho"]) if data.get("rho") is not None else None),
            sigma_y=(
                float(data["sigma_y"]) if data.get("sigma_y") is not None else None
            ),
            name=(str(data["name"]) if data.get("name") is not None else "Material"),
        )

    def copy(self, **updates: Any) -> Material:
        data = self.to_dict()
        data.update(updates)
        return Material.from_dict(data)

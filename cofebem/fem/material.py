import numpy as np


class Material:
    def __init__(self, E, nu, rho=None, sigma_y=None, name="Material"):
        assert rho > 0, "The density of the material must be positive."
        assert E > 0, "Young's modulus must be positive."
        assert 0 <= nu < 0.5, "Poisson's ratio must be in [0, 0.5)."
        assert sigma_y > 0, " The yield stress must be positive"
        self._rho = rho
        self._E = E
        self._nu = nu
        self._sigma_y = sigma_y
        self.name = name

    def __repr__(self):
        return f"Material(name={self.name}, rho={self.rho}, E={self.E}, nu={self.nu}, sigma_y={self.sigma_y})"

    @property
    def rho(self):
        return self._rho

    @property
    def E(self):
        return self._E

    @property
    def nu(self):
        return self._nu

    @property
    def sigma_y(self):
        return self._sigma_y

    @property
    def lmbda(self):
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def K(self):
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))

    def to_dict(self):
        """Converts material properties to a dictionary"""
        return {
            "name": self.name,
            "rho": self.rho,
            "E": self.E,
            "nu": self.nu,
            "sigma_y": self.sigma_y,
        }

    def from_dict(self, data):
        """Initialize material properties from a dictionary"""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary.")
        self.name = data.get("name", self.name)
        self._rho = data.get("rho", self.rho)
        self._E = data.get("E", self.E)
        self._nu = data.get("nu", self.nu)
        self._sigma_y = data.get("nu", self.sigma_y)

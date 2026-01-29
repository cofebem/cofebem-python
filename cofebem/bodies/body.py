from abc import ABC, abstractmethod
from .material import Material
from .geometry import Geometry


class Body(ABC):
    def __init__(self, geometry: Geometry, material: Material, tag=0):
        self._geometry = geometry
        self._material = material
        self._tag = tag

    @property
    def geometry(self):
        return self._geometry

    @property
    def material(self):
        return self._material

    @property
    def density(self):
        return self._density

    @property
    def tag(self):
        return self._tag

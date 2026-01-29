from abc import ABC


class Geometry(ABC):
    def __init__(self, dimensions, tag=0):
        self._dimensions = dimensions
        self._tag = tag

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def tag(self):
        return self._tag

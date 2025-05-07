class BC:
    def __init__(self, value, marker, tag):
        self._value = value
        self.marker = marker
        self._tag = tag

    @property
    def value(self):
        return self._value

    def tag(self):
        return self._tag


class DirichletBC:
    def __init__(self, uD, GammaD, tag=0):
        super.__init__(uD, GammaD, tag)
        self._uD = uD
        self.GammaD = GammaD
        self._tag = tag

    @property
    def uD(self):
        return self.uD

    @property
    def tag(self):
        return self._tag


class NeumannBC:
    def __init__(self, tN, GammaN, tag):
        super.__init__(tN, GammaN, tag)
        self._tN = tN
        self.GammaN = GammaN
        self._tag = tag

    @property
    def uD(self):
        return self.uD

    @property
    def tag(self):
        return self._tag

"""flexcontact - Contact computation library."""

from .solvers import constrained_cg, constrained_cg_python

__version__ = "0.1.0"

__all__ = [
    "constrained_cg",
    "constrained_cg_python",
]
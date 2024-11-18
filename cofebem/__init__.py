__version__ = "0.1.0"

# cofebem/__init__.py

"""
cofebem: A computational framework for finite and boundary element methods.

Subpackages:
- fem: Finite Element Method components.
- mesh: Mesh handling and operations.

Usage:
    import cofebem
    from cofebem import fem, mesh
"""

# Import subpackages to make them accessible when importing cofebem
from . import fem
from . import mesh

# Define the package's __all__ variable
__all__ = [
    'fem',
    'mesh',
]

__all__ = ["bem", "fem_inter", "contact", "utils"]

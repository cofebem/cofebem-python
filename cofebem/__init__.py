__version__ = "0.1.0"

from . import fem
from . import mesh

# Define the package's __all__ variable
__all__ = [
    "fem",
    "mesh",
]

__all__ = ["bem", "fem_inter", "contact", "utils"]

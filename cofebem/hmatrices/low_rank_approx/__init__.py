"""
Low-rank approximation methods for matrix compression.

Available methods:
- truncated_svd: Truncated singular value decomposition
- aca_full: Adaptive Cross Approximation (full)
- aca_partial: Adaptive Cross Approximation (partial) - default
- aca_plus: Adaptive Cross Approximation with enhanced recompression
- aca_gp: Adaptive Cross Approximation with Gaussian elimination pivoting
"""

from .truncated_svd import truncated_svd
from .aca_full import aca_full
from .aca_partial import aca_partial
from .aca_plus import aca_plus

__all__ = [
    "truncated_svd",
    "aca_full",
    "aca_partial",
    "aca_plus",
]

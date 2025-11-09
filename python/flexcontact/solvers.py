"""High-level Python interface for solvers."""

import numpy as np
from . import _core

def constrained_cg(K, error_type, coord, dofs, gap, max_iter, tolerance,
                   pressure_factor=1.0, initial_pressure=None):
    """
    Constrained Conjugate Gradient solver for contact problems.
    
    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Stiffness/flexibility matrix
    error_type : str
        Error type: 'displacement', 'mix', or 'nw'
    coord : np.ndarray
        Coordinates (not used in C++ implementation, kept for API compatibility)
    dofs : np.ndarray
        Degrees of freedom (not used in C++ implementation)
    gap : np.ndarray, shape (n,)
        Gap values
    max_iter : int
        Maximum iterations
    tolerance : float
        Convergence tolerance
    pressure_factor : float, optional
        Initial pressure scaling factor (default: 1.0)
    initial_pressure : np.ndarray, optional
        Initial pressure guess
    
    Returns
    -------
    pressure : np.ndarray
        Contact pressure
    displacement : np.ndarray
        Displacement field
    error_history : np.ndarray, shape (iterations, 3)
        Error history [displ_error, error, orthogonality]
    """
    # Input validation
    K = np.asarray(K, dtype=np.float64)
    gap = np.asarray(gap, dtype=np.float64)
    
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")
    
    if gap.shape[0] != K.shape[0]:
        raise ValueError("gap size must match K dimensions")
    
    if error_type not in ['displacement', 'mix', 'nw']:
        raise ValueError("error_type must be 'displacement', 'mix', or 'nw'")
    
    # Call C++ implementation
    result = _core.constrained_cg(
        K=K,
        error_type=error_type,
        gap=gap,
        max_iter=max_iter,
        tolerance=tolerance,
        pressure_factor=pressure_factor,
        initial_pressure=initial_pressure
    )
    
    return result.pressure, result.displacement, result.error_history


# Keep your original Python implementation as fallback
def constrained_cg_python(K, error_type, coord, dofs, gap, max_iter, 
                          tolerance, pressure_factor=1e12, initial_pressure=None):
    """Pure Python implementation (slower, for reference/testing)."""
    # [Your original Python code here]
    pass
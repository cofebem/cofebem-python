import numpy as np
from scipy.linalg import solve
from numba import njit
import cupy as cp

def schur_complement(A, B, C, D, assume_a='gen', overwrite_b=False, check_finite=False):
    """
    Compute the Schur complement of D in the block matrix M = [[A, B], [C, D]].

    Parameters:
    A (ndarray): Square matrix A.
    B (ndarray): Matrix B.
    C (ndarray): Matrix C.
    D (ndarray): Invertible square matrix D.
    assume_a (str, optional): Assumed type of the matrix D.
        Options are:
        - 'gen' : generic matrix (default)
        - 'sym' : symmetric matrix
        - 'her' : Hermitian matrix
        - 'pos' : symmetric positive definite
    overwrite_b (bool, optional): Allow overwriting data in C (may enhance performance).
    check_finite (bool, optional): Skip checking input matrices contain only finite numbers for performance.

    Returns:
    ndarray: The Schur complement matrix S = A - B D⁻¹ C.
    """
    # Solve D * X = C for X
    X = solve(D, C, assume_a=assume_a, overwrite_b=overwrite_b, check_finite=check_finite)

    # Compute B * X
    BX = np.dot(B, X)

    # Compute the Schur complement
    S = A - BX

    return S



@njit(fastmath=True, parallel=True)
def schur_complement_numba(A, B, C, D):

    # Solve D * X = C for X
    X = np.linalg.solve(D, C)

    # Compute B * X
    BX = np.dot(B, X)

    # Compute the Schur complement
    S = A - BX

    return S




def schur_complement_cupy(A, B, C, D):

    # Transfer data to GPU if not already
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    C_gpu = cp.asarray(C)
    D_gpu = cp.asarray(D)

    # Solve D * X = C for X on GPU
    X_gpu = cp.linalg.solve(D_gpu, C_gpu)

    # Compute B * X on GPU
    BX_gpu = cp.dot(B_gpu, X_gpu)

    # Compute the Schur complement on GPU
    S_gpu = A_gpu - BX_gpu

    # Transfer result back to CPU (if needed)
    S = cp.asnumpy(S_gpu)

    return S

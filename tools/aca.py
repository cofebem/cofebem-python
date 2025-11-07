"""
Adaptive Cross Approximation (ACA) Implementation

A simple implementation of the Adaptive Cross Approximation
algorithm for low-rank matrix decomposition.

Author: Vladislav A. Yastrebov
Affiliation: CNRS, Mines Paris, France
Created: May 2024
License: BSD 3-Clause License
"""

import numpy as np
import numba
import ClusterTree as ct

@numba.jit(nopython=True) #,parallel=True)
def frobenius_norm(M):
    total_sum = 0.0
    for i in numba.prange(M.shape[0]):
        for j in numba.prange(M.shape[1]):
            total_sum += M[i, j] ** 2
    return np.sqrt(total_sum)

def aca(A, tol=1e-6, max_rank=None):
    m, n = A.shape
    R = A.copy()
    Anorm = frobenius_norm(A)
    U = []
    V = []
    ranks = 0
    max_rank = min(m, n) if max_rank is None else min(max_rank, min(m, n))
    R_norm = np.zeros(max_rank)

    while np.linalg.norm(R, 'fro') > tol*Anorm and ranks < max_rank:
        # Find the pivot - largest element in the matrix by magnitude
        i_k, j_k = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        pivot = R[i_k, j_k]
        sign_ = np.sign(pivot)
        sqrt_pivot = np.sqrt(abs(pivot))
        if np.abs(pivot) < tol:  # If pivot is too small, break to avoid division by zero
            break

        # Extract the relevant column and row
        u_k = R[:, j_k]
        v_k = R[i_k, :]

        # Update the rank-1 components
        U.append(sign_ * u_k / sqrt_pivot)
        V.append(v_k / sqrt_pivot)
        
        # Update the residual matrix
        R -= np.outer(u_k, v_k) / pivot
        
        # R_norm[ranks] = np.linalg.norm(R, 'fro')
        # print(ranks, R_norm[ranks]/np.linalg.norm(A, 'fro'))
        # print(str(ranks) + " " + str(R_norm[ranks]))
        ranks += 1
        if max_rank is not None and ranks >= max_rank:
            break

    # Build the low rank approximation from the sum of outer products
    U = np.column_stack(U)
    V = np.stack(V)
    # A_approx = U @ V
    return U, V, np.linalg.norm(R, 'fro')/Anorm, ranks










"""

May be could be of help for the future, for the moment, within fenicsx-env I cannot make it work properly.


"""





# @numba.jit(nopython=True, parallel=True)
# def find_max_index(R):
#     max_vals = np.zeros(numba.get_num_threads(), dtype=np.float64)  # Use default number of threads, ensure dtype
#     max_indices = np.zeros((numba.get_num_threads(), 2), dtype=np.int64)  # Explicit dtype

#     for i in numba.prange(R.shape[0]):
#         for j in numba.prange(R.shape[1]):
#             thread_id = numba.np.ufunc.parallel._get_thread_id()
#             if np.abs(R[i, j]) > max_vals[thread_id]:
#                 max_vals[thread_id] = np.abs(R[i, j])
#                 max_indices[thread_id, 0] = i  # Assign row index separately
#                 max_indices[thread_id, 1] = j  # Assign column index separately

#     # Find the global maximum across all threads
#     global_max = max_vals[0]
#     global_index = (max_indices[0, 0], max_indices[0, 1])
#     for k in range(1, numba.get_num_threads()):
#         if max_vals[k] > global_max:
#             global_max = max_vals[k]
#             global_index = (max_indices[k, 0], max_indices[k, 1])

#     return global_index, global_max

@numba.jit(nopython=True) #,parallel=True)
def find_max_index(R):
    max_val = 0.0
    index = (0, 0)
    for i in numba.prange(R.shape[0]):
        for j in numba.prange(R.shape[1]):
            if np.abs(R[i, j]) > max_val:
                max_val = np.abs(R[i, j])
                index = (i, j)
    return index, max_val

# @numba.jit(nopython=True) #,parallel=True)
def outer_numba(a,b):
    result = np.zeros((a.shape[0], b.shape[0]))
    for i in numba.prange(a.shape[0]):
        for j in numba.prange(b.shape[0]):
            result[i, j] = a[i] * b[j]
    return result

# @numba.jit(nopython=True) #,parallel=True)
def aca_numba(A, tol=1e-6, max_rank=None):
    m, n = A.shape
    R = A.copy()
    max_possible_rank = min(m, n) if max_rank is None else min(max_rank, min(m, n))
    U = np.zeros((m, max_possible_rank))
    V = np.zeros((max_possible_rank, n))
    ranks = 0
    
    while frobenius_norm(R) > tol and ranks < max_possible_rank:
        # Find the pivot using the manual max index finder
        indices, pivot = find_max_index(R)
        i_k, j_k = indices
        # print("Pivot = ", pivot, ", index = ", i_k, j_k)
        if np.abs(pivot) < tol:
            print("Pivot is too small", pivot, ", index = ", i_k, j_k)
            break

        # Extract the relevant column and row
        u_k = R[:, j_k].copy() * np.sign(pivot) 
        v_k = R[i_k, :].copy() 

        u_k = u_k / np.sqrt(np.abs(pivot))
        v_k = v_k / np.sqrt(np.abs(pivot))

        # Update the rank-1 components
        U[:, ranks] = u_k.copy()
        V[ranks, :] = v_k.copy()

        # Update the residual matrix
        R -= outer_numba(u_k, v_k) 
        print(ranks,frobenius_norm(R))
        ranks += 1

    # Build the low rank approximation from the sum of outer products
    U_contiguous = np.ascontiguousarray(U[:, :ranks])
    V_contiguous = np.ascontiguousarray(V[:ranks, :])
    # A_approx = U_contiguous @ V_contiguous

    return U_contiguous, V_contiguous, frobenius_norm(R)/frobenius_norm(A), ranks

@numba.jit(nopython=True)
def aca2(A, tol=1e-6, max_rank=None):
    m, n = A.shape
    R = A.copy()
    ranks = 0
    if max_rank is None:
        max_rank = min(m, n)
    max_rank = min(max_rank, min(m, n))
    U = np.zeros((m, max_rank))
    V = np.zeros((max_rank, n))
    R_norm = np.zeros(max_rank)
    print(" Iteration | Frobenius norm(R)")

    while frobenius_norm(R) > tol:
        # Find the pivot - largest element in the matrix by magnitude
        (i_k, j_k), pivot = find_max_index(R)
        # pivot = R[i_k, j_k]
        sign_ = np.sign(pivot)
        sqrt_pivot = np.sqrt(abs(pivot))
        if np.abs(pivot) < tol:  # If pivot is too small, break to avoid division by zero
            break

        # Extract the relevant column and row
        u_k = R[:, j_k]
        v_k = R[i_k, :]

        # Update the rank-1 components
        # U.append(sign_ * u_k / sqrt_pivot)
        U[:, ranks] = sign_ * u_k / sqrt_pivot
        # V.append(v_k / sqrt_pivot)
        V[ranks, :] = v_k / sqrt_pivot
        # U.append(u_k / pivot)
        # V.append(v_k)
        
        # Update the residual matrix
        R -= np.outer(u_k, v_k) / pivot
        R_norm[ranks] = frobenius_norm(R)

        print(ranks, R_norm[ranks])
        # print(str(ranks) + " " + str(R_norm[ranks]))
        ranks += 1
        if max_rank is not None and ranks >= max_rank:
            break

    # Build the low rank approximation from the sum of outer products
    U = np.ascontiguousarray(U[:, :ranks])
    V = np.ascontiguousarray(V[:ranks, :])
    A_approx = U @ V
    return A_approx, ranks, R_norm


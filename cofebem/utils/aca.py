import numpy as np


class ACA:

    def __init__(self, tol=1e-6, max_rank=None, verbose=False):
        self.tol = tol
        self.max_rank = max_rank
        self.verbose = verbose
        self.U = None
        self.V = None
        self.error = None
        self.rank = None

    def decompose(self, A):
        m, n = A.shape
        max_rank = self.max_rank if self.max_rank is not None else min(m, n)

        U_list = []
        V_list = []

        i0 = np.argmax(np.linalg.norm(A, axis=1))
        if self.verbose:
            print(f"Initial pivot row index: {i0}")

        r = A[i0, :].copy()

        j0 = np.argmax(np.abs(r))
        pivot = r[j0]
        if self.verbose:
            print(f"Initial pivot column index: {j0}, pivot value: {pivot:.3e}")
        if np.abs(pivot) < self.tol:
            self.U = np.zeros((m, 0))
            self.V = np.zeros((0, n))
            self.error = 0.0
            self.rank = 0
            return self.U, self.V, self.error, self.rank

        u = A[:, j0].copy()

        v = r / pivot
        U_list.append(u)
        V_list.append(v)

        k = 1
        i_pivot = i0
        while k < max_rank:

            approx_row = np.zeros(n)
            for l in range(k):
                approx_row += U_list[l][i_pivot] * V_list[l]

            r = A[i_pivot, :] - approx_row

            j_pivot = np.argmax(np.abs(r))
            pivot = r[j_pivot]
            if self.verbose:
                print(
                    f"Iteration {k}: chosen pivot column index: {j_pivot}, pivot value: {pivot:.3e}"
                )
            if np.abs(pivot) < self.tol:
                if self.verbose:
                    print("Terminating ACA: pivot below tolerance (row residual).")
                break

            # For the selected column j_pivot, compute the residual for the entire column.
            approx_col = np.zeros(m)
            for l in range(k):
                approx_col += U_list[l] * V_list[l][j_pivot]
            col_residual = A[:, j_pivot] - approx_col

            # Choose new pivot row from the column residual.
            i_pivot = np.argmax(np.abs(col_residual))
            pivot = col_residual[i_pivot]
            if self.verbose:
                print(
                    f"Iteration {k}: new pivot row index: {i_pivot}, updated pivot value: {pivot:.3e}"
                )
            if np.abs(pivot) < self.tol:
                if self.verbose:
                    print("Terminating ACA: pivot below tolerance (column residual).")
                break

            u_new = A[:, j_pivot].copy()
            for l in range(k):
                u_new -= U_list[l] * V_list[l][j_pivot]

            approx_row = np.zeros(n)
            for l in range(k):
                approx_row += U_list[l][i_pivot] * V_list[l]
            r_new = A[i_pivot, :] - approx_row

            v_new = r_new / pivot

            U_list.append(u_new)
            V_list.append(v_new)
            k += 1

        U_mat = np.column_stack(U_list)
        V_mat = np.row_stack(V_list)

        if k > 0:
            err = np.linalg.norm(r) / np.linalg.norm(A[i_pivot, :])
        else:
            err = 0.0

        self.U = U_mat
        self.V = V_mat
        self.error = err
        self.rank = k

        if self.verbose:
            print(f"ACA completed with rank: {k}, estimated relative error: {err:.3e}")
        return U_mat, V_mat, err, k


import numpy as np
from numba import jit, prange, float64, int32
from numba.types import Tuple
import os
import math

os.environ["NUMBA_THREADING_LAYER"] = "omp"
NCPU = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(NCPU)

# Extra functions


def find_circle(x1, y1, x2, y2, x3, y3):
    """
    Find the circle passing through three points (x1, y1), (x2, y2), (x3, y3).
    """
    # Calculate the coefficients of the equations
    A1 = x2 - x1
    B1 = y2 - y1
    C1 = x2**2 - x1**2 + y2**2 - y1**2

    A2 = x3 - x1
    B2 = y3 - y1
    C2 = x3**2 - x1**2 + y3**2 - y1**2

    # Compute the determinant
    D = A1 * B2 - A2 * B1

    if D == 0:
        return 0, 0, np.inf

    # Calculate the center of the circle (x0, y0)
    x0 = (C1 * B2 - C2 * B1) / (2 * D)
    y0 = (A1 * C2 - A2 * C1) / (2 * D)

    # Calculate the radius of the circle
    r = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    return x0, y0, r


def distance_to_the_circle(x, y, x0, y0, r):
    return abs(np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - r)


# FIXME: separate function to compute the Frobenius norm to be adjusted for numba usage
def frobenius_norm(x):
    return np.linalg.norm(x)


# @numba.jit(nopython=True, parallel=True)
def line_kernel(t_coord, s_coord, E, kernel_power):
    """
    Compute the line kernel between two points.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        E (float): The kernel/Green function factor.
        kernel_power (int): The power of the kernel.

    Returns:
        result (numpy.ndarray): The line kernel between the target and source points.
    """
    factor = 1 / (np.pi * E)
    if t_coord.ndim == 1 and s_coord.ndim > 1:
        result = np.zeros(s_coord.shape[0])
        for i in prange(s_coord.shape[0]):
            dist = 0.0
            for j in prange(t_coord.shape[0]):
                dist += (t_coord[j] - s_coord[i, j]) ** 2
            result[i] = factor / np.sqrt(dist) ** kernel_power
        return result
    elif s_coord.ndim == 1 and t_coord.ndim > 1:
        result = np.zeros(t_coord.shape[0])
        for i in prange(t_coord.shape[0]):
            dist = 0.0
            for j in prange(s_coord.shape[0]):
                dist += (t_coord[i, j] - s_coord[j]) ** 2
            result[i] = factor / np.sqrt(dist) ** kernel_power
        return result
    elif s_coord.ndim == 1 and t_coord.ndim == 1:
        dist = 0.0
        for i in prange(t_coord.shape[0]):
            dist += (t_coord[i] - s_coord[i]) ** 2
        return factor / np.sqrt(dist) ** kernel_power
    else:
        raise ValueError(
            "Unknown dimension of t_coord or s_coord. One of two should be 1D array of size dim."
        )


def find_max_vector_element(v, Ik):
    """
    Find the maximum element in the vector v excluding the elements at the indices in Ik.
    """
    mask = np.ones(len(v), dtype=bool)
    mask[Ik] = False
    max_index = np.argmax(np.abs(v[mask]))
    return np.abs(v[mask][max_index]), np.arange(len(v))[mask][max_index]


###########################################
##        ACA  classic algorithm         ##
###########################################


def aca(t_coord, s_coord, tol, max_rank, min_pivot, kernel_factor, kernel_power):
    """
    Classical ACA with a random choice of pivot.

    Args:
        t_coord (numpy.ndarray):    The target cloud of points.
        s_coord (numpy.ndarray):    The source cloud of points.
        kernel_factor (float):              The kernel/Green function parameter.
        tol (float):                The tolerance for the ACA algorithm.
        max_rank (int):             The maximum rank of the ACA algorithm.
        min_pivot (float):          The minimum tolerable value of the pivot.
        kernel_power (int):               The power of the kernel.

    Returns:
        U_contiguous, V_contiguous (numpy.ndarray): The U and V matrices of the ACA algorithm which decompose the matrix A as U @ V.
        R_norm/M_norm (float): The ratio of the residual norm to the approximate matrix norm.
        ranks (int): The rank of the approximate decomposition.
        Jk (numpy.ndarray): The indices of the columns of the source cloud of points.
        Ik (numpy.ndarray): The indices of the rows of the target cloud of points.
    """
    n = t_coord.shape[0]
    m = s_coord.shape[0]
    U = np.zeros((n, max_rank), dtype=np.float64)
    V = np.zeros((max_rank, m), dtype=np.float64)
    ranks = 0
    R_norm = float64(1.0)
    M_norm = float64(1.0)
    Jk = []
    Ik = []
    history = np.zeros((max_rank, 4), dtype=np.float64)

    # Start algorithm
    # 1. Find randomly a column
    j1 = np.random.randint(0, m)
    # 2. Find the maximal elemnt
    u1 = line_kernel(t_coord, s_coord[j1], kernel_factor, kernel_power)
    _, i1 = find_max_vector_element(u1, Ik)
    v1 = line_kernel(t_coord[i1], s_coord, kernel_factor, kernel_power)
    Ik.append(i1)
    Jk.append(j1)
    pivot = v1[j1]
    if abs(pivot) < min_pivot:
        print("/ ACA Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks)
        return U, V, 0.0, ranks, Jk, Ik, history
        # raise ValueError("Pivot is too small: " +str(pivot))
    sign_pivot = np.sign(pivot)
    sqrt_pivot = np.sqrt(np.abs(pivot))
    # 4. Evaluate the associated row and column of the matrix
    U[:, 0] = sign_pivot * u1 / sqrt_pivot
    V[0, :] = v1 / sqrt_pivot

    # Compute matrix norm
    R_norm = frobenius_norm(u1) * frobenius_norm(v1) / np.abs(pivot)
    M_norm = R_norm
    history[0] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

    ranks = 1
    # Main loop
    while R_norm > tol * M_norm and ranks < max_rank:
        # Version 1: Get a random column which is not in Jk
        j_k = Jk[0]
        while j_k in Jk:
            j_k = np.random.randint(0, m)
        # Version 2: Search for maximal entry in v_k
        # _, j_k = find_max_vector_element(v_k, Jk)

        Jk.append(j_k)
        # Extract the column
        u_k = line_kernel(t_coord, s_coord[j_k], kernel_factor, kernel_power)
        # Remove the contribution of the previous ranks
        u_k -= np.dot(U[:, :ranks], V[:ranks, j_k])

        # Find the maximal element
        _, i_k = find_max_vector_element(u_k, Ik)
        # Store the column
        Ik.append(i_k)
        # Compute the new row and column
        v_k = line_kernel(t_coord[i_k], s_coord, kernel_factor, kernel_power)
        # Remove the contribution of the previous ranks
        v_k -= np.dot(U[i_k, :ranks], V[:ranks, :])

        pivot = u_k[i_k]
        if abs(pivot) < min_pivot:
            print(
                "/ ACA Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks
            )
            return U, V, 0.0, ranks, Jk, Ik, history
        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))

        U[:, ranks] = u_k * sign_pivot / sqrt_pivot
        V[ranks, :] = v_k / sqrt_pivot

        # Compute residual norm
        u_k_norm = frobenius_norm(u_k)
        v_k_norm = frobenius_norm(v_k)
        R_norm = float64(u_k_norm * v_k_norm / np.abs(pivot))

        # Approximate matrix norm
        cross_term = (
            2 * np.dot(np.dot(U[:, :ranks].T, u_k), np.dot(v_k, V[:ranks, :].T)) / pivot
        )
        M_norm = float64(np.sqrt(M_norm**2 + R_norm**2 + cross_term))

        # Increment the rank
        ranks += 1
        history[ranks - 1] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

    U_contiguous = np.ascontiguousarray(U[:, :ranks])
    V_contiguous = np.ascontiguousarray(V[:ranks, :])

    return U_contiguous, V_contiguous, R_norm / M_norm, ranks, Jk, Ik, history


###########################################
##           ACA-GP algorithm            ##
###########################################


def aca_gp(
    t_coord,
    s_coord,
    tol,
    max_rank,
    min_pivot,
    kernel_factor,
    kernel_power,
    central_fraction,
    square_shape=True,
):
    """
    Adaptive Cross Approximation with Geometrical Pivots (ACA-GP) for $m \\le n$.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        tol (float): The global tolerance for the ACA-GP algorithm.
        max_rank (int): The maximum rank for the ACA-GP algorithm.
        min_pivot (float): The pivot tolerance.
        kernel_factor (float): The kernel/Green function factor.
        kernel_power (int): The power of the kernel.
        central_fraction (float): A relative distance to define the central subsets.
        square_shape (bool): If True, the shape of the target and source clouds should be square-like, if False, the clouds can have arbitrary shape.
    Returns:
        U_contiguous, V_contiguous (numpy.ndarray): The U and V matrices of the ACA-GP algorithm which decompose the matrix A as U @ V.
        R_norm/M_norm (float): The ratio of the residual norm to the approximate matrix norm.
        ranks (int): The rank of the approximate decomposition.
        Jk (numpy.ndarray): The indices of the columns of the source cloud of points.
        Ik (numpy.ndarray): The indices of the rows of the target cloud of points.
        history (numpy.ndarray): The history of the ACA-GP iterations including the residual norm, the approximate matrix norm, the ratio of the residual norm to the approximate matrix norm, and the pivot value.
        central_fraction_s (float): The central fraction of the source cloud.
        central_fraction_t (float): The central fraction of the target cloud.
    """

    n = t_coord.shape[0]
    m = s_coord.shape[0]
    U = np.zeros((n, max_rank), dtype=np.float64)
    V = np.zeros((max_rank, m), dtype=np.float64)
    ranks = 0
    R_norm = 1.0
    M_norm = 1.0
    Jk = []
    Ik = []
    Ic = []
    Jc = []
    history = np.zeros((max_rank, 4), dtype=np.float64)

    # Find centers of both point clouds
    center_t = np.mean(t_coord, axis=0)
    center_s = np.mean(s_coord, axis=0)

    # Estimate diam of the point clouds
    diam_t = 2 * np.max(np.linalg.norm(t_coord - center_t, axis=1))
    diam_s = 2 * np.max(np.linalg.norm(s_coord - center_s, axis=1))

    # Find first pivot (i1, j1) as the closest to the center and on the side of the opposite cloud
    # select only those t_coord which verify the condition np.dot(t_coord-center_t, center_s-center_t) > 0
    t_coord_filtered = t_coord[np.dot(t_coord - center_t, center_s - center_t) > 0]
    t_coord_filtered_index = np.arange(n)[
        np.dot(t_coord - center_t, center_s - center_t) > 0
    ]
    distances_t = np.linalg.norm(t_coord_filtered - center_t, axis=1)
    i1 = t_coord_filtered_index[
        np.argmin(distances_t)
    ]  # [np.dot(t_coord-center_t, center_s-center_t) > 0])
    s_coord_filtered = s_coord[np.dot(s_coord - center_s, center_t - center_s) > 0]
    s_coord_filtered_index = np.arange(m)[
        np.dot(s_coord - center_s, center_t - center_s) > 0
    ]
    distances_s = np.linalg.norm(s_coord_filtered - center_s, axis=1)
    j1 = s_coord_filtered_index[
        np.argmin(distances_s)
    ]  # [np.dot(s_coord-center_s, center_t-center_s) > 0])

    Ik.append(i1)
    Jk.append(j1)

    # Construct central subsets.
    # If number of elements in the central subset is less than the maximal rank, increase the central subset
    central_fraction_t = central_fraction
    central_fraction_s = central_fraction
    while len(Ic) < max_rank + 5:
        for i in range(n):
            # if np.linalg.norm(t_coord[i] - center_t) < central_fraction_t  and i != i1:
            if (
                np.linalg.norm(t_coord[i] - t_coord[i1]) < central_fraction_t * diam_t
                and i != i1
            ):
                Ic.append(i)
        # If the central subset is still too small, increase the central fraction
        if len(Ic) < max_rank + 5:
            central_fraction_t *= 1.1
    while len(Jc) < max_rank + 5:
        for j in range(m):
            # if np.linalg.norm(s_coord[j] - center_s) < central_fraction_s  and j != j1:
            if (
                np.linalg.norm(s_coord[j] - s_coord[j1]) < central_fraction_s * diam_s
                and j != j1
            ):
                Jc.append(j)
        # If the central subset is still too small, increase the central fraction
        if len(Jc) < max_rank + 5:
            central_fraction_s *= 1.1
    if central_fraction_s != central_fraction:
        print(
            "/ ACA-GP Warning: Central fraction was increased for set Y to ",
            central_fraction_s,
        )
    if central_fraction_t != central_fraction:
        print(
            "/ ACA-GP Warning: Central fraction was increased for set X to ",
            central_fraction_t,
        )

    # Initialize first row and column for U and V matrices
    pivot = line_kernel(t_coord[i1], s_coord[j1], kernel_factor, kernel_power)
    if abs(pivot) < min_pivot:
        print(
            "/ ACA-GP Warning: Pivot is too small: ", pivot, ", stop at rank = ", ranks
        )
        return U, V, 0.0, ranks, Jk, Ik, history, central_fraction_s, central_fraction_t

    sign_pivot = np.sign(pivot)
    sqrt_pivot = np.sqrt(np.abs(pivot))

    u1 = line_kernel(t_coord, s_coord[j1], kernel_factor, kernel_power)
    U[:, 0] = sign_pivot * u1 / sqrt_pivot

    v1 = line_kernel(t_coord[i1], s_coord, kernel_factor, kernel_power)
    V[0, :] = v1 / sqrt_pivot

    # Compute initial residual and approximate matrix norms
    R_norm = np.linalg.norm(U[:, 0]) * np.linalg.norm(V[0, :])
    M_norm = R_norm
    history[0] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

    ranks = 1

    # Main ACA-GP loop
    while R_norm > tol * M_norm and ranks < max_rank:
        # Select next pivot points (ir, jr)
        if square_shape:
            if ranks == 1:  # k = 2
                ir, jr, pivot = rank2_selection(
                    t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, kernel_factor, kernel_power
                )
            elif ranks == 2:  # k = 3
                ir, jr, pivot = rank3_selection(
                    t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, kernel_factor, kernel_power
                )
            else:  # k >= 4
                ir, jr, pivot = higher_ranks_selection(
                    t_coord, s_coord, Ic, Jc, U, V, ranks, kernel_factor, kernel_power
                )
        else:
            ir, jr, pivot = higher_ranks_selection(
                t_coord, s_coord, Ic, Jc, U, V, ranks, kernel_factor, kernel_power
            )

        # Add pivots to the selected sets
        Ik.append(ir)
        Jk.append(jr)
        # remove pivots from central subsets
        # if ranks <= 2:
        if ir in Ic:
            Ic.remove(ir)
        if jr in Jc:
            Jc.remove(jr)

        # Compute new row and column
        if abs(pivot) < min_pivot:
            print(
                "/ ACA-GP Warning: Pivot is too small: {0:.2e}, stop at rank = {1:d}".format(
                    pivot, ranks
                )
            )
            return (
                U,
                V,
                0.0,
                ranks,
                Jk,
                Ik,
                history,
                central_fraction_s,
                central_fraction_t,
            )

        sign_pivot = np.sign(pivot)
        sqrt_pivot = np.sqrt(np.abs(pivot))

        ur = line_kernel(t_coord, s_coord[jr], kernel_factor, kernel_power) - np.dot(
            U[:, : ranks + 1], V[: ranks + 1, jr]
        )
        vr = line_kernel(t_coord[ir], s_coord, kernel_factor, kernel_power) - np.dot(
            U[ir, : ranks + 1], V[: ranks + 1, :]
        )
        assert np.abs(pivot - ur[ir]) < 1e-10, "Pivot is not correct"
        assert np.abs(pivot - vr[jr]) < 1e-10, "Pivot is not correct"
        # Update U and V
        U[:, ranks] = ur * sign_pivot / sqrt_pivot
        V[ranks, :] = vr / sqrt_pivot

        # Update residual norm
        ur_norm = np.linalg.norm(ur)
        vr_norm = np.linalg.norm(vr)
        R_norm = float(ur_norm * vr_norm / np.abs(pivot))

        # Update matrix norm
        cross_term = (
            2 * np.dot(np.dot(U[:, :ranks].T, ur), np.dot(vr, V[:ranks, :].T)) / pivot
        )
        M_norm = float(np.sqrt(M_norm**2 + R_norm**2 + cross_term))
        history[ranks] = np.array([R_norm, M_norm, R_norm / M_norm, pivot])

        # Increment rank
        ranks += 1

    U_contiguous = np.ascontiguousarray(U[:, : ranks + 1])
    V_contiguous = np.ascontiguousarray(V[: ranks + 1, :])

    return (
        U_contiguous,
        V_contiguous,
        R_norm / M_norm,
        ranks,
        Jk,
        Ik,
        history,
        central_fraction_s,
        central_fraction_t,
    )


def rank2_selection(
    t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, kernel_factor, kernel_power
):
    """
    Rank 2 selection of optimal points based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
    Returns:
        i2, j2 (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a random point from the target set excluding the first pivot
    i2 = np.random.choice(Ic)

    # Step 2: construct the circle passing through the first pivots (x_i1,y_j1) and the selected point x_i2
    x1 = t_coord[Ik[0]][0]
    y1 = t_coord[Ik[0]][1]
    x2 = s_coord[Jk[0]][0]
    y2 = s_coord[Jk[0]][1]
    x3 = t_coord[i2][0]
    y3 = t_coord[i2][1]

    x0, y0, r = find_circle(x1, y1, x2, y2, x3, y3)

    # Step 3: Initialization
    l = 0
    Jcl = Jc.copy()
    prev_pivot = 0
    while True:
        # Step 4: Select a point y_{j^l_2} that minimizes distance to the circle
        distances = [
            distance_to_the_circle(s_coord[j][0], s_coord[j][1], x0, y0, r) for j in Jcl
        ]
        j2_l = Jcl[np.argmin(distances)]

        # Step 5: Evaluate the residual component
        pivot = (
            line_kernel(t_coord[i2], s_coord[j2_l], kernel_factor, kernel_power)
            - U[i2, 0] * V[0, j2_l]
        )

        # Step 6: If the pivot's absolute value is smaller than the previous pivot, return the previous pivot
        if abs(pivot) <= abs(prev_pivot):
            return i2, j2_l_, prev_pivot

        # Step 7: Update l and the central subset
        prev_pivot = pivot
        l += 1
        j2_l_ = j2_l
        Jcl.remove(j2_l)


def rank3_selection(
    t_coord, s_coord, Ik, Jk, Ic, Jc, U, V, kernel_factor, kernel_power
):
    """
    Rank 3 selection of optimal points based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
    Returns:
        i3, j3 (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a point i_3 that minimizes distance to the conjugate circle
    distances = [np.linalg.norm(t_coord[i] - t_coord[Ik[0]]) for i in Ic]
    i3 = Ic[np.argmin(distances)]

    # Step 2: Initialize l and adjust the central subset of the source cloud
    l = 1
    Jcl = Jc.copy()

    x1 = t_coord[Ik[0]][0]
    y1 = t_coord[Ik[0]][1]
    x2 = s_coord[Jk[0]][0]
    y2 = s_coord[Jk[0]][1]
    x3 = t_coord[Ik[1]][0]
    y3 = t_coord[Ik[1]][1]
    x0, y0, r = find_circle(x1, y1, x2, y2, x3, y3)
    # conjugate circle at x1,y1
    rad = np.array([x1 - x0, y1 - y0])
    tan = np.zeros(2)
    tan[0] = -rad[1]
    tan[1] = rad[0]
    tan /= np.linalg.norm(tan)
    if np.dot(tan, np.array([x2 - x1, y2 - y1])) < 0:
        tan *= -1
    x1ort, y1ort = x1 + r * tan[0], y1 + r * tan[1]
    # conjugate circle at x2,y2
    rad = np.array([x2 - x0, y2 - y0])
    tan[0] = -rad[1]
    tan[1] = rad[0]
    tan /= np.linalg.norm(tan)
    if np.dot(tan, np.array([x1 - x2, y1 - y2])) < 0:
        tan *= -1
    x2ort, y2ort = x2 + r * tan[0], y2 + r * tan[1]

    # Select a point in Ic close to the conjugate circle at x1,y1
    distances = [
        distance_to_the_circle(t_coord[i][0], t_coord[i][1], x1ort, y1ort, r)
        for i in Ic
    ]
    i3 = Ic[np.argmin(distances)]
    j3_l_ = None
    prev_pivot = 0

    while True:
        # Step 3: Select a point y_{j^l_3} that minimizes distance to the conjugate circle at x2,y2
        distances = [
            distance_to_the_circle(s_coord[j][0], s_coord[j][1], x2ort, y2ort, r)
            for j in Jcl
        ]
        j3_l = Jcl[np.argmin(distances)]

        # Step 4: Evaluate the residual component
        pivot = line_kernel(
            t_coord[i3], s_coord[j3_l], kernel_factor, kernel_power
        ) - np.dot(U[i3, :2], V[:2, j3_l])

        # Step 5: Check if the residual condition is met
        if abs(pivot) <= abs(prev_pivot):
            return i3, j3_l_, prev_pivot

        # Step 6: Update l and the central subset
        prev_pivot = pivot
        l += 1
        j3_l_ = j3_l
        Jcl.remove(j3_l)


def higher_ranks_selection(
    t_coord, s_coord, Ic, Jc, U, V, k, kernel_factor, kernel_power
):
    """
    Pivot Choice for Higher Ranks $k \\ge 4$ based on ACA-GP algorithm.

    Args:
        t_coord (numpy.ndarray): The target cloud of points.
        s_coord (numpy.ndarray): The source cloud of points.
        Ik (list): Indices of previously selected rows.
        Jk (list): Indices of previously selected columns.
        Ic (list): Indices of the central subset of the target cloud.
        Jc (list): Indices of the central subset of the source cloud.
        U,V (numpy.ndarray): The U and V matrices of the ACA-GP algorithm.
        k (int): Current rank.
    Returns:
        i_k, j_k (int, int): Indices of the next selected points.
        pivot (float): The pivot value.
    """
    # Step 1: Select a trial point i^t randomly from the target set excluding previously selected points
    Ick = Ic.copy()
    Jck = Jc.copy()
    i_t = np.random.choice(Ick)

    # Step 2: Compute R_{i^t j} for all j in J^c_k
    pivots = [
        line_kernel(t_coord[i_t], s_coord[j], kernel_factor, kernel_power)
        - np.dot(U[i_t, :k], V[:k, j])
        for j in Jck
    ]

    # Step 3: Select the column j_k with the maximal absolute value
    j_k_idx = np.argmax(np.abs(pivots))
    j_k = Jck[j_k_idx]
    pivot = pivots[j_k_idx]

    # Step 4: Compute R_{i j_k} for all i in I^c_k
    pivots = [
        line_kernel(t_coord[i], s_coord[j_k], kernel_factor, kernel_power)
        - np.dot(U[i, :k], V[:k, j_k])
        for i in Ick
    ]

    # Step 5: Select the row i_k with the maximal absolute value
    i_k_idx = np.argmax(np.abs(pivots))  # Get index in Ick
    i_k = Ick[i_k_idx]  # Get actual i_k value
    pivot = pivots[i_k_idx]  # Get corresponding pivot value

    return i_k, j_k, pivot

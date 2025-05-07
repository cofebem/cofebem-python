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
            # If the matrix is nearly zero, return empty factors.
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

from cofebem.bem.construct_S import FenicsLE
from cofebem.contact.lcp_solvers.lemke import lemkelcp
from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import time

import sys

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "semisphere.msh", MPI.COMM_WORLD, 0, gdim=3
)


# Initialize FenicsLE
fenics_le = FenicsLE(mesh=mesh, E=1.0e9, nu=0.3)

# Define boundary condition selector
selector_tol = 0.06


# Define boundary condition selector
def boundary_selector1(x):
    return np.isclose(x[2], 0, atol=1e-5)


# Add Dirichlet boundary condition
fenics_le.add_dirichlet_bc(value=np.array([0.0, 0.0, 0.0]), locator=boundary_selector1)


# Define boundary condition selector
def boundary_selector2(x):
    return (0.9 <= x[2]) & (x[2] <= 1)


# # Define boundary condition selector
# def boundary_selector2(x):
#     return (0.9 <= x[2]) & (x[2] <= 1)


# # Add Dirichlet boundary condition
# fenics_le.add_neumann_bc(
#     value=np.array([0.0, 0.0, -10.0]), locator=boundary_selector2, marker_id=1
# )


# Set up and solve the problem
fenics_le.setup()
fenics_le.solve()

# Solution
uh = fenics_le.get_solution()

# Visualize
# fenics_le.visualize()


# Define boundary condition selector
def boundary_selector3(x):
    return (1 - selector_tol <= x[0] ** 2 + x[1] ** 2 + x[2] ** 2) & (
        x[0] ** 2 + x[1] ** 2 + x[2] ** 2 <= 1 + selector_tol
    )


S, coords, _ = fenics_le.compute_S(
    selector=boundary_selector3, force_magnitude=-1e8, method="bruteforce"
)

print(S.shape)


def _flat_indenter(x, y, x0, y0, R, z0):
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < R:
        return z0
    else:
        return z0 + 10.0


flat_indenter = np.vectorize(_flat_indenter)


def _parabolic_indenter(x, y, x0, y0, R, z0):
    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > R:
        return z0 + R
    else:
        return z0 + R - np.sqrt(R**2 - (x - x0) ** 2 - (y - y0) ** 2)
        # return z0 + (x-x0)**2/(2*R) + (y-y0)**2/(2*R)


parabolic_indenter = np.vectorize(_parabolic_indenter)


# Constrained CG python
def constrained_CG(
    S,
    error_type,
    gap,
    max_iter,
    tolerance,
    pressure_factor=1e12,
    initial_pressure=None,
):
    error_history = np.zeros((max_iter, 3))
    ub = -gap
    # Warmed start does not work well
    if initial_pressure is not None:
        # p = initial_pressure
        # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
        # p[gap>0] = 0
        p = np.maximum(-gap, 0) * pressure_factor
    else:
        p = np.zeros_like(ub)
        p = np.maximum(-gap, 0) * pressure_factor

    w = np.inner(S, p) - ub
    # w -= np.mean(w) #new
    t = w
    t_ = np.zeros_like(w)
    d = 0
    error = 1
    error_ = 1
    for iter in range(max_iter):
        if iter > 0:
            t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
            t[p <= 0] = 0
        q = np.inner(S, t)
        tau = np.inner(w, t) / np.inner(t, q)
        p = p - tau * t
        p = np.maximum(p, 0)
        zero_pressure = np.where(p == 0)[0]
        penetration = np.where(w < 0)[0]
        set_I = np.intersect1d(zero_pressure, penetration)
        if len(set_I) == 0:
            d = 1
        else:
            d = 0
            p[set_I] -= tau * w[set_I]
        t_ = t

        w = np.inner(S, p) - ub
        nw = np.linalg.norm(w, 2)

        error_ = error
        displ_error = np.linalg.norm(w[p > 0], 2) / nw
        ort = np.abs(np.dot(w, p) / nw)

        if error_type == "displacement":
            error = displ_error
        elif error_type == "mix":
            error = np.sqrt(displ_error * ort)
        elif error_type == "nw":
            error = nw
            if abs((error - error_) / error_) < tolerance:
                error_history[iter, 0] = displ_error
                error_history[iter, 1] = abs((error - error_) / error_)
                error_history[iter, 2] = ort
                return p, np.inner(S, p), error_history[: iter + 1]
        error_history[iter, 0] = displ_error
        error_history[iter, 1] = error
        error_history[iter, 2] = ort
        if error < tolerance:
            break
    return p, np.inner(S, p), error_history[: iter + 1]


tri = Triangulation(coords[:, 0], coords[:, 1])

# Vertical penetration of the indenter
displ = 0.4
# Indenter radius
Rindenter = 0.4

# Solve the problem
max_iter = 1000
tolerance = 1e-5
error_type = "nw"
# pfactor is factor linking the trial pressure to the initial penetration for warmed-up start of the CG
pfactor = 1e8
# Number of frames for the animation
Nframes = 10

# # Example with animation
# ANIMATION = True
# if ANIMATION == True:
#     x_center = np.linspace(-0.75, 0.75, Nframes)
#     for frame, xc in enumerate(x_center):
#         contact_center = np.array([xc, 0.0])
#         # Uncomment indenter type
#         # gap = flat_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]
#         # gap = conical_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]
#         gap = (
#             parabolic_indenter(
#                 coords[:, 0],
#                 coords[:, 1],
#                 contact_center[0],
#                 contact_center[1],
#                 Rindenter,
#                 np.ones_like(coords[:, 2]) - displ,
#             )
#             - coords[:, 2]
#         )
#         penetrating_nodes = np.where(gap < 0)[0]

#         # Solve the problem
#         start = time.time()
#         # pfactor = 1. # for conical indenter
#         if frame == 0:
#             pressure, displacement, error_history = constrained_CG(
#                 S, error_type, gap, max_iter, tolerance, pfactor
#             )
#         else:
#             pressure, displacement, error_history = constrained_CG(
#                 S,
#                 error_type,
#                 gap,
#                 max_iter,
#                 tolerance,
#                 pfactor,
#                 pressure,
#             )

#         X_, Y_, Z_ = coords[:, 0], coords[:, 1], coords[:, 2] - displacement
#         X_ = X_.reshape(-1, 1)
#         Y_ = Y_.reshape(-1, 1)
#         Z_ = Z_.reshape(-1, 1)
#         disp_ = displacement.reshape(-1, 1)
#         output = np.hstack((X_, Y_, Z_, disp_))
#         # np.savetxt(
#         #     fname=f"output_{frame}.csv",
#         #     X=output,
#         #     header="X, Y, Z, disp",
#         #     comments="",
#         #     delimiter=",",
#         # )
#         print(
#             "Iters: {0:3d}, Error {1:.3e}".format(
#                 len(error_history), error_history[-1, 1]
#             )
#         )


# Example with animation
ANIMATION = True
if ANIMATION == True:
    x_center = np.linspace(-0.75, 0.75, Nframes)
    for frame, xc in enumerate(x_center):
        contact_center = np.array([xc, 0.0])
        # Uncomment indenter type
        # gap = flat_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]
        # gap = conical_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]
        gap = (
            parabolic_indenter(
                coords[:, 0],
                coords[:, 1],
                contact_center[0],
                contact_center[1],
                Rindenter,
                np.ones_like(coords[:, 2]) - displ,
            )
            - coords[:, 2]
        )
        penetrating_nodes = np.where(gap < 0)[0]

        # Solve the problem
        start = time.time()
        # pfactor = 1. # for conical indenter
        if frame == 0:
            pressure, _, exit_ = lemkelcp(
                S,
                gap,
                max_iter,
            )
        else:
            pressure, _, exit_ = lemkelcp(
                S,
                gap,
                max_iter,
            )

        end = time.time()
        it_time = end - start
        displacement = S @ pressure
        X_, Y_, Z_ = coords[:, 0], coords[:, 1], coords[:, 2] - displacement
        X_ = X_.reshape(-1, 1)
        Y_ = Y_.reshape(-1, 1)
        Z_ = Z_.reshape(-1, 1)
        disp_ = displacement.reshape(-1, 1)
        output = np.hstack((X_, Y_, Z_, disp_))
        np.savetxt(
            fname=f"output_{frame}.csv",
            X=output,
            header="X, Y, Z, disp",
            comments="",
            delimiter=",",
        )
        print(f"exit = {exit_} \n it_time = {it_time}")

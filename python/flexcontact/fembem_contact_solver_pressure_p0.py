"""
Code that solves contact problem using constructed BEM matrix
for p0 pressure interpolation

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL, Centre des Matériaux)
Date: Nov 2025
License: BSD 3-Clause
"""

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from numba import jit, prange
import time
from scipy.optimize import nnls
import pyvista as pv
from scipy.spatial import Delaunay, QhullError

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

def zoom_to_data(self, my_bounds):
    if not self.camera_set:
        self.view_isometric()
    self.reset_camera(bounds=my_bounds)
    self.camera_set = True
    self.reset_camera_clipping_range()

@jit(nopython=True, parallel=True)
def back_to_grid(x,y,z,xp,yp,dx,dy):
    Z = np.zeros((len(yp),len(xp)))
    for i in prange(len(x)):
        ix = round((x[i] - xp[0])/dx)
        iy = round((y[i] - yp[0])/dy)
        if Z[iy,ix] == 0:
            Z[iy,ix] = z[i]
        else:
            Z[iy,ix] = (Z[iy,ix] + z[i])/2
            print("Warning: multiple points in the same cell, ix = ", ix, "iy = ", iy, " x = ", x[i]-xp[0], " y = ", y[i])
    return Z

def indenter(x,y,x0,y0,R,z0):
    return z0 + (x-x0)**2/(2*R) + (y-y0)**2/(2*R)

def _parabolic_indenter(x,y,x0,y0,R,z0):
    if np.sqrt((x-x0)**2 + (y-y0)**2) > R:
        return z0 + R
    else:
        return z0 + R - np.sqrt(R**2 - (x-x0)**2 - (y-y0)**2)
        # return z0 + (x-x0)**2/(2*R) + (y-y0)**2/(2*R)

parabolic_indenter = np.vectorize(_parabolic_indenter)

def conical_indenter(x,y,x0,y0,R,z0):
    return z0 + np.sqrt((x-x0)**2 + (y-y0)**2)/R

def _flat_indenter(x,y,x0,y0,R,z0):
    if np.sqrt((x-x0)**2 + (y-y0)**2) < R:
        return z0
    else:
        return z0 + 10.

flat_indenter = np.vectorize(_flat_indenter)

def ids_to_indices(node_of_elem_ids, nodal_ids):
    """
    node_of_elem_ids : list[np.ndarray]  # IDs per element (your current object)
    nodal_ids        : (n,) array of node IDs aligned with g, K rows
    returns          : list[np.ndarray] with 0..n-1 indices
    """
    id2idx = {int(nid): i for i, nid in enumerate(nodal_ids)}
    node_of_elem_idx = []
    for ids in node_of_elem_ids:
        idxs = np.fromiter((id2idx[int(x)] for x in ids), dtype=int, count=len(ids))
        node_of_elem_idx.append(idxs)
    return node_of_elem_idx



def build_facet_projection_matrix(nodal_coords, facet_centers):
    """
    Construct a matrix that maps nodal quantities to facet-center quantities
    using barycentric interpolation on the (x, y) plane. Each row contains the
    interpolation weights for a single facet center.
    """
    coords_2d = nodal_coords[:, :2]
    centers_2d = facet_centers[:, :2]
    n_facets = centers_2d.shape[0]
    n_nodes = coords_2d.shape[0]
    projection = np.zeros((n_facets, n_nodes))

    try:
        delaunay = Delaunay(coords_2d)
        simplices = delaunay.find_simplex(centers_2d)
    except QhullError:
        delaunay = None
        simplices = np.full(n_facets, -1, dtype=int)

    for idx, simplex in enumerate(simplices):
        if simplex == -1 or delaunay is None:
            nearest = np.argmin(np.linalg.norm(coords_2d - centers_2d[idx], axis=1))
            projection[idx, nearest] = 1.0
            continue

        vertices = delaunay.simplices[simplex]
        coords = coords_2d[vertices]
        A = np.vstack((coords.T, np.ones(coords.shape[0])))
        b = np.hstack((centers_2d[idx], [1.0]))
        try:
            weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            nearest = np.argmin(np.linalg.norm(coords_2d - centers_2d[idx], axis=1))
            projection[idx, nearest] = 1.0
            continue

        weights = np.clip(weights, 0.0, None)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            nearest = np.argmin(np.linalg.norm(coords_2d - centers_2d[idx], axis=1))
            projection[idx, nearest] = 1.0
        else:
            projection[idx, vertices] = weights / weight_sum

    return projection


def build_node_of_elem(facet_centers, nodal_coords, nodal_ids, k=4):
    """
    For each quad facet, find its 4 nodes by nearest-neighbour search.

    Inputs
    ------
    facet_centers : (m, 3) array
        Coordinates of facet centers
    nodal_coords  : (n, 3) array
        Coordinates of all nodes
    nodal_ids     : (n,) int array
        Node IDs (global numbering)
    k : int
        Number of nearest nodes to return (4 for quads)

    Output
    ------
    node_of_elem : list of length m
        node_of_elem[e] is an array of node IDs attached to facet e,
        sorted by distance.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(nodal_coords)

    node_of_elem = []
    for center in facet_centers:
        dist, idx = tree.query(center, k=k)
        node_ids = nodal_ids[idx]
        node_of_elem.append(node_ids)

    return node_of_elem

def plot_results(displacement, pressure, facet_centers, nodal_coords, tri_facet, tri_nodal, contact_center, Rindenter, displ):
    # Set viewing angles
    elevation_angle = 60  # Lower number to "raise" the camera view
    azimuth_angle = -60  # Adjust as needed

    plt.rcParams['figure.figsize'] = [10,5]
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

    ax[0].view_init(elev=elevation_angle, azim=azimuth_angle)
    ax[1].view_init(elev=elevation_angle, azim=azimuth_angle)

    ax[0].set_xlim([-0.,1.])
    ax[0].set_ylim([-0.,1.])
    # ax[0].set_zlim([-displ,0])
    x = np.linspace(-0.,1.,100)
    y = np.linspace(-0.,1.,100)
    X,Y = np.meshgrid(x,y)
    Z = parabolic_indenter(X,Y, contact_center[0], contact_center[1], Rindenter, displ) - nodal_coords[0,2]
    Z[Z>0.] = np.nan
    surf1 = ax[0].plot_trisurf(nodal_coords[:,0], nodal_coords[:,1], -displacement, triangles=tri_nodal.triangles, cmap='coolwarm') #, vmin = -displ, vmax = 0)
    cb1 = fig.colorbar(surf1, ax=ax[0], shrink=0.6, aspect=10, orientation='horizontal')
    cb1.set_label("$u_z$")
    # ax[0].plot_surface(X,Y,Z, alpha=0.1, cmap='gray',  rcount = X.shape[0], ccount = X.shape[1], edgecolor='k', linewidth=0.1)
    ax[0].set_title("Vertical displacement")

    ax[1].set_xlim([0,1])
    ax[1].set_ylim([0,1])
    # ax[1].set_zlim([0,16])
    surf2 = ax[1].plot_trisurf(facet_centers[:,0], facet_centers[:,1], pressure, triangles=tri_facet.triangles, cmap='coolwarm') #, vmin = 0, vmax = 1.5e6)
    cb2 = fig.colorbar(surf2, ax=ax[1], shrink=0.6, aspect=10, orientation='horizontal')
    cb2.set_label("$p/E$")
    ax[1].set_title("Pressure/Young's modulus")
    fig.tight_layout()
    fig.savefig("Contact_parabolic_indenter.png", dpi=300)

def plot_gaps(gap, g, tri_nodal):
    vmin = min(gap.min(), g.min())
    vmax = max(gap.max(), g.max())
    fig_gap, axes_gap = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    contour_init = axes_gap[0].tricontourf(tri_nodal, gap, levels=50, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes_gap[0].set_title("Initial gap")
    axes_gap[0].set_xlabel("x")
    axes_gap[0].set_ylabel("y")
    axes_gap[0].set_aspect("equal")

    contour_final = axes_gap[1].tricontourf(tri_nodal, g, levels=50, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes_gap[1].set_title("Final gap")
    axes_gap[1].set_xlabel("x")
    axes_gap[1].set_ylabel("y")
    axes_gap[1].set_aspect("equal")

    cbar = fig_gap.colorbar(contour_final, ax=axes_gap.ravel().tolist(), shrink=0.8)
    cbar.set_label("gap")

    fig_gap.savefig(f"gap_comparison.png", dpi=300)


def main():
    # The stored BEM matrix is located in the file FlexData_{N}x{N}.npz
    N = 21
    fname = "../out_elasticity/FlexData_{0}x{0}.npz".format(N)
    data = np.load(fname)
    K = data["K"]
    M = np.asarray(data["M"],dtype=float)
    facet_centers = data["facet_centers"]
    nodal_coords = data["boundary_coords"]

    n, mfac = K.shape
    assert M.shape[0] == n

    # Vertical penetration of the indenter
    displ = 0.2
    # Indenter radius
    Rindenter = 1.


    contact_center = np.array([0.5, 0.5])
    x0 = contact_center[0]
    y0 = contact_center[1]
    R = Rindenter
    z0 = nodal_coords[0,2] - displ
    gap = parabolic_indenter(nodal_coords[:,0], nodal_coords[:,1], x0, y0, R, z0) \
        - nodal_coords[:,2]

    # Solve the problem
    from py_solvers import solve_contact_qp

    pressure = solve_contact_qp(K, M, gap, lam=1e-6, backend="quadprog")
    displacement =  K @ pressure
    g = gap - displacement

    print("displacement in ", np.min(displacement), np.max(displacement))
    print("pressure in ", np.min(pressure), np.max(pressure))

    # Plot the initial and final gaps on a single figure, sharing the color scale
    PLOT = True
    if PLOT:
        tri_facet = Triangulation(facet_centers[:,0], facet_centers[:,1])
        tri_nodal = Triangulation(nodal_coords[:,0], nodal_coords[:,1])
        plot_gaps(gap, g, tri_nodal)

        plot_results(displacement, pressure, facet_centers, nodal_coords, tri_facet, tri_nodal, contact_center, Rindenter, displ)

if __name__ == "__main__":
    main()
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


# Constrained CG python
def constrained_CG(K, error_type, coord, dofs, gap, max_iter, tolerance,
                   initial_pressure=None, projection_matrix=None, facet_centers=None):
    num_nodes, num_facets = K.shape

    if initial_pressure is None or initial_pressure.shape[0] != num_facets:
        raise ValueError(
            "Initial pressure must be provided with length equal to the number of facets ({0}).".format(num_facets)
        )
    p = initial_pressure.copy()

    if projection_matrix is None:
        if facet_centers is None:
            raise ValueError("facet_centers must be provided when projection_matrix is not supplied.")
        projection_matrix = build_facet_projection_matrix(coord, facet_centers)

    if projection_matrix.shape != (num_facets, num_nodes):
        raise ValueError(
            "Projection matrix must have shape ({0}, {1}). Received {2}.".format(
                num_facets, num_nodes, projection_matrix.shape
            )
        )

    K_local = projection_matrix @ K
    gap_local = projection_matrix @ gap

    error_history = np.zeros((max_iter, 3))
    ub = -gap_local

    w = K_local @ p - ub
    t = w.copy()
    t_ = np.zeros_like(w)
    d = 0
    error = 1.0
    error_ = 1.0
    for iter in range(max_iter):
        if iter > 0:
            active = p > 0
            t[active] = w[active] + d * error / error_ * t_[active]
            t[~active] = 0
        q = K_local @ t
        denom = np.dot(t, q)
        if denom <= 0:
            denom = np.dot(t, t)
            if denom <= 0:
                denom = 1.0
        tau = np.dot(w, t) / denom
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
            p = np.maximum(p, 0)
        t_ = t
        w = K_local @ p - ub
        nw = np.linalg.norm(w, 2)

        error_ = error
        if nw > 0:
            displ_error = np.linalg.norm(w[p > 0], 2) / nw
            ort = np.abs(np.dot(w, p) / nw)
        else:
            displ_error = 0.0
            ort = 0.0

        if error_type == "displacement":
            error = displ_error
        elif error_type == "mix":
            error = np.sqrt(displ_error * ort)
        elif error_type == "nw":
            error = nw
            denominator = abs(error_) if error_ != 0 else 1.0
            if abs(error - error_) / denominator < tolerance:
                error_history[iter, 0] = displ_error
                error_history[iter, 1] = abs(error - error_) / denominator
                error_history[iter, 2] = ort
                return p, K @ p, error_history[:iter + 1]
        error_history[iter, 0] = displ_error
        error_history[iter, 1] = error
        error_history[iter, 2] = ort
        if error < tolerance:
            break
    return p, K @ p, error_history[:iter + 1]

def main():
    # The stored BEM matrix is located in the file FlexData.npz
    fname = "../out_elasticity/FlexData_11x11.npz"
    data = np.load(fname)

    K, facet_ids, facet_centers, nodal_dofs, nodal_coords = data["K"],data["facet_ids"],data["facet_centers"],data["boundary_dofs"],data["boundary_coords"]

    tri_facet = Triangulation(facet_centers[:,0], facet_centers[:,1])
    tri_nodal = Triangulation(nodal_coords[:,0], nodal_coords[:,1])
    facet_projection = build_facet_projection_matrix(nodal_coords, facet_centers)

    # Vertical penetration of the indenter
    displ = 0.15
    # Indenter radius
    Rindenter = 1.

    # Solve the problem
    max_iter = 100
    tolerance = 1e-5
    error_type = "nw"
    # pfactor is factor linking the trial pressure to the initial penetration for warmed-up start of the CG
    E = 1.0e9
    nu = 0.3
    E_star = E / (1 - nu**2)
    pfactor = E_star/100.
    # Number of frames for the animation
    Nframes = 1

    # Example with animation
    ANIMATION = True
    if ANIMATION == True:
        x_center = np.linspace(-0.,1.,Nframes)
        # x_center = np.linspace(0.5,0.5,1)
        for frame, xc in enumerate(x_center):
            contact_center = np.array([xc, 0.5])
            # Uncomment indenter type
            # gap = flat_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]
            # gap = conical_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]
            gap = parabolic_indenter(nodal_coords[:,0], nodal_coords[:,1], contact_center[0], contact_center[1], Rindenter, nodal_coords[0,2]-displ) - nodal_coords[:,2]
            penetrating_nodes = np.where(gap < 0)[0]

            # Solve the problem
            start = time.time()
            # pfactor = 1. # for conical indenter
            if frame == 0:
                p_trial_nodes = pfactor * np.maximum(-gap, 0)
                p_trial_facet = facet_projection @ p_trial_nodes
                p_trial_facet = np.maximum(p_trial_facet, 0)
                pressure, displacement, error_history = constrained_CG(
                    K,
                    error_type,
                    nodal_coords,
                    nodal_dofs,
                    gap,
                    max_iter,
                    tolerance,
                    initial_pressure=p_trial_facet,
                    projection_matrix=facet_projection
                )
            else:
                pressure, displacement, error_history = constrained_CG(
                    K,
                    error_type,
                    nodal_coords,
                    nodal_dofs,
                    gap,
                    max_iter,
                    tolerance,
                    initial_pressure=pressure,
                    projection_matrix=facet_projection
                )
            print("Iters: {0:3d}, Error {1:.3e}".format(len(error_history), error_history[-1,1]))

    ## Plot using Matplotlib

            # Set viewing angles
            elevation_angle = 60  # Lower number to "raise" the camera view
            azimuth_angle = -60  # Adjust as needed

            plt.rcParams['figure.figsize'] = [10,5]
            fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})

            ax[0].view_init(elev=elevation_angle, azim=azimuth_angle)
            ax[1].view_init(elev=elevation_angle, azim=azimuth_angle)

            ax[0].set_xlim([-0.,1.])
            ax[0].set_ylim([-0.,1.])
            ax[0].set_zlim([-displ,0])
            x = np.linspace(-0.,1.,100)
            y = np.linspace(-0.,1.,100)
            X,Y = np.meshgrid(x,y)
            # Z = parabolic_indenter(X,Y, contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[0,2]
            # Z = conical_indenter(X,Y, contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[0,2]
            Z = flat_indenter(X,Y, contact_center[0], contact_center[1], Rindenter, nodal_coords[0,2]-displ) - nodal_coords[0,2]
            Z[Z>0.] = np.nan
            surf1 = ax[0].plot_trisurf(nodal_coords[:,0], nodal_coords[:,1], -displacement, triangles=tri_nodal.triangles, cmap='coolwarm', vmin = -displ, vmax = 0)
            cb1 = fig.colorbar(surf1, ax=ax[0], shrink=0.6, aspect=10, orientation='horizontal')
            cb1.set_label("$u_z$")
            ax[0].plot_surface(X,Y,Z, alpha=0.1, cmap='gray',  rcount = X.shape[0], ccount = X.shape[1], edgecolor='k', linewidth=0.1)
            ax[0].set_title("Vertical displacement")

            ax[1].set_xlim([0,1])
            ax[1].set_ylim([0,1])
            # ax[1].set_zlim([0,16])
            surf2 = ax[1].plot_trisurf(facet_centers[:,0], facet_centers[:,1], pressure, triangles=tri_facet.triangles, cmap='coolwarm', vmin = 0, vmax = 1.5e6)
            cb2 = fig.colorbar(surf2, ax=ax[1], shrink=0.6, aspect=10, orientation='horizontal')
            cb2.set_label("$p/E$")
            ax[1].set_title("Pressure/Young's modulus")
            fig.tight_layout()
            fig.savefig("Contact_cone_{0:03d}.png".format(frame), dpi=300)

    # Plot with Pyvista 
            # # Create a PyVista mesh for the deformed surface
            # Config = np.zeros_like(coords)
            # Config[:,0] = coords[:,0]
            # Config[:,1] = coords[:,1]
            # Config[:,2] = -displacement
            # cloud = pv.PolyData(Config)
            # cloud["z"] = -displacement  # Add displacement as a scalar array
            # surf = cloud.delaunay_2d(alpha=0.1)
            
            # Nr = 10
            # dx = 0.6*R/Nr
            # Ntheta = 30
            # Ind = np.zeros((Nr*Ntheta, 3))
            # for ir in range(Nr):
            #     r = ir*dx
            #     for it in range(Ntheta):                
            #         dt = 2*np.pi/Ntheta
            #         theta = it*dt
            #         Ind[it+ir*Ntheta,0] = contact_center[0] + r*np.cos(theta)
            #         Ind[it+ir*Ntheta,1] = contact_center[1] + r*np.sin(theta)
            #         Ind[it+ir*Ntheta,2] = parabolic_indenter(Ind[it+ir*Ntheta,0], Ind[it+ir*Ntheta,1], contact_center[0], contact_center[1], R, coords[0,2]-displ) - coords[0,2] + 1e-3 # - displ + 1e-3
            # cloud2 = pv.PolyData(Ind)
            # cloud2["z"] = Ind[:,2]*0  # Add displacement as a scalar array
            # surf2 = cloud2.delaunay_2d(alpha=0.1)

            # # Create a plotter object
            # plotter = pv.Plotter(off_screen=True)

            # # Add the indenter
            # plotter.add_mesh(surf2, scalars="z", cmap='grey', opacity=0.25, show_edges=True, clim=[-0.2, 0])
            # # Add the deformed surface to the plotter
            # plotter.add_mesh(surf, scalars="z", cmap='coolwarm', clim=[-0.2, 0])

            # # zoom_to_data(plotter, my_bounds=[-0.25,1.25,-0.25,1.25,-0.2,0])
            # zoom_to_data(plotter, my_bounds=[0,1,0,1,-0.25,0.25])

            # # Show the plot
            # # plotter.show()
            # # Save plotter file
            # plotter.screenshot("Contact_displ_{0:03d}.png".format(frame), transparent_background=True)
            # plotter.close()

            # # Pyvista output for pressure
            # plotter = pv.Plotter(off_screen=True)

            # # Create a PyVista mesh for the pressure
            # Config = np.zeros_like(coords)
            # Config[:,0] = coords[:,0]
            # Config[:,1] = coords[:,1]
            # Config[:,2] = pressure*np.pi*Es/2e6
            # cloud = pv.PolyData(Config)
            # cloud["z"] = pressure*np.pi*Es/2e6  # Add displacement as a scalar array
            # surf = cloud.delaunay_2d(alpha=0.1)        

            # # Add the deformed surface to the plotter
            # plotter.add_mesh(surf, scalars="z", cmap='coolwarm', clim=[0, 4/20])

            # zoom_to_data(plotter, my_bounds=[0,1,0,1,-0.25,0.25])

            # # Show the plot
            # # plotter.show()
            # # Save plotter file
            # plotter.screenshot("Contact_pressure_{0:03d}.png".format(frame), transparent_background=True)


if __name__ == "__main__":
    main()
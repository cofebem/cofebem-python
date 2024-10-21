from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from numba import jit, prange
import time
from scipy.optimize import nnls
import pyvista as pv

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

fname = "out_elasticity/FlexData.npz"
data = np.load(fname)
K, coords, dofs = data["K"],data["coords"],data["dofs"]
n = np.sqrt(K.shape[0]).astype(int)

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

# Python implementation of the constrained CG algorithm
# @jit(npython=True, paralle=True)
def inner_numba(A,b):
    shape = A.shape
    if len(shape) == 1:
        n = shape[0]
        if n != b.shape[0]:
            raise ValueError("inner_numba: vectors must have compatible dimensions.")
        else:
            result = 0
            for i in prange(n):
                result += A[i]*b[i]
            return result
    elif len(shape) == 2:
        print("Matrix vector product")
        n = shape[0]
        m = shape[1]
        if n != b.shape[0]:
            raise ValueError("inner_numba: matrix and vector must have compatible dimensions.")
        else:
            result = np.zeros(m)
            for i in prange(n):
                result[i] += inner_numba(A[i],b)
            return result
    else:
        raise ValueError("inner_numba is implemented only for matrices and vectors.")    

# @jit(nopython=True, parallel=True)
def constrained_CG_numba(K, coord, dofs, gap, max_iter, tolerance):
    ub = -gap
    p = np.zeros_like(ub)
    p = np.maximum(-gap, 0)*1e5
    w = inner_numba(K, p) - ub
    nw  = 1
    nw_ = 1
    t  = w
    t_ = np.zeros_like(w)
    d = 0
    error = 1e5
    iter_stop = 0
    for iter in range(max_iter):
        if iter > 0:
            t = w + d * nw/nw_ * t_
        q = inner_numba(K, t)
        tau = inner_numba(w, t) / inner_numba(t, q)
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
        nw_ = nw
        w = inner_numba(K, p) - ub
        nw = np.linalg.norm(w, 2)
        error = abs((nw - nw_ )/nw_)
        # print("{0:d} | {1:.3f} | {1:.3f}".format(iter, nw, error))
        iter_stop = iter
        if error < tolerance:
            break
    return p,inner_numba(K,p),iter_stop,error

# Non-negative least squares method
def NNLS_method(K, coord, dofs, gap, max_iter, tolerance, pressure_factor=1e5):

    p, _ = nnls(K, -gap)
    w = np.inner(K, p) - gap
    # w = np.maximum(w, 0)  
    return p, w, [0]

    # error_history = np.zeros(max_iter)
    # ub = -gap
    # init = False
    # finished = False
    # p = np.maximum(-gap, 0) * pressure_factor
    # print(np.sum(p))
    # for iter in range(max_iter):
    #     I = np.where(p > 0)[0]
    #     print("// init I: ", len(I))
    #     if len(I) == 0:
    #         init = True
    #     step_3 = True

    #     for inner_iter in range(10):
    #         if step_3:
    #             w = np.inner(K, p) - ub
    #             wn = np.linalg.norm(w, 2)
    #             error_history[iter] = wn
    #             print("     Inner loop: ", inner_iter, "Error: ", wn)
    #             if wn < tolerance or (init and len(I) == len(p)):
    #                 finished = True
    #                 break
    #             if init == True:                
    #                 not_I = np.where(p <= 0)[0]
    #                 min_index = not_I[np.argmin(w[not_I])]
    #                 print("Min index:", min_index)
    #                 print("Before adding min_index I: ", len(I))
    #                 I = np.append(I, min_index)
    #                 print("After adding min_index I: ", len(I))
    #             else:
    #                 init = True

    #         print("// After step_3 I: ", len(I))
    #         K_I = K[I,:]
    #         K_I = K_I[:,I]
    #         ubI = ub[I]

    #         sI = np.linalg.solve(K_I, ubI)
    #         s = np.zeros_like(p)
    #         s[I] = sI
    #         sn = np.linalg.norm(s, 2)
    #         p = s
    #         if sn < tolerance:            
    #             break
    #         else:
    #             j = np.argmin(s[s<0])
    #             print("J: ", j)
    #             print("s[j]: ", s[j])
    #             p = p + p[j]/(p[j] - s[j]) * (s - p)

    #             print("S=", s)
    #             plt.imshow(p.reshape(n,n), cmap='coolwarm')
    #             plt.show()

    #             print("// Before step_4 I: ", len(I))
    #             I = np.where(p > 0)[0]
    #             print("// After step_4 I: ", len(I))
    #             step_3 = False
    #     if finished:
    #         break
    
    # return p, w + ub, error_history[:iter+1]

# Constrained CG python
def constrained_CG(K, coord, dofs, gap, max_iter, tolerance, pressure_factor=100, initial_pressure=None):
    error_history = np.zeros((max_iter,3))
    ub = -gap
    print(" {0:10s}   {1:10s}   {2:10s}  {3:10s}".format("Iteration", "Error sqrt(R1*R2)", "Displ. Error, R1 ", "Orthogonality, R2"))
    # Warmed start does not work well
    if initial_pressure is not None:        
        # p = initial_pressure
        # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
        # p[gap>0] = 0
        p = np.maximum(-gap, 0) * pressure_factor
    else:
        p = np.zeros_like(ub)
        p = np.maximum(-gap, 0) * pressure_factor

    w = np.inner(K, p) - ub
    # w -= np.mean(w) #new
    t  = w
    t_ = np.zeros_like(w)
    d = 0
    error  = 1
    error_ = 1
    for iter in range(max_iter):
        if iter > 0:
            t[p>0] = w[p>0] + d * error/error_ * t_[p>0]
            t[p<=0] = 0
        q = np.inner(K, t)
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

        w = np.inner(K, p) - ub
        nw = np.linalg.norm(w, 2)

        error_ = error
        displ_error = np.linalg.norm(w[p>0], 2) / nw
        ort = np.abs(np.dot(w,p)/nw) 
        error = np.sqrt(displ_error * ort) # The most optimal according to my tests
                                           # The most optimal is displ_error, but geom. average is more stable

        print("   {0:5d}     {1:.3e}     {2:.3e}     {3:.3e}".format(iter, error, displ_error, ort))
        error_history[iter,0] = displ_error
        error_history[iter,1] = error
        error_history[iter,2] = ort
        if error < tolerance:
            break
    return p,np.inner(K,p),error_history[:iter+1]

# Check initial penetration
E = 1.0e9
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
Es = E/(1-nu**2)
Lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

tri = Triangulation(coords[:,0], coords[:,1])

displ = 0.3
# contact_center = np.array([1., 0.5])
# Rindenter = 0.3
# gap = flat_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]

# contact_center = np.array([.5, 0.5])
# Rindenter = 5.
# gap = conical_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]

contact_center = np.array([-0.35, 0.5])
Rindenter = 1.
gap = parabolic_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], Rindenter, coords[0,2]-displ) - coords[:,2]

penetrating_nodes = np.where(gap < 0)[0]


# Solve the problem
max_iter = 100
tolerance = 1e-6
start = time.time()
# FIXME: clearly need to look in the convergence of this constrained CG algorithm, I can suppose that something was wrongly coded.
pressure, displacement, error_history = constrained_CG(K, coords, dofs, gap, max_iter, tolerance, 100) #e-3)

# Use NNLS method
from scipy.optimize import lsq_linear
# Check https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html#scipy.optimize.nnls
# pressure, error = nnls(K, -gap, maxiter=10000)

# pressure = lsq_linear(K, -gap, bounds=(0, np.inf),verbose=2).x #, tol=1e-12).x #, method='bvls').x
# displacement = np.inner(K, pressure) 
# print("NNLS Error: ", error)
# It seems not to work...




# print("Number of iterations:", len(error_history))
# print("Error:", error_history[-1])

# Set figure size
plt.rcParams['figure.figsize'] = [5,8]
fig,ax = plt.subplots()
plt.yscale("log")
plt.ylim([1e-7,1e2])
plt.xlim([0,50])
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Errors")
# Displ. error
# plt.plot(range(len(error_history)), error_history[:,0],"-", color="k", lw=2, label="Displ. error, $E_d$")
# plt.plot(range(len(error_history)), error_history[:,2],"-.", color="navy",label="Orthogonality, $E_o$")
# Ort. error
# plt.plot(range(len(error_history)), error_history[:,0],"-", color="g", label="Displ. error, $E_d$")
# plt.plot(range(len(error_history)), error_history[:,2],"-.", color="k", lw=2, label="Orthogonality, $E_o$")
# Nw. error
# plt.plot(range(len(error_history)), error_history[:,1],"-", color="k", lw=2, label="Error, $\|w\|_2$")
# plt.plot(range(len(error_history)), error_history[:,2],"-.", color="navy", label="Orthogonality, $E_o$")
# Geom. av. error
plt.plot(range(len(error_history)), error_history[:,0],"--", color="g", label="Displ. error, $E_d$")
plt.plot(range(len(error_history)), error_history[:,1],"-", lw=2, color="k",label="Error, $\\sqrt{E_d E_o}$")
plt.plot(range(len(error_history)), error_history[:,2],"-.", color="navy",label="Orthogonality, $E_o$")
plt.legend(loc="best")
# fig.savefig("Convergence_geom_av_error.pdf")
# fig.savefig("Convergence_disp_error.pdf")
# fig.savefig("Convergence_ort_error.pdf")
# fig.savefig("Convergence_nw_error.pdf")
plt.show()

plt.rcParams['figure.figsize'] = [10,5]
fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
ax[0].set_xlim([0,1])
ax[0].set_ylim([0,1])
ax[0].set_zlim([-0.2,0])
ax[0].plot_trisurf(coords[:,0], coords[:,1], -displacement, triangles=tri.triangles, cmap='coolwarm')

ax[1].set_xlim([0,1])
ax[1].set_ylim([0,1])
# ax[1].set_zlim([0,1e-4*np.pi*Es])
ax[1].plot_trisurf(coords[:,0], coords[:,1], pressure*np.pi*Es, triangles=tri.triangles, cmap='coolwarm')
plt.show()


# Example with animation
ANIMATION = False
if ANIMATION == True:
    x_center = np.linspace(-0.4,1.4,10)
    for frame, xc in enumerate(x_center):
        contact_center = np.array([xc, 0.5])
        displ = 0.2
        R = 1.
        gap = parabolic_indenter(coords[:,0], coords[:,1], contact_center[0], contact_center[1], R, coords[0,2]-displ) - coords[:,2]
        penetrating_nodes = np.where(gap < 0)[0]


        # Solve the problem
        max_iter = 1000
        tolerance = 1e-7
        start = time.time()
        if frame == 0:
            pressure, displacement, error_history = constrained_CG(K, coords, dofs, gap, max_iter, tolerance, 100.)
        else:
            pressure, displacement, error_history = constrained_CG(K, coords, dofs, gap, max_iter, tolerance, 100, pressure)

        print("Iters: {0:3d}, Error {1:.3e}".format(len(error_history), error_history[-1]))

## Plot using Matplotlib
        # plt.rcParams['figure.figsize'] = [10,5]
        # fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
        # ax[0].set_xlim([-0.,1.])
        # ax[0].set_ylim([-0.,1.])
        # ax[0].set_zlim([-0.2,0])
        # x = np.linspace(-0.,1.,100)
        # y = np.linspace(-0.,1.,100)
        # X,Y = np.meshgrid(x,y)
        # Z = parabolic_indenter(X,Y, contact_center[0], contact_center[1], 1., coords[0,2]-displ) - coords[0,2]
        # Z[Z>0.] = np.nan
        # surf1 = ax[0].plot_trisurf(coords[:,0], coords[:,1], -displacement, triangles=tri.triangles, cmap='coolwarm', vmin = -0.2, vmax = 0)
        # cb1 = fig.colorbar(surf1, ax=ax[0], shrink=0.6, aspect=10, orientation='horizontal')
        # cb1.set_label("Displacement")
        # ax[0].plot_surface(X,Y,Z, alpha=0.1, cmap='gray',  rcount = X.shape[0], ccount = X.shape[1], edgecolor='k', linewidth=0.1)
        # ax[0].set_title("Deformation")

        # ax[1].set_xlim([0,1])
        # ax[1].set_ylim([0,1])
        # ax[1].set_zlim([0,1e-4*np.pi*Es/1e5])
        # surf2 = ax[1].plot_trisurf(coords[:,0], coords[:,1], pressure*np.pi*Es/1e5, triangles=tri.triangles, cmap='coolwarm', vmin = 0, vmax = 4)
        # cb2 = fig.colorbar(surf2, ax=ax[1], shrink=0.6, aspect=10, orientation='horizontal')
        # cb2.set_label("Pressure")
        # ax[1].set_title("Pressure")
        # fig.tight_layout()
        # fig.savefig("Contact_{0:03d}.png".format(frame), dpi=300)

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



exit(0)

# Put nodes on the surface using Dirichlet BC implemented with the penalty method
penalty = 1e10
Kp = penalized_DBC(K, penetrating_nodes, penalty)
n = Kp.shape[0]

###############################
# Create a PETSc matrix from the NumPy array
A = PETSc.Mat().createDense((n,n),comm=PETSc.COMM_WORLD)
A.setFromOptions()
A.setUp()

for i in range(n):
    indexes = np.linspace(0, n-1, n, dtype=np.int32)
    values = Kp[i, :]
    A.setValues(i, indexes, values)
A.assemble()


# Just to check it's working
# A_np = A.getValues(range(0, A.getSize()[0]), range(0,  A.getSize()[1]))
# plt.imshow(np.log(np.abs(A_np)))
# plt.show()
# exit()  


# Create a PETSc vector from the NumPy array
x = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)  # Solution vector
b = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)  # RHS vector

indices = np.array(penetrating_nodes, dtype=PETSc.IntType)
values = np.array( gap[penetrating_nodes], dtype=PETSc.ScalarType)
b.setValues(indices,values)
b.assemble()


# Create the KSP solver context
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)  # Conjugate Gradient solver
ksp.getPC().setType(PETSc.PC.Type.JACOBI)  # Jacobi preconditioner

# Set additional options, if needed
ksp.setTolerances(rtol=1e-6)  # Relative tolerance

# Solve the system
ksp.solve(b, x)

# Retrieve the solution into a NumPy array
x_sol = x.getArray()

# Print the solution
print("Solution vector:\n", x_sol)

PLOT2 = True
if PLOT2:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.tri import Triangulation

    tri = Triangulation(coords[:,0], coords[:,1])

    deformed_surface = coords.copy()
    deformed_surface[:,2] += x_sol

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(coords[:,0], coords[:,1], gap, c=gap, cmap='coolwarm')
    # ax.plot_trisurf(coords[:,0], coords[:,1], gap, triangles=tri.triangles, cmap='gray', alpha=0.5)
    ax.plot_trisurf(deformed_surface[:,0], deformed_surface[:,1], x_sol, triangles=tri.triangles, cmap='coolwarm')

    plt.show()



exit()

# Kp_petsc = PETSc.Mat().createDense((n,n),comm=PETSc.COMM_WORLD)
# Kp_petsc.setFromOptions()
# Kp_petsc.setUp()

# Fill the PETSc matrix with data from the NumPy array
# Kp_petsc.setValues(range(Kp.shape[0]), range(Kp.shape[1]), Kp.flatten())
# for i in range(n):
#     indices = np.linspace(0, n-1, n, dtype=np.int32)
#     Kp_petsc.setValue(i, indices, Kp[i, :])
# # Assemble the matrix to finalize its structure
# Kp_petsc.assemble()

exit(0)

n = Kp.shape[0]
x = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)  # Solution vector
b = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)  # RHS vector

# b.set(1)
indices = np.array(penetrating_nodes, dtype=PETSc.IntType)
values = np.array( gap[penetrating_nodes], dtype=PETSc.ScalarType)
b.setValues(indices,values)
b.assemble()

# for i in range(len(penetrating_nodes)):
#     b.setValues(3*boundary_dofs[penetrating_nodes[i]]+2, gap[penetrating_nodes[i]])

# Create a KSP solver object
ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
ksp.setOperators(Kp_petsc)  # Set the matrix as the operator
ksp.setType(PETSc.KSP.Type.PREONLY)  # Set solver type, e.g., preonly if using a direct solver
ksp.getPC().setType(PETSc.PC.Type.LU)  # Set preconditioner type, e.g., LU for direct solve

# Set additional solver options as needed
ksp.setFromOptions()  # This allows runtime options to be set from the command line

# Solve the system
ksp.solve(b, x)

# Optionally, retrieve the solution array to use in Python
solution_array = x.getArray()
print("Solution: ", solution_array)

# rhs = problem.b.copy()
# rhs.set(0)
# for i in range(len(penetrating_nodes)):
#     rhs.setValues(3*boundary_dofs[penetrating_nodes[i]]+2, gap)
# rhs.assemble()

# Solve the problem
# solver.solve(rhs, uh)


exit(0)
    # locator = lambda x: np.isclose(x[0], 1, atol = 1e-5)
    # fdim = msh.topology.dim - 1
    # facet = locate_entities_boundary(msh, fdim, locator)
    # for i in range(len(facet)):
    #     dof_idx = locate_dofs_topological(V, fdim, [facet[i]])
    #     rhs.set(0)
    #     rhs.setValues(dof_idx, 1e5*np.ones(len(dof_idx), dtype=default_scalar_type))

# From X to uh
# uh = la.create_function(V)



# rhs.set(0)
# dof_idx = locate_dofs_topological(V, fdim, boundary_facets)
# rhs.setValues(len(dof_idx), dof_idx, 1e3*np.ones(9, dtype=default_scalar_type)) #, addv=True)
# rhs.assemble()

# solver.solve(rhs, uh.x)


# Save displacement on the deformed mesh
original_coordinates = msh.geometry.x.copy()
dx = uh.x.array
factor = 10.
msh.geometry.x[:, :] += factor*dx.reshape(-1, 3)

with XDMFFile(msh.comm, "out_elasticity/deformed_mesh.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

deviatoric_stress = sigma(uh) - (1 / 3) * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_mises_stress = ufl.sqrt((3 / 2) * inner(deviatoric_stress, deviatoric_stress))
W = functionspace(msh, ("Discontinuous Lagrange", 0))
sigma_vm_expr = Expression(von_mises_stress, W.element.interpolation_points())
sigma_vm_h = Function(W)
sigma_vm_h.interpolate(sigma_vm_expr)
with XDMFFile(msh.comm, "out_elasticity/von_mises_stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(sigma_vm_h)

# Get to the original configuration
# msh.geometry.x[:, :] = original_coordinates

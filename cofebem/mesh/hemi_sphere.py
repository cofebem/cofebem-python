import numpy as np
import meshio

# Hemisphere parameters
R = 1.0  # radius
nu = 5  # subdivisions in the radial direction (from center to boundary)
nv = 10  # subdivisions in the polar direction (v from 0 at north pole to pi/2 at equator)
nw = 20  # subdivisions in the azimuthal direction (w from 0 to 2*pi)

# Generate points using a structured grid in parameter space.
# Here u in [0,1] is the normalized radial coordinate,
# v in [0, pi/2] is the polar angle (0 = north pole, pi/2 = equator),
# and w in [0, 2*pi] is the azimuth.
points = []
for i in range(nu + 1):
    u = i / nu
    for j in range(nv + 1):
        v = (j / nv) * (np.pi / 2)
        for k in range(nw + 1):
            w = (k / nw) * (2 * np.pi)
            x = R * u * np.sin(v) * np.cos(w)
            y = R * u * np.sin(v) * np.sin(w)
            z = R * u * np.cos(v)
            points.append([x, y, z])
points = np.array(points)


# Helper: compute the global index for a grid point with indices (i, j, k)
def idx(i, j, k):
    return i * ((nv + 1) * (nw + 1)) + j * (nw + 1) + k


tetrahedra = []

# Loop over each hexahedral cell in the structured lattice.
# The cell indices run from i=0..nu-1, j=0..nv-1, k=0..nw-1.
for i in range(nu):
    for j in range(nv):
        for k in range(nw):
            # Define the 8 corners of the hexahedron:
            v0 = idx(i, j, k)
            v1 = idx(i + 1, j, k)
            v2 = idx(i + 1, j + 1, k)
            v3 = idx(i, j + 1, k)
            v4 = idx(i, j, k + 1)
            v5 = idx(i + 1, j, k + 1)
            v6 = idx(i + 1, j + 1, k + 1)
            v7 = idx(i, j + 1, k + 1)

            # Subdivide the hexahedron into 6 tetrahedra.
            # This is one possible subdivision strategy.
            tetrahedra.append([v0, v1, v3, v7])
            tetrahedra.append([v0, v1, v7, v5])
            tetrahedra.append([v0, v3, v7, v4])
            tetrahedra.append([v1, v2, v3, v7])
            tetrahedra.append([v1, v6, v7, v5])
            tetrahedra.append([v3, v7, v4, v6])

tetrahedra = np.array(tetrahedra)

# Create a meshio Mesh object using the computed points and tetrahedral connectivity.
mesh = meshio.Mesh(points=points, cells=[("tetra", tetrahedra)])

# Write the mesh to an XDMF file.
meshio.write("hemisphere.xdmf", mesh)
print("XDMF mesh 'hemisphere.xdmf' has been created.")

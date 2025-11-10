import numpy as np
import matplotlib.pyplot as plt

filename = "../out_elasticity/FlexData_21x21.npz"
data = np.load(filename)

element_centers = data['facet_centers']
boundary_coords = data['boundary_coords']
K = data['K']
p = np.zeros(K.shape[1])
print("p.shape = ", p.shape)
print("K.shape = ", K.shape)
print('elemet_shape', element_centers.shape)
print('boundary_coords', boundary_coords.shape)
fig,ax = plt.subplots()
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.scatter(element_centers[:,0], element_centers[:,1], s=5)
plt.scatter(boundary_coords[:,0], boundary_coords[:,1], s=3,marker="o")
plt.show()

# Select elements near the center
a = 0.3
x0 = 0.5
y0 = 0.5

filter = (element_centers[:,0]-x0)**2 + (element_centers[:,1]-y0)**2 < a**2
selected_elements = np.where(filter)[0]

# Put Hertzian pressure p0*np.sqrt(1-((x-x0)**2 + (y-y0)**2)/a**2) on these elements
p0 = 1e3  # Max pressure value
x = element_centers[selected_elements, 0]
y = element_centers[selected_elements, 1]
p[selected_elements] = p0 * np.sqrt(1-((x-x0)**2 + (y-y0)**2)/a**2)

print("Shapes:")
print("p.shape = ", p.shape)
print("K.shape = ", K.shape)
print("element_centers.shape = ", element_centers.shape)
print("boundary_coords.shape = ", boundary_coords.shape)

fig,ax = plt.subplots()
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
# Show pressure distribution
plt.scatter(element_centers[:,0], element_centers[:,1], c=p, s=10, marker="s",cmap='viridis')
plt.colorbar(label='Pressure magnitude')
plt.title('Applied Pressure Distribution on Elements')
plt.show()

# Solve for displacements (use transpose so shapes match: K was loaded as data['K'].T)
u = K @ p

# Get unique node positions (boundary_coords may have duplicate positions for different DOF components)
# Round to avoid floating point comparison issues
boundary_coords_rounded = np.round(boundary_coords, decimals=10)
unique_coords, unique_indices = np.unique(boundary_coords_rounded, axis=0, return_index=True)

print(f"Total boundary coordinates: {boundary_coords.shape[0]}")
print(f"Unique node positions: {unique_coords.shape[0]}")
print(f"Displacement vector size: {len(u)}")

# Extract displacements for unique nodes only
u_unique = u[unique_indices]

# Create a regular Cartesian grid from unique node coordinates
x_unique = np.unique(unique_coords[:,0])
y_unique = np.unique(unique_coords[:,1])
nx, ny = len(x_unique), len(y_unique)

print(f"Grid size: {nx} x {ny} = {nx*ny} points")
print(f"Should match unique nodes: {len(u_unique)}")

# Create meshgrid
X, Y = np.meshgrid(x_unique, y_unique)

# Map displacements to grid
Z = np.zeros_like(X)
for i in range(len(unique_coords)):
    x, y = unique_coords[i, 0], unique_coords[i, 1]
    # Find indices in grid
    ix = np.argmin(np.abs(x_unique - x))
    iy = np.argmin(np.abs(y_unique - y))
    if i < len(u_unique):
        Z[iy, ix] = u_unique[i]

# Plot deformed Cartesian grid
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 5))

# Original grid
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, np.zeros_like(X), alpha=0.3, color='gray')
ax1.plot_wireframe(X, Y, np.zeros_like(X), color='black', linewidth=0.5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Grid')

# Deformed grid
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax2.plot_wireframe(X, Y, Z, color='black', linewidth=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Displacement u')
ax2.set_title('Deformed Grid')

plt.tight_layout()
plt.show()

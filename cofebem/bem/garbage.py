import mpi4py.MPI as MPI
import numpy as np
from dolfinx.mesh import create_box, CellType, GhostMode
from dolfinx.io import XDMFFile


# W = functionspace(self.mesh, ("CG", 1, (self.mesh.geometry.dim,)))

#             # Symbolic outward normal and boundary measure
#             n_ufl = FacetNormal(self.mesh)
#             ds = Measure("ds", domain=self.mesh)

#             #
#             u_ = TrialFunction(W)
#             v_ = TestFunction(W)

#             eps = 1.0e-14
#             a_ = inner(u_, v_) * ds + eps * inner(u_, v_) * dx
#             L_ = inner(n_ufl, v_) * ds

#             normal_fn = Function(W)

#             normal_problem = LinearProblem(
#                 a_,
#                 L_,
#                 bcs=[],
#                 u=normal_fn,
#                 petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
#             )
#             normal_problem.solve()
#             normal_fn.x.scatter_forward()

#             for i, dof_applied in enumerate(
#                 tqdm(dofs, desc="Computing Contact Compliance Matrix", unit="it")
#             ):
#                 rhs.set(0)
#                 rhs.setValue(
#                     dof_applied * self.mesh.geometry.dim + force_direction,
#                     force_magnitude,
#                 )
#                 rhs.assemble()

#                 solver.solve(rhs, uh)
#                 uh_values = uh.array

#                 for j, dof_measured in enumerate(dofs):
#                     # Get the corresponding normal vector
#                     # coord = self.V.tabulate_dof_coordinates()[dof_measured]
#                     # Convert dof -> node index
#                     node_measured = dof_measured // self.mesh.geometry.dim
#                     start = node_measured * self.mesh.geometry.dim
#                     stop = (node_measured + 1) * self.mesh.geometry.dim

#                     # Extract the projected normal from normal_fn
#                     raw_normal = normal_fn.x.array[start:stop].copy()
#                     norm_val = np.linalg.norm(raw_normal)
#                     if norm_val > 1e-14:
#                         n = raw_normal / norm_val
#                     else:
#                         # e.g., at corners
#                         n = np.zeros_like(raw_normal)


# def __S_by_bruteforce(
#     self,
#     selector: Callable,
#     force_magnitude: float,
#     save: bool,
#     full: bool = True,
#     force_direction: int = None,
# ) -> np.ndarray:
#     """
#     Compute the contact compliance (flexibility) matrix S by applying a unit force
#     at each boundary node and extracting the resulting displacement. This procedure
#     follows the approach in Yastrebov and Feng (2024), where the displacement u and the
#     applied force F at the contact vertices are related via:

#         u_j = S_{ji} F_i,

#     and S is interpreted in a tensorial (block) sense. In our implementation, two modes
#     are available:

#       1. Full (vector) mode (full=True): For each boundary node the force is applied in
#          all gdim components. The resulting matrix S has shape
#          (gdim*n_vertices, gdim*n_vertices), where n_vertices is the number of unique boundary vertices.
#       2. Single–direction mode (full=False): A force is applied only in the specified force_direction,
#          and only the displacement response in that direction is recorded. Then S has shape
#          (n_vertices, n_vertices).

#     Parameters:
#       selector: A callable used to identify the boundary facets.
#       force_magnitude: The magnitude of the applied force (a nonzero scalar).
#       save: If True, the computed S and boundary node coordinates are saved.
#       full: If True, compute the full (vector) compliance matrix. If False, compute S only for one direction.
#       force_direction: The index of the force direction (0 for x, 1 for y, 2 for z). Must be specified if full is False.

#     Returns:
#       S: The computed compliance matrix.
#          - If full==True, S.shape == (gdim*n_vertices, gdim*n_vertices)
#          - If full==False, S.shape == (n_vertices, n_vertices)
#     """
#     from tqdm import tqdm
#     import numpy as np
#     import time
#     from petsc4py import PETSc

#     # Determine spatial dimension (2 for 2D, 3 for 3D)
#     gdim = self.mesh.geometry.dim

#     # In single-direction mode, force_direction must be provided.
#     if not full:
#         assert (
#             force_direction is not None
#         ), "force_direction must be specified in single–direction mode."
#         assert 0 <= force_direction < gdim, f"force_direction must be in [0, {gdim-1}]."

#     # Locate boundary facets using the provided selector.
#     fdim = (
#         self.mesh.topology.dim - 1
#     )  # boundary facets are one dimension lower than the mesh
#     facets = locate_entities_boundary(self.mesh, fdim, selector)
#     if facets.size == 0:
#         raise ValueError("No boundary facets found for the given selector.")

#     # Locate DOFs on these boundary facets.
#     # For a CG1 vector space, DOFs are interleaved and correspond to vertices.
#     dofs = locate_dofs_topological(self.V, fdim, facets)
#     if dofs.size == 0:
#         raise ValueError("No DOFs found on the boundary for the given selector.")

#     # Extract unique boundary node indices. For a vector space with interleaved ordering,
#     # each node gives gdim DOFs.
#     boundary_vertices = np.unique(dofs // gdim)
#     n_vertices = len(boundary_vertices)

#     if full:
#         # Construct a list of DOF indices corresponding to each node (all components).
#         # The ordering is: [node0_x, node0_y, (node0_z), node1_x, node1_y, (node1_z), ...]
#         full_dofs = np.array(
#             [node * gdim + comp for node in boundary_vertices for comp in range(gdim)],
#             dtype=np.int32,
#         )
#         S = np.zeros((gdim * n_vertices, gdim * n_vertices), dtype=PETSc.ScalarType)
#     else:
#         # For single–direction mode, pick the DOF corresponding to force_direction for each node.
#         selected_dofs = boundary_vertices * gdim + force_direction
#         S = np.zeros((n_vertices, n_vertices), dtype=PETSc.ScalarType)

#     # Assemble the stiffness matrix and configure the PETSc solver.
#     self.problem.A.assemble()
#     solver = PETSc.KSP().create(self.mesh.comm)
#     solver.setOperators(self.problem.A)
#     solver.setType("preonly")
#     solver.getPC().setType("lu")
#     solver.setFromOptions()
#     solver.setUp()

#     # Prepare the right-hand side and solution vectors.
#     rhs = self.problem.b.copy()
#     uh = PETSc.Vec().createMPI(rhs.getSize(), comm=self.mesh.comm)

#     # Timing start (optional)
#     start_loop = time.perf_counter()

#     if full:
#         # Loop over each DOF (in the node-wise ordering) and apply a unit force.
#         for i, global_row in enumerate(
#             tqdm(full_dofs, desc="Computing full compliance matrix", unit="it")
#         ):
#             rhs.set(0)
#             # Apply the force at the given DOF index.
#             rhs.setValue(global_row, force_magnitude)
#             rhs.assemble()

#             # Solve for the displacement response.
#             solver.solve(rhs, uh)
#             uh_values = uh.array

#             # Record the response at the same set of DOFs.
#             # (Divide by force_magnitude to get compliance per unit force.)
#             S[i, :] = uh_values[full_dofs] / force_magnitude
#     else:
#         # Single–direction mode: for each node, apply a force only in the chosen direction.
#         for i, global_row in enumerate(
#             tqdm(
#                 selected_dofs,
#                 desc="Computing single–direction compliance matrix",
#                 unit="it",
#             )
#         ):
#             rhs.set(0)
#             rhs.setValue(global_row, force_magnitude)
#             rhs.assemble()

#             solver.solve(rhs, uh)
#             uh_values = uh.array

#             # Extract the displacement in the chosen direction for each boundary node.
#             S[i, :] = (
#                 np.array(
#                     [
#                         uh_values[node * gdim + force_direction]
#                         for node in boundary_vertices
#                     ]
#                 )
#                 / force_magnitude
#             )

#     end_loop = time.perf_counter()
#     logging.info(
#         f"Compliance matrix S computed in {end_loop - start_loop:.3f} seconds."
#     )

#     # Optionally, extract the boundary node coordinates (one per node) and save the data.
#     # We reshape the coordinates to obtain one coordinate per node.
#     coords = self.V.tabulate_dof_coordinates().reshape(-1, gdim)
#     boundary_coords = coords[boundary_vertices, :]

#     if save:
#         np.savez(
#             "out_elasticity/BEM_Data.npz",
#             S=S,
#             coords=boundary_coords,
#             vertices=boundary_vertices,
#         )
#         logging.info("Compliance matrix saved to out_elasticity/BEM_Data.npz")

#     return S

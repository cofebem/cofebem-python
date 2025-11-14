# import numpy as np
# import time

# from dolfinx import *
# from dolfinx.mesh import (
#     Mesh,
#     CellType,
#     GhostMode,
#     create_box,
#     locate_entities_boundary,
#     locate_entities,
#     meshtags,
# )
# from dolfinx.fem import (
#     FunctionSpace,
#     Function,
#     Constant,
#     functionspace,
#     dirichletbc,
#     locate_dofs_topological,
#     locate_dofs_geometrical,
#     form,
# )
# from dolfinx.mesh import exterior_facet_indices
# from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
# from ufl import (
#     Measure,
#     Identity,
#     Form,
#     TrialFunction,
#     TestFunction,
#     sym,
#     grad,
#     inner,
#     tr,
#     zero,
#     FacetNormal,
#     dx,
#     ds,
# )
# from dolfinx.io import XDMFFile

# from mpi4py import MPI

if __name__ == "__main__":
    # cube = create_box(
    #     MPI.COMM_WORLD,
    #     [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    #     [2, 2, 3],
    #     CellType.hexahedron,
    #     ghost_mode=GhostMode.shared_facet,
    # )

    # V = functionspace(cube, ("CG", 1, (3,)))

    # tdim = cube.topology.dim
    # fdim = tdim - 1
    # vdim = 0
    # cube.topology.create_connectivity(fdim, vdim)
    # cube.topology.create_connectivity(fdim, tdim)

    # conn = cube.topology.connectivity(fdim, 0)

    # # # Get the exterior facets (i.e. those on the boundary)
    # boundary_facets = exterior_facet_indices(cube.topology)
    # print(f"boundary facets = {boundary_facets}\ntotal facets = {len(boundary_facets)}")
    # boundary_vertices = locate_dofs_topological(
    #     V, cube.topology.dim - 1, boundary_facets
    # )
    # print(
    #     f"boundary vertices = {boundary_vertices} \ntotal vertices = {len(boundary_vertices)}"
    # )

    # mapping = {dof: i for i, dof in enumerate(boundary_vertices)}
    # perm = np.array([mapping[dof] for dof in boundary_vertices])
    # print(
    #     f"boundary vertices reodered = {perm} \ntotal vertices = {len(boundary_vertices)}"
    # )

    # for f in boundary_facets:
    #     # print(conn.links(f))
    #     # print(f"facet : {f}")
    #     vertices = conn.links(f)
    #     print(f" facet = {f}")
    #     print(f"vertices = {vertices}")
    #     v1, v2, v3, v4 = (
    #         cube.geometry.x[vertices[0]],
    #         cube.geometry.x[vertices[1]],
    #         cube.geometry.x[vertices[2]],
    #         cube.geometry.x[vertices[3]],
    #     )

    # with XDMFFile(cube.comm, "cube.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(cube)
    #     coords = V.tabulate_dof_coordinates()
    #     v1_, v2_, v3_, v4_ = (
    #         coords[vertices[0]],
    #         coords[vertices[1]],
    #         coords[vertices[2]],
    #         coords[vertices[3]],
    #     )
    # print(v1)
    # print(v1_)
    # print(v2)
    # print(v2_)
    # print(v3)
    # print(v3_)
    # print(v4)
    # print(v4_)
    # centroid = np.mean(np.vstack((v1, v2, v3, v4)), axis=0)
    # print(f"centroid = {centroid}")
    # break

    # Extract the boundary DOFs; note that facets in a 3D cube have dimension 2.
    # boundary_vertices = locate_dofs_topological(
    #     V, cube.topology.dim - 1, np.asarray([boundary_facets[0]])
    # )

    # boundary_dofs_coords = V.tabulate_dof_coordinates()[boundary_dofs]
    # print("Boundary DOFs:", boundary_dofs)
    # print("Boundary DOFs size:", boundary_dofs.size)
    # print("total number of dofs: ", V.dofmap.index_map.size_global)
    # print(boundary_dofs_coords)

    # -------------------------------------------------------------------------------------------------------
    #  compute normals
    # -------------------------------------------------------------------------------------------------------

    # from dolfinx.io import XDMFFile
    # from dolfinx.io import VTXWriter
    # from petsc4py import PETSc

    # # Create the mesh (Unit Disc)
    # with XDMFFile(MPI.COMM_WORLD, "hex_hollow_cylinder.xdmf", "r") as xdmf:
    #     msh = xdmf.read_mesh(name="Grid")

    # msh = create_box(
    #     MPI.COMM_WORLD,
    #     [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
    #     [2, 2, 1],
    #     CellType.hexahedron,
    #     ghost_mode=GhostMode.shared_facet,
    # )

    # V_ = functionspace(msh, ("CG", 1, (msh.topology.dim,)))

    # n = FacetNormal(msh)

    # u_, v_ = TrialFunction(V_), TestFunction(V_)

    # normal_fn = Function(V_)
    # normal_fn.name = "normal"

    # eps = 1.0e-8
    # a_ = eps * inner(u_, v_) * dx + inner(u_, v_) * ds

    # L_ = inner(n, v_) * ds

    # problem = LinearProblem(
    #     a=a_,
    #     L=L_,
    #     bcs=[],
    #     u=normal_fn,
    #     petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    # )

    # problem.solve()

    # normal_fn.x.scatter_forward()

    # with XDMFFile(msh.comm, "normals_tyre.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(msh)
    #     xdmf.write_function(normal_fn)

    # import numpy as np

    # def z(t):
    #     return (1 - t) * 1j * np.pi

    # def dz_dt(t):
    #     return -1j * np.pi

    # def f(z):
    #     return np.exp(-z)

    # a, b = 0, 1
    # N = 1000  # number of subdivisions
    # ts = np.linspace(a, b, N)

    # integrand = f(z(ts)) * dz_dt(ts)

    # integral = np.trapz(integrand, ts)

    # print("Numerical contour integral:", np.real(integral))

    # from mpi4py import MPI
    # import dolfinx

    # mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    # mesh.name = "InitialMesh"
    # element_type = "Lagrange"
    # element_degree = 1

    # V = dolfinx.fem.functionspace(
    #     mesh, (element_type, element_degree, (mesh.geometry.dim,))
    # )
    # u = dolfinx.fem.Function(V)
    # # u.interpolate(lambda x: x[0] * x[1])
    # u.name = "f"

    # xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "functions.xdmf", "w")
    # xdmf.write_mesh(mesh)
    # xdmf.write_function(u, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")

    # mesh.topology.create_connectivity(1, 2)
    # r_mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 15, 15)
    # r_mesh.name = "Refined"
    # Vr = dolfinx.fem.functionspace(
    #     r_mesh, (element_type, element_degree, (mesh.geometry.dim,))
    # )
    # ur = dolfinx.fem.Function(Vr)
    # # ur.interpolate(lambda x: x[0] * x[1])
    # ur.name = "f"

    # xdmf.write_mesh(r_mesh)
    # xdmf.write_function(ur, t=1, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{r_mesh.name}']")
    # xdmf.close()

    # import numpy as np
    # import matplotlib.pyplot as plt

    # u = np.linspace(0, 1, 16)
    # v = np.linspace(0, 1, 16)
    # U, V = np.meshgrid(u, v)

    # Xi1 = U * (1 - V)
    # Xi2 = U * V
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plt.subplots_adjust(wspace=0.25)

    # # ===== Left: Unit square (u,v) =====
    # ax = axes[0]
    # for i in range(len(v)):
    #     ax.plot(u, v[i] * np.ones_like(u), color="gray", lw=0.5)
    # for j in range(len(u)):
    #     ax.plot(u[j] * np.ones_like(v), v, color="gray", lw=0.5)

    # ax.scatter(U, V, color="black", s=8)
    # ax.set_title("Unit Square $(u,v)$")
    # ax.set_xlabel("$u$")
    # ax.set_ylabel("$v$")
    # ax.set_xlim(-0.02, 1.02)
    # ax.set_ylim(-0.02, 1.02)
    # ax.set_aspect("equal")

    # ax = axes[1]
    # for i in range(len(v)):
    #     ax.plot(Xi1[i, :], Xi2[i, :], color="RoyalBlue", lw=0.7, alpha=0.7)
    # for j in range(len(u)):
    #     ax.plot(Xi1[:, j], Xi2[:, j], color="RoyalBlue", lw=0.7, alpha=0.7)

    # # Triangle boundary
    # triangle = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    # ax.plot(triangle[:, 0], triangle[:, 1], "k-", lw=1.2)
    # ax.scatter(Xi1, Xi2, color="RoyalBlue", s=8)

    # ax.set_title(r"Mapped Triangle $(\xi_1,\xi_2)=(u(1-v), u v)$")
    # ax.set_xlabel(r"$\xi_1$")
    # ax.set_ylabel(r"$\xi_2$")
    # ax.set_xlim(-0.02, 1.02)
    # ax.set_ylim(-0.02, 1.02)
    # ax.set_aspect("equal")

    # plt.tight_layout()
    # # plt.savefig("figures/duffy_mapping_points.png", dpi=300)
    # plt.show()
    from cofebem.mesh.hollow_cylinder import hollow_cylinder

    # -------------------------------------------------------------------------------------------------------
    #  Mesh and material parameters
    # -------------------------------------------------------------------------------------------------------
    nr = 30
    nt = 200
    nz = 1

    r_inner = 1
    r_outer = 5

    hollow_cylinder(nr, nt, nz, r_inner, r_outer)

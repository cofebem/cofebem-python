import numpy as np
import time

from dolfinx.mesh import (
    Mesh,
    CellType,
    GhostMode,
    create_box,
    locate_entities_boundary,
    locate_entities,
    meshtags,
    exterior_facet_indices,
)
from dolfinx.fem import (
    FunctionSpace,
    Function,
    Constant,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
    locate_dofs_geometrical,
    form,
)
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from ufl import (
    Measure,
    Identity,
    Form,
    TrialFunction,
    TestFunction,
    sym,
    grad,
    inner,
    tr,
    zero,
    FacetNormal,
    dx,
    ds,
)
from dolfinx.io import XDMFFile

from mpi4py import MPI
from petsc4py import PETSc
from typing import Callable, Optional, Union
from tqdm import tqdm
import logging
from scipy.sparse.linalg import splu, spsolve
from scipy.linalg import solve

from cofebem.utils.linalg.schur_complement import schur_complement


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FenicsLE:
    def __init__(
        self,
        mesh: Mesh,
        element_type: str = "Lagrange",
        element_degree: int = 1,
        Vforce: Optional[Union[Callable, np.ndarray, float]] = None,
        E: float = 1e9,
        nu: float = 0.3,
    ):
        """
        A class for solving linear elasticity problems using FEniCSx.

        Parameters
        ----------
        mesh : Mesh
            The computational mesh used for the finite element discretization.
        element_type : str, optional
            The type of finite element to use, e.g., "CG" (default is "CG").
        element_degree : int, optional
            The degree of the finite element basis functions (default is 1).
        Vforce : Callable, np.ndarray, or float, optional
            The volumetric force applied to the domain (default is None).
        E : float, optional
            Young's modulus of the material (default = Steel is 2.1e11).
        nu : float, optional
            Poisson's ratio of the material (default is 0.3).
        """

        self.mesh = mesh

        # Define function space
        self.V = functionspace(
            mesh, (element_type, element_degree, (self.mesh.geometry.dim,))
        )
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        assert E > 0, "Young's modulus must be positive."
        assert 0 <= nu < 0.5, "Poisson's ratio must be in [0, 0.5)."
        self.E = E
        self.nu = nu

        # Volumique force
        self.Vforce = self.__initialize_Vforce(Vforce)

        # Lame's parameters
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))

        # Boundary conditions list
        self.dirichlet_bcs = []
        self.dirichlet_dofs = []
        self.fdim = self.mesh.topology.dim - 1
        self.facets = []
        self.facets_markers = []
        self.meshtags = []
        self.neumann_bcs = []

        # Placeholder for problem and solution
        self.problem = None
        # self.free_problem = None
        self.uh = None

    def __initialize_Vforce(
        self, Vforce: Optional[Union[Callable, np.ndarray, float]]
    ) -> Optional[Union[Constant, Function]]:
        """
        Initialize the volumetric force.

        Parameters
        ----------
        Vforce : Callable, np.ndarray, or float
            The volumetric force to be applied.

        Returns
        -------
        Constant or Function
            The volumetric force as a FEniCS object.
        """
        dim = self.mesh.geometry.dim  # Dimensionality of the problem

        if Vforce is None:
            # Return a zero vector if no force is provided
            return Constant(self.mesh, PETSc.ScalarType([0.0] * dim))

        if isinstance(Vforce, float):
            # Convert scalar to vector with identical components in each direction
            return Constant(self.mesh, PETSc.ScalarType([Vforce] * dim))

        if isinstance(Vforce, np.ndarray):
            # Ensure the array is vector-valued and has the correct dimensionality
            if Vforce.shape != (dim,):
                raise ValueError(
                    f"Vforce must have shape ({dim},), but got shape {Vforce.shape}."
                )
            return Constant(self.mesh, PETSc.ScalarType(Vforce))

        if callable(Vforce):
            # Interpolate a callable into a Function
            force_function = Function(self.V)
            force_function.interpolate(Vforce)
            return force_function

        raise TypeError("Force must be a float, np.ndarray, callable, or None.")

    def __check_value(self, value):
        """
        Helper method to create a Constant or Function from a given value.

        Parameters
        ----------
        value : Union[Callable, float, np.ndarray, Function, Constant]
            The input value to be converted.

        Returns
        -------
        Constant or Function
            The corresponding FEniCS object.
        """
        if callable(value):
            func = Function(self.V)
            func.interpolate(value)
            return func
        elif isinstance(value, (float, np.ndarray)):
            return Constant(self.mesh, PETSc.ScalarType(value))
        elif isinstance(value, (Function, Constant)):
            return value
        else:
            raise TypeError("Invalid value type for value.")

    def epsilon(self, v) -> Form:
        """
        Compute the symmetric strain tensor.

        Parameters
        ----------
        v : Function or TrialFunction/TestFunction
            The displacement field.

        Returns
        -------
        ufl.form.Form
            Symmetric gradient of the displacement field.
        """
        return sym(grad(v))

    def sigma(self, u) -> Form:
        """
        Compute the stress tensor using Hooke's law.

        Parameters
        ----------
        u : Function or TrialFunction/TestFunction
            The displacement field.

        Returns
        -------
        ufl.form.Form
            The stress tensor.
        """
        return 2.0 * self.mu * self.epsilon(u) + self.lmbda * tr(
            self.epsilon(u)
        ) * Identity(len(u))

    def a(self) -> Form:
        """
        Define the bilinear form for the elasticity problem.

        Returns
        -------
        ufl.form.Form
            The bilinear form.
        """
        return inner(self.sigma(self.u), self.epsilon(self.v)) * dx

    def L(self) -> Form:
        """
        Define the linear form for the elasticity problem.

        Returns
        -------
        ufl.form.Form
            The linear form.
        """
        L_form = 0
        if self.Vforce:
            L_form += inner(self.Vforce, self.v) * dx

        for value, marker_id in self.neumann_bcs:
            L_form += 1e8 * inner(value, self.v) * self.ds(marker_id)
        return L_form

    def add_dirichlet_bc(self, value, locator: Callable) -> None:
        """
        Add a Dirichlet boundary condition.

        Parameters
        ----------
        value : float, np.ndarray, Function, or Constant
            The boundary displacement values.
        locator : Callable
            A function to identify boundary nodes.
        """
        # Locate boundary facets using the locator
        fdim = self.mesh.topology.dim - 1  # Boundary facets are of dimension (dim - 1)
        facets = locate_entities_boundary(self.mesh, fdim, locator)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        # Locate DOFs on these boundary facets
        dofs = locate_dofs_topological(self.V, fdim, facets)

        # Ensure DOFs are extracted
        if not dofs.size:
            raise ValueError("No DOFs found on the boundary for the given locator.")

        # Handle the value parameter
        value = self.__check_value(value)

        bc = dirichletbc(value, dofs, self.V)
        # Add the boundary condition to the list
        self.dirichlet_bcs.append(bc)
        self.dirichlet_dofs.append(dofs)

    def add_neumann_bc(self, value, locator: Callable, marker_id: int) -> None:
        """
        Add a Neumann boundary condition.

        Parameters
        ----------
        value : Callable, float, np.ndarray, Function, or Constant
            The traction vector applied on the boundary.
        locator : Callable
            A function to identify boundary facets.
        marker_id : int
            The marker ID for the boundary facets.
        """
        facets = locate_entities_boundary(self.mesh, self.fdim, locator)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        self.facets.append(facets)
        self.facets_markers.append(np.full(facets.size, marker_id))

        # tags = meshtags(self.mesh, fdim, facets, np.full(len(facets), marker_id))
        # measure = Measure("ds", domain=self.mesh, subdomain_data=tags)(marker_id)

        # Handle the value parameter
        value = self.__check_value(value)

        self.neumann_bcs.append((value, marker_id))

    def setup(self, petsc_options: Optional[dict] = None) -> None:
        """
        Set up the linear problem.

        Parameters
        ----------
        petsc_options : dict, optional
            PETSc solver options (default is None).
        """
        if petsc_options is None:
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

        if self.neumann_bcs:
            self.facets = np.hstack(self.facets).astype(np.int32)
            self.facets_markers = np.hstack(self.facets_markers).astype(np.int32)
            sorted_facets = np.argsort(self.facets)

            self.meshtags = meshtags(
                self.mesh,
                self.fdim,
                self.facets[sorted_facets],
                self.facets_markers[sorted_facets],
            )
            self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.meshtags)

        # Create the linear problem
        self.problem = LinearProblem(
            a=self.a(), L=self.L(), bcs=self.dirichlet_bcs, petsc_options=petsc_options
        )

    def solve(self) -> None:
        """
        Solve the elasticity problem.

        Returns
        -------
        Function
            The solution displacement field.
        """
        if self.problem is None:
            self.setup_problem()

        self.uh = self.problem.solve()
        logging.info("Solution computed successfully.")

    def set_force(self, force: Union[Callable, np.ndarray, float]) -> None:
        """
        Update the force after initialization.

        :param force: A constant value, callable function, or None.
        :type force: Union[Callable, np.ndarray, float]
        """
        self.Vforce = self.__initialize_Vforce(force)
        self.setup_problem()

    def get_solution(self):
        """
        Get the computed solution.

        :return: The computed displacement field.
        :rtype: Function
        """
        if self.uh is None:
            raise RuntimeError(
                "Problem not solved yet. Call solve() before get_solution()."
            )
        return self.uh

    def visualize(self, filename: str = "deformed_solution.xdmf"):
        """
        Visualize the deformed solution by writing it to an XDMF file.

        Parameters
        ----------
        filename : str, optional
            The name of the XDMF file to save the results (default is 'deformed_solution.xdmf').
        scale : float, optional
            Scaling factor for the deformation (default is 1.0).
        """
        if self.uh is None:
            raise RuntimeError(
                "No solution found. Solve the problem before visualizing."
            )

        self.uh.name = "Displacement"
        with XDMFFile(self.mesh.comm, f"{filename}", "w") as xdmf:
            xdmf.write_mesh(self.mesh)
            xdmf.write_function(
                self.uh
            )  # Use WarpByVector in Paraview to visualize the deformation
        logging.info("Solution saved for visualization in Paraview.")

    def compute_S(
        self,
        selector: Callable,
        method: str = "bruteforce",
        force_direction: int = 2,
        force_magnitude: float = 1.0,
        save: bool = False,
    ) -> np.ndarray:
        """
        Compute the contact compliance matrix (S) for the current mesh and problem setup using the specified method.

        Parameters
        ----------
        selector : Callable
            A function to select boundary nodes for the BEM computation.
        method : str, optional
            Method to compute the BEM matrix, either "bruteforce" or "schur" (default is "bruteforce").
        force_direction : int, optional
            The direction in which to apply the force for the "bruteforce" method (0 for x, 1 for y, 2 for z; default is 2).
        force_magnitude : float, optional
            The magnitude of the force to be applied for the "bruteforce" method (default is 1.0).
        save : bool, optional
            Whether to save the BEM matrix and associated data to a file (default is True).

        Returns
        -------
        np.ndarray
            The computed BEM matrix.
        """
        if method == "bruteforce":
            return self.__S_by_bruteforce(
                selector, force_direction, force_magnitude, save
            )
        elif method == "schur":
            return self.__S_by_schur(selector, force_direction, save)
        elif method == "bem":
            return self.__S_by_bem(selector, force_direction, save)
        else:
            raise ValueError(
                "Invalid method. Choose either 'bruteforce' or 'schur' or 'bem'."
            )

    def __S_by_bruteforce(
        self,
        selector: Callable,
        force_direction: int,
        force_magnitude: float,
        save: bool,
        include_tangential: bool = False,
    ) -> np.ndarray:
        assert (
            0 <= force_direction <= 2
        ), "Force direction must be 0 (x), 1 (y), or 2 (z)."
        assert isinstance(
            force_magnitude, (float, int)
        ), "Force magnitude must be a float or integer."

        # Locate boundary facets using the locator
        fdim = self.mesh.topology.dim - 1  # Boundary facets are of dimension (dim - 1)
        facets = locate_entities_boundary(self.mesh, fdim, selector)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        # Locate DOFs on these boundary facets
        dofs = locate_dofs_topological(self.V, fdim, facets)

        # Ensure DOFs are extracted
        if not dofs.size:
            raise ValueError("No DOFs found on the boundary for the given locator.")

        # Initialize PETSc solver
        self.problem.A.assemble()
        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(self.problem.A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        solver.setUp()

        # Initialize right-hand side and solution vectors
        rhs = self.problem.b.copy()
        uh = PETSc.Vec().createMPI(rhs.getSize(), comm=self.mesh.comm)
        if include_tangential:
            # Store normal and tangential components
            S = np.zeros(
                (dofs.size, dofs.size, 3)
            )  # [DOF applied, DOF measured, Components: Normal, Tangential1, Tangential2]

            for i, dof_applied in enumerate(
                tqdm(dofs, desc="Computing Contact Compliance Matrix", unit="it")
            ):
                rhs.set(0)
                rhs.setValue(
                    dof_applied * self.mesh.geometry.dim + force_direction,
                    force_magnitude,
                )
                rhs.assemble()

                solver.solve(rhs, uh)
                uh_values = uh.array

                for j, dof_measured in enumerate(dofs):
                    # Get the corresponding normal vector
                    # coord = self.V.tabulate_dof_coordinates()[dof_measured]
                    # Convert dof -> node index
                    node_measured = dof_measured // self.mesh.geometry.dim
                    start = node_measured * self.mesh.geometry.dim
                    stop = (node_measured + 1) * self.mesh.geometry.dim

                    # Extract displacement vector
                    disp_vector = uh_values[
                        dof_measured
                        * self.mesh.geometry.dim : (dof_measured + 1)
                        * self.mesh.geometry.dim
                    ]

                    S[i, j, :] = disp_vector

        else:
            # Compute S matrix
            S = np.zeros((dofs.size, dofs.size), dtype=PETSc.ScalarType)

            # time for one step
            start_loop = time.perf_counter()
            for i, dof in enumerate(
                tqdm(dofs, desc="Computing Contact Compliance Matrix", unit="it")
            ):
                # Reset the right-hand side
                rhs.set(0)
                rhs.setValue(
                    dof * self.mesh.geometry.dim + force_direction, force_magnitude
                )  # Apply force along specified direction
                rhs.assemble()

                # Solve the linear system
                solver.solve(rhs, uh)

                # Extract the response at boundary DOFs
                uh_values = uh.array
                S[i, :] = [
                    uh_values[dof_idx * self.mesh.geometry.dim + force_direction]
                    / force_magnitude
                    for dof_idx in dofs
                ]
                if i == 0:
                    end_step1 = time.perf_counter()
                    duration_step1 = end_step1 - start_loop

            end_loop = time.perf_counter()
            duration_loop = end_loop - start_loop
            duration_iteration = (duration_loop - duration_step1) / (dofs.size - 1)
            # print(f"Loop duration = {duration_loop}")
            # print(f"Duration of step 1 (iteration 0) = {duration_step1}")
            # print(f"Duration of an iteration (except. step1) = {duration_iteration}")
            # print(
            #     f"Relative difference = {((duration_step1-duration_iteration)/duration_iteration)*100} %"
            # )

        logging.info(f"H computed successfully by brut force")

        # Extract boundary node coordinates
        boundary_coords = self.V.tabulate_dof_coordinates()[dofs]
        if save:
            # Save the BEM matrix and associated data
            np.savez(
                "out_elasticity/BEM_Data.npz",
                S=S,
                coords=boundary_coords,
                dofs=dofs,
            )
            logging.info(f"Compliance matrix saved to out_elasticity/BEM_Data.npz")

        return S, boundary_coords, dofs

    def __S_by_schur(
        self, selector: Callable, force_direction: int, save: bool
    ) -> np.ndarray:
        # Only the case with Vforce = f_v = 0 is  implemented
        # ToDO: Implement the general case
        # Locate boundary facets using the locator
        fdim = self.mesh.topology.dim - 1  # Boundary facets are of dimension (dim - 1)
        facets = locate_entities_boundary(self.mesh, fdim, selector)

        # Ensure boundary facets are found
        if not facets.size:
            raise ValueError("No boundary facets found for the given locator.")

        # Locate DOFs on these boundary facets
        boundary_dofs = locate_dofs_topological(self.V, fdim, facets)
        boundary_dofs = self.mesh.geometry.dim * boundary_dofs + force_direction

        # Ensure DOFs are extracted
        if not boundary_dofs.size:
            raise ValueError("No DOFs found on the boundary for the given locator.")

        # # Assemble the global matrix
        # self.problem.A.assemble()
        # K = self.problem.A
        # K.assemble()
        # K.setUp()
        # K = K.convert("aij")

        # # Partition the global matrix into blocks
        # all_dofs = np.arange(boundary_dofs.size, dtype=np.int32)
        # uv_dofs = np.setdiff1d(all_dofs, boundary_dofs)
        # # uv_dofs = np.setdiff1d(uv_dofs, np.array(self.dirichlet_dofs))
        # uc_dofs = boundary_dofs

        # IS_uv = PETSc.IS().createGeneral(uv_dofs, comm=PETSc.COMM_WORLD)
        # IS_uc = PETSc.IS().createGeneral(uc_dofs, comm=PETSc.COMM_WORLD)

        # # Extract the four blocks as submatrices.
        # Kvv = K.getLocalSubMatrix(IS_uv, IS_uv)
        # Kvc = K.getLocalSubMatrix(IS_uv, IS_uc)
        # Kcv = K.getLocalSubMatrix(IS_uc, IS_uv)
        # Kcc = K.getLocalSubMatrix(IS_uc, IS_uc)

        # # Now, create a Schur complement object from these blocks.
        # schur = PETSc.Mat().createSchurComplement(Kvv, Kvc, Kcv, Kcc)
        # schur.assemble()
        # schur.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        # schur.setUp()

        # # Convert the Schur complement to AIJ format.
        # schur.convert("aij")

        # ksp = PETSc.KSP().create(self.mesh.comm)
        # ksp.setOperators(schur)
        # ksp.setType("preonly")
        # ksp.getPC().setType("lu")
        # ksp.setFromOptions()
        # ksp.setUp()

        # # Create a PETSc dense matrix representing the identity matrix B of size n x n.
        # B = PETSc.Mat().createDense(
        #     [boundary_dofs.size, boundary_dofs.size], comm=self.mesh.comm
        # )
        # B.setUp()
        # B.zeroEntries()  # Set all entries to 0.
        # # Fill B with the identity.
        # for i in range(boundary_dofs.size):
        #     B.setValue(i, i, 1.0)
        # B.assemble()

        # # Create a PETSc dense matrix X to hold the solution (i.e. the inverse of A).
        # S = PETSc.Mat().createDense(
        #     [boundary_dofs.size, boundary_dofs.size], comm=self.mesh.comm
        # )
        # S.setUp()

        # # Now, solve the matrix equation A*X = B.
        # # This call will compute X = A^{-1} * B, i.e. X = A^{-1}.
        # ksp.matMatSolve(B, S)
        print("Starting the conversion from sparse to dense")
        K = self.problem.A.convert("dense").getDenseArray()
        print("ending the conversion from sparse to dense")

        # Partition the global matrix into blocks
        all_dofs = np.arange(K.shape[0])
        uv_dofs = np.setdiff1d(all_dofs, boundary_dofs)
        # uv_dofs = np.setdiff1d(uv_dofs, np.array(self.dirichlet_dofs))
        uc_dofs = boundary_dofs

        Kvv = K[np.ix_(uv_dofs, uv_dofs)]
        Kvc = K[np.ix_(uv_dofs, uc_dofs)]
        Kcv = K[np.ix_(uc_dofs, uv_dofs)]
        Kcc = K[np.ix_(uc_dofs, uc_dofs)]

        S = np.linalg.inv(schur_complement(Kvv, Kvc, Kcv, Kcc))

        # import scipy.sparse.linalg as spla

        # solver = spla.gmres
        # T = schur_complement(Kvv, Kvc, Kcv, Kcc)
        # n = T.shape[0]
        # I = np.eye(n)  # Identity matrix
        # S = np.zeros_like(T, dtype=np.float64)  # Storage for inverse

        # # Solve Ax = I column-by-column
        # for i in range(n):
        #     e_i = I[:, i]  # i-th unit vector
        #     S[:, i], _ = solver(T, e_i, rtol=1e-15, maxiter=1000)

        logging.info(f"H computed successfully by schur complement")

        # Extract boundary node coordinates
        boundary_coords = self.V.tabulate_dof_coordinates()[
            (boundary_dofs - force_direction) // self.mesh.geometry.dim
        ]
        if save:
            # Extract boundary node coordinates
            boundary_coords = self.V.tabulate_dof_coordinates()[boundary_dofs]
            # Save the BEM matrix and associated data
            np.savez(
                "out_elasticity/BEM_Data_Schur.npz",
                S=S,
                coords=boundary_coords,
                dofs=boundary_dofs,
            )
            logging.info(f"BEM matrix saved to out_elasticity/BEM_Data.npz")

        return S, boundary_coords, boundary_dofs

    ############################# S_c by bem #################################################

    ######################
    # def __S_by_bem(
    #     self, selector: Callable, force_direction: int, save: bool
    # ) -> np.ndarray:
    #     """
    #     Parameters
    #     ----------
    #     selector : Callable
    #         A function to select boundary facets.
    #     force_direction : int
    #         The index of the force/displacement component (0 for x, 1 for y, 2 for z).
    #     save : bool
    #         Whether to save the computed S matrix (together with coordinates and DOFs) to file.

    #     Returns
    #     -------
    #     tuple
    #         (S, boundary_coords, boundary_dofs) where S is the computed compliance matrix,
    #         boundary_coords are the coordinates of the boundary nodes (each row is a d‐vector),
    #         and boundary_dofs are the corresponding DOF indices.
    #     """
    #     import math

    #     # Check that force_direction is valid.
    #     if force_direction not in [0, 1, 2]:
    #         raise ValueError("Force direction must be 0 (x), 1 (y), or 2 (z).")

    #     self.mesh.topology.create_connectivity(
    #         self.mesh.topology.dim - 1, self.mesh.topology.dim
    #     )
    #     Gamma = exterior_facet_indices(self.mesh.topology)
    #     Gamma_c = locate_entities_boundary(
    #         self.mesh, self.mesh.topology.dim - 1, selector
    #     )

    #     N = self.V.dofmap.index_map.size_global

    #     # # Identify boundary facets using the provided selector.
    #     # fdim = self.mesh.topology.dim - 1
    #     # facets = locate_entities_boundary(self.mesh, fdim, selector)
    #     # if facets.size == 0:
    #     #     raise ValueError("No boundary facets found for the given selector.")

    #     # # Locate the boundary DOFs (nodes)
    #     # boundary_nodes = locate_dofs_topological(self.V, fdim, facets)
    #     # boundary_dofs = boundary_nodes
    #     # if boundary_nodes.size == 0:
    #     #     raise ValueError("No boundary DOFs found for the given selector.")

    #     # Get the coordinates of the selected DOFs.
    #     # The coordinate array has shape (num_dofs, d)
    #     boundary_coords = self.V.tabulate_dof_coordinates()[boundary_dofs]
    #     N = boundary_coords.shape[0]  # number of boundary nodes

    #     # --- Compute an approximate patch area for each boundary node ---
    #     # Compute total boundary area from the selected facets and then assign the average area to each node.
    #     # Ensure that connectivity from facets to vertices exists.
    #     conn = self.mesh.topology.connectivity(fdim, 0)
    #     if conn is None:
    #         self.mesh.topology.create_connectivity(fdim, 0)
    #         conn = self.mesh.topology.connectivity(fdim, 0)
    #     facet2vertex = self.mesh.topology.connectivity(fdim, 0).array
    #     vertices = self.mesh.geometry.x  # coordinates of all vertices

    #     total_area = 0.0
    #     for facet in facets:
    #         facet_vertices = facet2vertex[facet : facet + 4]
    #         if facet_vertices.size == 4:
    #             v0, v1, v2, v3 = vertices[facet_vertices]
    #             # Split quadrilateral into two triangles.
    #             area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    #             area2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
    #             facet_area = area1 + area2
    #         elif facet_vertices.size == 3:
    #             v0, v1, v2 = vertices[facet_vertices]
    #             facet_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    #         else:
    #             # raise NotImplementedError(
    #             #     "Facet with {} vertices not implemented".format(facet_vertices.size)
    #             # )
    #             v0, v1, v2, v3 = vertices[facet_vertices : facet_vertices + 4]
    #             # Split quadrilateral into two triangles.
    #             area1 = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    #             area2 = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))
    #             facet_area = area1 + area2
    #         total_area += facet_area
    #     A_node = total_area / float(N)

    #     # --- Assemble the G and H matrices ---
    #     G = np.zeros((N, N), dtype=np.float64)
    #     H = np.zeros((N, N), dtype=np.float64)

    #     # Material constants from the Kelvin solution (for an infinite isotropic elastic medium)
    #     mu = self.mu
    #     nu = self.nu
    #     coeff_U = 1.0 / (16.0 * math.pi * mu * (1.0 - nu))
    #     coeff_T = -1.0 / (8.0 * math.pi * (1.0 - nu))
    #     tol = 1e-12  # tolerance for near-singularity

    #     # Effective radius for PART integration (assuming a circular patch)
    #     r_eff = math.sqrt(A_node / math.pi)

    #     # Loop over all collocation (observation) nodes i and source nodes j.
    #     for i in range(N):
    #         for j in range(N):
    #             if i == j:
    #                 # Self-term: use PART integration (analytic treatment over a circular patch)
    #                 U_val = coeff_U * ((3.0 - 4.0 * nu + 0.5) / r_eff)
    #                 T_val = coeff_T * (((1.0 - 2.0 * nu) - 1.5) / (r_eff**3))
    #             else:
    #                 r_vec = boundary_coords[i] - boundary_coords[j]
    #                 r_norm = np.linalg.norm(r_vec)
    #                 if r_norm < tol:
    #                     r_norm = r_eff  # safeguard
    #                 r_comp = r_vec[force_direction]
    #                 U_val = coeff_U * (
    #                     (3.0 - 4.0 * nu) / r_norm + (r_comp**2) / (r_norm**3)
    #                 )
    #                 T_val = coeff_T * (
    #                     (1.0 - 2.0 * nu) / (r_norm**3) - 3.0 * (r_comp**2) / (r_norm**5)
    #                 )
    #             G[i, j] = A_node * U_val
    #             H[i, j] = A_node * T_val

    #     M = H + 0.5 * np.eye(N)
    #     S = np.linalg.solve(M, G)

    #     logging.info(
    #         "Compliance matrix S computed successfully by BEM with PART integration for singular terms."
    #     )

    #     if save:
    #         np.savez(
    #             "out_elasticity/BEM_Data.npz",
    #             S=S,
    #             coords=boundary_coords,
    #             dofs=boundary_dofs,
    #         )
    #         logging.info("BEM data saved to out_elasticity/BEM_Data.npz")

    #     return S, boundary_coords, boundary_dofs


if __name__ == "__main__":
    from dolfinx.mesh import create_unit_cube
    from mpi4py import MPI
    import numpy as np

    # Create mesh
    # mesh = create_unit_cube(MPI.COMM_WORLD, 15, 15, 5)
    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [100, 50, 10],
        CellType.hexahedron,
        ghost_mode=GhostMode.shared_facet,
    )

    # Volumic force
    rho = 7850
    g = 9.81
    force = np.array([0, 0, -rho * g])

    # Define boundary condition selector
    def boundary_selector1(x):
        return np.isclose(x[2], 0, atol=1e-5)

    # Define boundary condition selector for H
    selector_tol = 0.06

    def boundary_selector2(x):
        return (
            np.isclose(x[0], 0.5, atol=selector_tol)
            & np.isclose(x[1], 0.5, atol=selector_tol)
            & np.isclose(x[2], 1.0, atol=selector_tol)
        )

    # def boundary_selector4(x):
    #     return (
    #         np.isclose(x[0], 0.25, atol=selector_tol)
    #         & np.isclose(x[1], 0.0, atol=selector_tol)
    #         & np.isclose(x[2], 0.1, atol=selector_tol)
    # )

    # Define boundary condition selector for H
    def boundary_selector3(x):
        return np.isclose(x[2], 1, atol=1e-5)

    # Initialize FenicsLE
    fenics_le = FenicsLE(mesh=mesh, E=1.0e9, nu=0.3)

    # Add Dirichlet boundary condition
    fenics_le.add_dirichlet_bc(
        value=np.array([0.0, 0.0, 0.0]), locator=boundary_selector1
    )

    fenics_le.add_neumann_bc(
        value=np.array([0.0, 0.0, -1e2]),
        locator=boundary_selector2,
        marker_id=1,
    )

    # fenics_le.add_neumann_bc(
    #     value=np.array([0.0, 1e2, 0.0]),
    #     locator=boundary_selector4,
    #     marker_id=2,
    # )

    # Set up and solve the problem
    # fenics_le.setup()
    # fenics_le.solve()

    # # Solution
    # uh = fenics_le.get_solution()

    # # Visualize
    # fenics_le.visualize()

    # tol = 1e-01
    # tdim = mesh.topology.dim
    # fdim = tdim - 1

    # def Gamma_c_selector(x):
    #     return (
    #         np.isclose(x[2], 1, atol=tol)
    #         & (x[1] >= 0.5 - tol)
    #         & (x[1] <= 0.5 + tol)
    #         & (x[0] >= 0.5)
    #     )

    # Gamma_c = locate_entities_boundary(mesh, dim=fdim, marker=Gamma_c_selector)
    # Ic = locate_dofs_topological(fenics_le.V, fdim, Gamma_c)
    # Gamma_c_x = mesh.geometry.x[Ic].reshape(-1, tdim)

    # radii = np.sqrt((Gamma_c_x[:, 0] - 0.5) ** 2 + (Gamma_c_x[:, 1] - 0.5) ** 2)
    # order_radii = np.argsort(radii)
    # Ic_sorted = Ic[order_radii]

    # u2plot = uh.x.array[Ic_sorted * tdim + 2]
    # mapping = {dof: i for i, dof in enumerate(Ic_sorted)}

    # perm = np.array([mapping[dof] for dof in Ic])

    import matplotlib.pyplot as plt

    # Create the figure and axis
    # fig, ax = plt.subplots(figsize=(4, 3))

    # x = np.linspace(0.1, 0.5, len(radii))
    # # Plot the data
    # ax.plot(
    #     radii, np.abs(u2plot), "o-", label="displacement", markersize=6, linewidth=2
    # )
    # # ax.plot(x, 1 / x, "--", color="black", label="1/r")
    # # Logarithmic scale for better readability
    # # ax.set_xscale("log")
    # # ax.set_yscale("log")

    # # Labels and title
    # ax.set_xlabel("r distance to source point", fontsize=8)
    # ax.set_ylabel("u displacement", fontsize=8)

    # ax.set_title("u(r)", fontsize=16)

    # # Grid and legend
    # ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    # ax.legend(fontsize=8, loc="upper left")

    # # Improve layout
    # plt.tight_layout()

    # fig.savefig("u(r).png", format="png")

    # # Show the plot
    # plt.show()

    # start_brut = time.perf_counter()

    # S2, coords, _ = fenics_le.compute_S(
    #     selector=boundary_selector3, force_magnitude=-1e8, method="schur"
    # )
    # print(S2.size)
    # S, coords, _ = fenics_le.compute_S(
    #     selector=boundary_selector3, force_magnitude=-1e8, method="bem"
    # )
    # print(S.shape)

    # S1, coords1, _ = fenics_le.compute_S(
    #     selector=boundary_selector3, force_magnitude=-1e8, method="bruteforce"
    # )

    # print(f" Sbem = {S}")
    # print(f" Sbrute = {S1}")
    # print(f"error relative = {np.linalg.norm(S - S1) / np.linalg.norm(S1):.2f} %")
    # print(S)
    # print(S)

    # end_brut = time.perf_counter()
    # elapsed_time_brut = end_brut - start_brut

    # print(f"Brut time = {elapsed_time_brut:.6f} seconds")

    # start_schur = time.perf_counter()
    # Hschur = fenics_le.compute_S(
    #     selector=boundary_selector3, force_magnitude=-1e8, method="schur"
    # )
    # end_schur = time.perf_counter()
    # elapsed_time_schur = end_schur - start_schur
    # print(f"Schur time = {elapsed_time_schur:.6f} seconds")

    # fname = "out_elasticity/FlexData.npz"
    # data = np.load(fname)
    # H_Vlad = data["K"]

    ###############################################################################################
    # import matplotlib.pyplot as plt
    # import matplotlib.cm as cm

    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)

    # bar1 = ax1.imshow(Hbrut, cmap="viridis")
    # ax1.set
    # ax1.set_xlabel("Hbrut")

    # ax2 = fig.add_subplot(122)
    # ax2.imshow(Hschur, cmap="viridis")
    # ax2.set_xlabel("Hschur")
    # plt.show()

    # print(np.linalg.norm(Hschur - Hbrut) / np.linalg.norm(Hbrut))

    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    # # Data extracted from the image
    dofs = np.array([25, 100, 324, 400, 676, 961, 1296, 1521, 1681])
    x_lin = np.linspace(100, 1681, 10)
    brut_times = np.array(
        [
            0.013589,
            0.121507,
            1.809009,
            3.103817,
            9.302821,
            20.086458,
            38.231056,
            52.474020,
            65.840583,
            # 1200,
            # 1800,
            # 2400,
        ]
    )
    schur_times_lu = np.array(
        [
            0.014657,
            0.137595,
            1.738470,
            3.097349,
            10.606437,
            25.781998,
            60.205876,
            91.310023,
            138.289542,
        ]
    )

    schur_times_gmres = np.array(
        [
            0.054452,
            0.732104,
            7.769769,
            12.125511,
            30.394951,
            69.287577,
            148.345563,
            230.163766,
            317.156277,
        ]
    )

    # Shifted reference power curves to start at the same point as the first data point
    shift_value = schur_times_gmres[0]
    # power_1 = dofs / dofs[0] * shift_value
    # power_2 = (dofs / dofs[0]) ** 2 * shift_value
    power_1 = x_lin / x_lin[0] * shift_value
    power_2 = (x_lin / x_lin[0]) ** 2 * shift_value

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(dofs, brut_times, "o-", label="Direct Sampling", markersize=6, linewidth=2)
    ax.plot(
        dofs,
        schur_times_lu,
        "s-",
        label="Schur Compl. Direct (LU)",
        markersize=6,
        linewidth=2,
    )
    ax.plot(
        dofs,
        schur_times_gmres,
        "v-",
        label="Schur Compl. Iterative (GMRES)",
        markersize=6,
        linewidth=2,
    )

    ax.plot(x_lin, power_1, "--", color="black")  # label="O(N)")
    ax.plot(
        x_lin, power_2, "-.", color="black"
    )  # label="O(N²)")  # Annotate power curves

    ax.text(
        dofs[-1],
        power_1[-1],
        "O(N)",
        fontsize=8,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.text(
        dofs[-1],
        power_2[-1] - 10,
        "O(N²)",
        fontsize=8,
        color="black",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # Logarithmic scale for better readability
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=10)
    ax.set_ylabel("CPU Time (s)", fontsize=10)

    # ax.set_title("Comparison of Brute Force and Schur Complement Methods", fontsize=16)

    # Grid and legend
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left")

    # Improve layout
    plt.tight_layout()

    fig.savefig("schur_vs_sampling.pdf", format="pdf")

    # Show the plot
    plt.show()

    duration_1step = np.array(
        [
            0.00526,
            0.00760,
            0.00955,
            0.01083,
            0.01864,
            0.02415,
            0.03442,
            0.04143,
            0.04586,
        ]
    )

    duration_iteration = np.array(
        [
            0.00013,
            0.00076,
            0.00512,
            0.00626,
            0.01164,
            0.01841,
            0.02609,
            0.03157,
            0.03476,
        ]
    )

    relative_difference = np.divide(
        100 * (duration_1step - duration_iteration), duration_iteration
    )

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # Plot the data
    ax1.plot(
        dofs, duration_1step, "o-", label="Duration 1st it", markersize=6, linewidth=2
    )
    ax1.plot(
        dofs,
        duration_iteration,
        "s-",
        label="Duration of any other it",
        markersize=6,
        linewidth=2,
    )
    ax2.plot(
        dofs,
        relative_difference,
        "v-",
        label="Relative difference",
        markersize=6,
        linewidth=2,
    )

    # Labels and title
    ax1.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12)
    ax1.set_ylabel("CPU Time (s)", fontsize=12)

    ax1.set_title("1st iteration vs following iterations", fontsize=16)

    # Grid and legend
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend(fontsize=12, loc="upper left")

    # Labels and title
    ax2.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12)
    ax2.set_ylabel("CPU Time (s)", fontsize=12)

    ax2.set_title("1st iteration vs following iterations", fontsize=16)

    # Grid and legend
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend(fontsize=12, loc="upper left")

    fig
    plt.tight_layout()

    # Show the plot
    plt.show()

    ######################### Solve contact problem ####################

    # def _flat_indenter(x, y, x0, y0, R, z0):
    #     if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < R:
    #         return z0
    #     else:
    #         return z0 + 10.0

    # flat_indenter = np.vectorize(_flat_indenter)

    # def _parabolic_indenter(x, y, x0, y0, R, z0):
    #     if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > R:
    #         return z0 + R
    #     else:
    #         return z0 + R - np.sqrt(R**2 - (x - x0) ** 2 - (y - y0) ** 2)
    #         # return z0 + (x-x0)**2/(2*R) + (y-y0)**2/(2*R)

    # parabolic_indenter = np.vectorize(_parabolic_indenter)

    # # Constrained CG python
    # def constrained_CG(
    #     S,
    #     error_type,
    #     gap,
    #     max_iter,
    #     tolerance,
    #     pressure_factor=1e12,
    #     initial_pressure=None,
    # ):
    #     error_history = np.zeros((max_iter, 3))
    #     ub = -gap
    #     # Warmed start does not work well
    #     if initial_pressure is not None:
    #         # p = initial_pressure
    #         # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
    #         # p[gap>0] = 0
    #         p = np.maximum(-gap, 0) * pressure_factor
    #     else:
    #         p = np.zeros_like(ub)
    #         p = np.maximum(-gap, 0) * pressure_factor

    #     w = np.inner(S, p) - ub
    #     # w -= np.mean(w) #new
    #     t = w
    #     t_ = np.zeros_like(w)
    #     d = 0
    #     error = 1
    #     error_ = 1
    #     for iter in range(max_iter):
    #         if iter > 0:
    #             t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
    #             t[p <= 0] = 0
    #         q = np.inner(S, t)
    #         tau = np.inner(w, t) / np.inner(t, q)
    #         p = p - tau * t
    #         p = np.maximum(p, 0)
    #         zero_pressure = np.where(p == 0)[0]
    #         penetration = np.where(w < 0)[0]
    #         set_I = np.intersect1d(zero_pressure, penetration)
    #         if len(set_I) == 0:
    #             d = 1
    #         else:
    #             d = 0
    #             p[set_I] -= tau * w[set_I]
    #         t_ = t

    #         w = np.inner(S, p) - ub
    #         nw = np.linalg.norm(w, 2)

    #         error_ = error
    #         displ_error = np.linalg.norm(w[p > 0], 2) / nw
    #         ort = np.abs(np.dot(w, p) / nw)

    #         if error_type == "displacement":
    #             error = displ_error
    #         elif error_type == "mix":
    #             error = np.sqrt(displ_error * ort)
    #         elif error_type == "nw":
    #             error = nw
    #             if abs((error - error_) / error_) < tolerance:
    #                 error_history[iter, 0] = displ_error
    #                 error_history[iter, 1] = abs((error - error_) / error_)
    #                 error_history[iter, 2] = ort
    #                 return p, np.inner(S, p), error_history[: iter + 1]
    #         error_history[iter, 0] = displ_error
    #         error_history[iter, 1] = error
    #         error_history[iter, 2] = ort
    #         if error < tolerance:
    #             break
    #     return p, np.inner(S, p), error_history[: iter + 1]

    # tri = Triangulation(coords[:, 0], coords[:, 1])

    # # Vertical penetration of the indenter
    # displ = 0.15
    # # Indenter radius
    # Rindenter = 1.0

    # # Solve the problem
    # max_iter = 100
    # tolerance = 1e-5
    # error_type = "nw"
    # # pfactor is factor linking the trial pressure to the initial penetration for warmed-up start of the CG
    # pfactor = 1e8
    # # Number of frames for the animation
    # Nframes = 10

    # # Example with animation
    # ANIMATION = True
    # if ANIMATION == True:
    #     x_center = np.linspace(-0.3, 1.3, Nframes)
    #     for frame, xc in enumerate(x_center):
    #         contact_center = np.array([xc, 0.5])
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
    #                 coords[0, 2] - displ,
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
    #         np.savetxt(
    #             fname=f"output_{frame}.csv",
    #             X=output,
    #             header="X, Y, Z, disp",
    #             comments="",
    #             delimiter=",",
    #         )
    #         print(
    #             "Iters: {0:3d}, Error {1:.3e}".format(
    #                 len(error_history), error_history[-1, 1]
    #             )
    #         )

    #         ## Plot using Matplotlib

    #         # Set viewing angles
    #         elevation_angle = 45  # Lower number to "raise" the camera view
    #         azimuth_angle = -45  # Adjust as needed

    #         plt.rcParams["figure.figsize"] = [10, 5]
    #         fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

    #         ax[0].view_init(elev=elevation_angle, azim=azimuth_angle)
    #         ax[1].view_init(elev=elevation_angle, azim=azimuth_angle)

    #         ax[0].set_xlim([-0.0, 1.0])
    #         ax[0].set_ylim([-0.0, 1.0])
    #         ax[0].set_zlim([-displ, 0])
    #         x = np.linspace(-0.0, 1.0, 100)
    #         y = np.linspace(-0.0, 1.0, 100)
    #         X, Y = np.meshgrid(x, y)
    #         Z = (
    #             parabolic_indenter(
    #                 X,
    #                 Y,
    #                 contact_center[0],
    #                 contact_center[1],
    #                 Rindenter,
    #                 coords[0, 2] - displ,
    #             )
    #             - coords[0, 2]
    #         )

    #         # Z = (
    #         #     flat_indenter(
    #         #         X,
    #         #         Y,
    #         #         contact_center[0],
    #         #         contact_center[1],
    #         #         Rindenter,
    #         #         coords[0, 2] - displ,
    #         #     )
    #         #     - coords[0, 2]
    #         # )
    #         surf1 = ax[0].plot_trisurf(
    #             coords[:, 0],
    #             coords[:, 1],
    #             -displacement,
    #             triangles=tri.triangles,
    #             cmap="coolwarm",
    #             vmin=-displ,
    #             vmax=0,
    #         )
    #         cb1 = fig.colorbar(
    #             surf1, ax=ax[0], shrink=0.6, aspect=10, orientation="horizontal"
    #         )
    #         cb1.set_label("$u_z$")
    #         ax[0].plot_surface(
    #             X,
    #             Y,
    #             Z,
    #             alpha=0.1,
    #             cmap="gray",
    #             rcount=X.shape[0],
    #             ccount=X.shape[1],
    #             edgecolor="k",
    #             linewidth=0.1,
    #         )
    #         ax[0].set_title("Vertical displacement")

    #         ax[1].set_xlim([0, 1])
    #         ax[1].set_ylim([0, 1])
    #         # ax[1].set_zlim([0,16])
    #         surf2 = ax[1].plot_trisurf(
    #             coords[:, 0],
    #             coords[:, 1],
    #             pressure,
    #             triangles=tri.triangles,
    #             cmap="coolwarm",
    #             vmin=0,
    #             vmax=1.5e6,
    #         )
    #         cb2 = fig.colorbar(
    #             surf2, ax=ax[1], shrink=0.6, aspect=10, orientation="horizontal"
    #         )
    #         cb2.set_label("$p/E$")
    #         ax[1].set_title("Pressure/Young's modulus")
    #         fig.tight_layout()
    #         fig.savefig("Contact_cone_{0:03d}.png".format(frame), dpi=300)

    ##########################################Contact with tangential disp################################################

    # def _flat_indenter(x, y, x0, y0, R, z0):
    #     if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < R:
    #         return z0
    #     else:
    #         return z0 + 10.0

    # flat_indenter = np.vectorize(_flat_indenter)

    # def _parabolic_indenter(x, y, x0, y0, R, z0):
    #     if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) > R:
    #         return z0 + R
    #     else:
    #         return z0 + R - np.sqrt(R**2 - (x - x0) ** 2 - (y - y0) ** 2)
    #         # return z0 + (x-x0)**2/(2*R) + (y-y0)**2/(2*R)

    # parabolic_indenter = np.vectorize(_parabolic_indenter)

    # # Constrained CG python
    # def constrained_CG(
    #     S,
    #     error_type,
    #     gap,
    #     normals,
    #     max_iter,
    #     tolerance,
    #     pressure_factor=1e12,
    #     initial_pressure=None,
    # ):
    #     error_history = np.zeros((max_iter, 3))
    #     ub = -gap
    #     # Warmed start does not work well
    #     if initial_pressure is not None:
    #         # p = initial_pressure
    #         # p[np.logical_and(gap<0, p == 0)] = pressure_factor * gap[np.logical_and(gap<0, p == 0)]
    #         # p[gap>0] = 0
    #         p = np.maximum(-gap, 0) * pressure_factor
    #     else:
    #         p = np.zeros_like(ub)
    #         p = np.maximum(-gap, 0) * pressure_factor

    #     w = np.einsum("ijm,j,im->i", S, p, normals)  # np.inner(S, p) - ub
    #     # w -= np.mean(w) #new
    #     t = w
    #     t_ = np.zeros_like(w)
    #     d = 0
    #     error = 1
    #     error_ = 1
    #     for iter in range(max_iter):
    #         if iter > 0:
    #             t[p > 0] = w[p > 0] + d * error / error_ * t_[p > 0]
    #             t[p <= 0] = 0
    #         q = np.einsum("ijm,j,im->i", S, t, normals)  # np.inner(S, t)
    #         tau = np.inner(w, t) / np.inner(t, q)
    #         p = p - tau * t
    #         p = np.maximum(p, 0)
    #         zero_pressure = np.where(p == 0)[0]
    #         penetration = np.where(w < 0)[0]
    #         set_I = np.intersect1d(zero_pressure, penetration)
    #         if len(set_I) == 0:
    #             d = 1
    #         else:
    #             d = 0
    #             p[set_I] -= tau * w[set_I]
    #         t_ = t

    #         w = np.einsum("ijm,j,im->i", S, p, normals) - ub
    #         nw = np.linalg.norm(w, 2)

    #         error_ = error
    #         displ_error = np.linalg.norm(w[p > 0], 2) / nw
    #         ort = np.abs(np.dot(w, p) / nw)

    #         if error_type == "displacement":
    #             error = displ_error
    #         elif error_type == "mix":
    #             error = np.sqrt(displ_error * ort)
    #         elif error_type == "nw":
    #             error = nw
    #             if abs((error - error_) / error_) < tolerance:
    #                 error_history[iter, 0] = displ_error
    #                 error_history[iter, 1] = abs((error - error_) / error_)
    #                 error_history[iter, 2] = ort
    #                 return (
    #                     p,
    #                     np.einsum("ijm,j,im->i", S, p, normals),
    #                     error_history[: iter + 1],
    #                 )
    #         error_history[iter, 0] = displ_error
    #         error_history[iter, 1] = error
    #         error_history[iter, 2] = ort
    #         if error < tolerance:
    #             break
    #     return p, np.einsum("ijm,j,im->i", S, p, normals), error_history[: iter + 1]

    # tri = Triangulation(coords[:, 0], coords[:, 1])

    # normals = np.full_like(coords, np.array([0, 0, -1]))
    # # Vertical penetration of the indenter
    # displ = 0.15
    # # Indenter radius
    # Rindenter = 1.0

    # # Solve the problem
    # max_iter = 100
    # tolerance = 1e-5
    # error_type = "nw"
    # # pfactor is factor linking the trial pressure to the initial penetration for warmed-up start of the CG
    # pfactor = 1e8
    # # Number of frames for the animation
    # Nframes = 10

    # # Example with animation
    # ANIMATION = True
    # if ANIMATION == True:
    #     x_center = np.linspace(-0.3, 1.3, Nframes)
    #     for frame, xc in enumerate(x_center):
    #         contact_center = np.array([xc, 0.5])
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
    #                 coords[0, 2] - displ,
    #             )
    #             - coords[:, 2]
    #         )
    #         penetrating_nodes = np.where(gap < 0)[0]

    #         # Solve the problem
    #         start = time.time()
    #         # pfactor = 1. # for conical indenter
    #         if frame == 0:
    #             pressure, displacement, error_history = constrained_CG(
    #                 S, error_type, gap, normals, max_iter, tolerance, pfactor
    #             )
    #         else:
    #             pressure, displacement, error_history = constrained_CG(
    #                 S,
    #                 error_type,
    #                 gap,
    #                 normals,
    #                 max_iter,
    #                 tolerance,
    #                 pfactor,
    #                 pressure,
    #             )
    #         print(
    #             "Iters: {0:3d}, Error {1:.3e}".format(
    #                 len(error_history), error_history[-1, 1]
    #             )
    #         )

    #         ## Plot using Matplotlib

    #         # Set viewing angles
    #         elevation_angle = 60  # Lower number to "raise" the camera view
    #         azimuth_angle = -60  # Adjust as needed

    #         plt.rcParams["figure.figsize"] = [10, 5]
    #         fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

    #         ax[0].view_init(elev=elevation_angle, azim=azimuth_angle)
    #         ax[1].view_init(elev=elevation_angle, azim=azimuth_angle)

    #         ax[0].set_xlim([-0.0, 1.0])
    #         ax[0].set_ylim([-0.0, 1.0])
    #         ax[0].set_zlim([-displ, 0])
    #         x = np.linspace(-0.0, 1.0, 100)
    #         y = np.linspace(-0.0, 1.0, 100)
    #         X, Y = np.meshgrid(x, y)
    #         Z = (
    #             parabolic_indenter(
    #                 X,
    #                 Y,
    #                 contact_center[0],
    #                 contact_center[1],
    #                 Rindenter,
    #                 coords[0, 2] - displ,
    #             )
    #             - coords[0, 2]
    #         )

    #         # Z = (
    #         #     flat_indenter(
    #         #         X,
    #         #         Y,
    #         #         contact_center[0],
    #         #         contact_center[1],
    #         #         Rindenter,
    #         #         coords[0, 2] - displ,
    #         #     )
    #         #     - coords[0, 2]
    #         # )
    #         surf1 = ax[0].plot_trisurf(
    #             coords[:, 0],
    #             coords[:, 1],
    #             -displacement,
    #             triangles=tri.triangles,
    #             cmap="coolwarm",
    #             vmin=-displ,
    #             vmax=0,
    #         )
    #         cb1 = fig.colorbar(
    #             surf1, ax=ax[0], shrink=0.6, aspect=10, orientation="horizontal"
    #         )
    #         cb1.set_label("$u_z$")
    #         ax[0].plot_surface(
    #             X,
    #             Y,
    #             Z,
    #             alpha=0.1,
    #             cmap="gray",
    #             rcount=X.shape[0],
    #             ccount=X.shape[1],
    #             edgecolor="k",
    #             linewidth=0.1,
    #         )
    #         ax[0].set_title("Vertical displacement")

    #         ax[1].set_xlim([0, 1])
    #         ax[1].set_ylim([0, 1])
    #         # ax[1].set_zlim([0,16])
    #         surf2 = ax[1].plot_trisurf(
    #             coords[:, 0],
    #             coords[:, 1],
    #             pressure,
    #             triangles=tri.triangles,
    #             cmap="coolwarm",
    #             vmin=0,
    #             vmax=1.5e6,
    #         )
    #         cb2 = fig.colorbar(
    #             surf2, ax=ax[1], shrink=0.6, aspect=10, orientation="horizontal"
    #         )
    #         cb2.set_label("$p/E$")
    #         ax[1].set_title("Pressure/Young's modulus")
    #         fig.tight_layout()
    #         fig.savefig("Contact_cone_{0:03d}.png".format(frame), dpi=300)

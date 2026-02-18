import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import Function, functionspace, locate_dofs_topological, form
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_matrix_mat,
    assemble_vector,
    apply_lifting,
    LinearProblem,
)
from ufl import TrialFunction, TestFunction, inner, FacetNormal, dx, ds

from ..contact.lcp_solvers.ccg import CCG
from ..contact.lcp_solvers.lemke import lemkelcp
from ..contact.Sc_normal import Sc_normal


class Contact_normal:
    def __init__(
        self,
        mesh,
        indenter,
        tc: Function,
        Gamma_c,
        ds,
        Gamma_c_id: int,
        problem: LinearProblem,
        solver: str = "ccg",
        normals_eps: float = 1e-8,
        f0_sampling: float = 1e9,
    ):
        self.mesh = mesh
        self.indenter = indenter
        self.tc = tc
        self.Gamma_c = Gamma_c
        self.ds = ds
        self.Gamma_c_id = int(Gamma_c_id)
        self.problem = problem
        self.solver = solver

        self.comm = self.mesh.comm
        self.tdim = self.mesh.topology.dim
        self.fdim = self.tdim - 1

        self.V = self.tc.function_space
        self.Ic = None
        self.nc = None
        self.normals = None

        self.dofs_c_W = None
        self.dofs_c_Vx = None
        self.dofs_c_Vy = None
        self.dofs_c_Vz = None

        self.Mcc = None
        self.ksp_Mcc = None
        self._rhs_arr = None
        self._sol_arr = None
        self._rhs = None
        self._sol = None

        self.Snn = None
        self.fc = None

        # Build pipeline
        self._build_contact_sets()
        self._build_normals(eps=normals_eps)
        self._build_Snn(f0=f0_sampling)
        self._build_Mcc()

    def _build_contact_sets(self):
        Wz, Wz_to_V = self.V.sub(2).collapse()
        self.Wz = Wz
        self.Wz_to_V = np.asarray(Wz_to_V, dtype=np.int32)

        dofs_c_W = locate_dofs_topological(Wz, self.fdim, self.Gamma_c)
        self.dofs_c_W = np.asarray(dofs_c_W, dtype=np.int32)

        self.Ic = self.dofs_c_W.astype(np.int64)
        self.nc = int(self.Ic.size)

        self.dofs_c_Vx = self._vertex_dofs_in_V(comp=0, vertex_ids=self.Ic)
        self.dofs_c_Vy = self._vertex_dofs_in_V(comp=1, vertex_ids=self.Ic)
        self.dofs_c_Vz = self._vertex_dofs_in_V(comp=2, vertex_ids=self.Ic)

    def _vertex_dofs_in_V(self, comp: int, vertex_ids: np.ndarray) -> np.ndarray:
        Wc, Wc_to_V = self.V.sub(comp).collapse()
        Wc_to_V = np.asarray(Wc_to_V, dtype=np.int32)

        dofs_c_Wc = locate_dofs_topological(Wc, 0, vertex_ids)
        dofs_c_Wc = np.asarray(dofs_c_Wc, dtype=np.int32)

        return Wc_to_V[dofs_c_Wc]

    def _build_normals(self, eps: float = 1e-8):
        gdim = self.mesh.geometry.dim
        Vn = functionspace(self.mesh, ("CG", 1, (gdim,)))

        n = FacetNormal(self.mesh)
        u = TrialFunction(Vn)
        v = TestFunction(Vn)

        a = eps * inner(u, v) * dx + inner(u, v) * self.ds(self.Gamma_c_id)
        L = inner(n, v) * self.ds(self.Gamma_c_id)

        normal_fn = Function(Vn)
        normal_fn.name = "contact_normals"

        proj = LinearProblem(
            a=a,
            L=L,
            bcs=[],
            u=normal_fn,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        proj.solve()
        normal_fn.x.scatter_forward()
        self.normal_fn = normal_fn
        normals = np.zeros((self.nc, gdim), dtype=float)
        for comp in range(gdim):
            Vc, Vc_to_Vn = Vn.sub(comp).collapse()
            Vc_to_Vn = np.asarray(Vc_to_Vn, dtype=np.int32)

            dofs_comp = locate_dofs_topological(Vc, 0, self.Ic)
            dofs_comp = np.asarray(dofs_comp, dtype=np.int32)
            parent_dofs = Vc_to_Vn[dofs_comp]

            normals[:, comp] = normal_fn.x.array[parent_dofs]

        nrm = np.linalg.norm(normals, axis=1)
        if np.any(nrm == 0.0):
            raise ValueError("Some computed normals have zero norm on the contact set.")
        normals = normals / nrm[:, None]

        self.normals = normals

    def _build_Snn(self, f0: float = 1e9):
        self.problem._A.zeroEntries()
        assemble_matrix_mat(self.problem._A, self.problem._a, bcs=self.problem.bcs)
        self.problem._A.assemble()

        with self.problem._b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.problem._b, self.problem._L)

        apply_lifting(self.problem._b, [self.problem._a], bcs=[self.problem.bcs])
        self.problem._b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        for bc in self.problem.bcs:
            bc.set(self.problem._b.array_w)

        self.Snn = Sc_normal(
            A=self.problem.A,
            b=self.problem.b,
            Ic=self.Ic,
            normals=self.normals,
            tdim=self.tdim,
            f0=f0,
        ).sample_n()

    def _build_Mcc(self):
        p = TrialFunction(self.Wz)
        q = TestFunction(self.Wz)
        m_ufl = inner(p, q) * self.ds(self.Gamma_c_id)
        m_form = form(m_ufl)

        M = assemble_matrix(m_form)
        M.assemble()

        is_c = PETSc.IS().createGeneral(self.dofs_c_W, comm=self.comm)
        self.Mcc = M.createSubMatrix(is_c, is_c)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.Mcc)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setUp()
        self.ksp_Mcc = ksp

        n = len(self.dofs_c_W)
        self._rhs_arr = np.zeros(n, dtype=PETSc.ScalarType)
        self._sol_arr = np.zeros(n, dtype=PETSc.ScalarType)
        self._rhs = PETSc.Vec().createWithArray(self._rhs_arr, comm=self.comm)
        self._sol = PETSc.Vec().createWithArray(self._sol_arr, comm=self.comm)

    def fc_to_tc(self, fc: np.ndarray) -> np.ndarray:
        self._rhs_arr[:] = -fc
        self.ksp_Mcc.solve(self._rhs, self._sol)
        return self._sol_arr

    def pts_c(self) -> np.ndarray:
        return self.mesh.geometry.x[self.Ic].reshape(-1, self.tdim)

    def gap_n(self) -> np.ndarray:
        pts = self.pts_c()
        return self.indenter.new_gap_n(pts, self.normals, warn_dist=0.3)

    def solve(self, max_iter=1000, tol=1e-6, *args, **kwargs):
        g = self.gap_n()

        if self.solver == "ccg":
            lcp_solver = CCG(
                Sc=self.Snn,
                err_type="displacement",
                g=g,
                max_iter=max_iter,
                tol=tol,
                *args,
                **kwargs,
            )
            self.fc, _, _ = lcp_solver.solve()
        elif self.solver == "lemke":
            self.fc, _, _ = lemkelcp(self.Snn, g, maxIter=max_iter)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def apply_contact_forces(self):
        self.tc.x.array[:] = 0.0

        if self.fc is None:
            raise ValueError(
                "Contact forces have not been computed. Call solve() first."
            )

        t_scalar = self.fc_to_tc(self.fc)  # (nc,)
        n = self.normals  # (nc,3)

        self.tc.x.array[self.dofs_c_Vx] = t_scalar * n[:, 0]
        self.tc.x.array[self.dofs_c_Vy] = t_scalar * n[:, 1]
        self.tc.x.array[self.dofs_c_Vz] = t_scalar * n[:, 2]

        self.tc.x.scatter_forward()

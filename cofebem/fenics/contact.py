from cofebem.contact.Sc import Sc
from cofebem.contact.lcp_solvers.ccg import CCG
from cofebem.contact.lcp_solvers.lemke import lemkelcp
from petsc4py import PETSc
from dolfinx.fem import locate_dofs_topological, form
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_matrix_mat,
    assemble_vector,
    apply_lifting,
)
from ufl import TrialFunction, TestFunction, inner

import numpy as np


class Contact:
    def __init__(
        self,
        mesh,
        indenter,
        tc,
        Gamma_c,
        ds,
        Gamma_c_id,
        problem,
        solver="ccg",
        save_matrix=True,
    ):
        self.mesh = mesh
        self.indenter = indenter
        self.tc = tc
        self.Gamma_c = Gamma_c
        self.Gamma_c_dofs = locate_dofs_topological(
            tc.function_space, mesh.topology.dim - 1, Gamma_c
        )
        self.ds = ds
        self.Gamma_c_id = Gamma_c_id
        self.solver = solver
        self.problem = problem
        self.Mcc = None
        self.Sc = None
        self.fc = None
        self.save_matrix = save_matrix
        self.build_Sc()
        self.build_Mcc()

    def build_Mcc(self):
        V = self.tc.function_space
        comm = self.mesh.comm
        fdim = self.mesh.topology.dim - 1

        W, W_to_V = V.sub(2).collapse()

        dofs_c_W = locate_dofs_topological(W, fdim, self.Gamma_c)
        self.dofs_c_W = np.asarray(dofs_c_W, dtype=np.int32)

        self.dofs_c_V = np.asarray(W_to_V, dtype=np.int32)[self.dofs_c_W]

        p = TrialFunction(W)
        q = TestFunction(W)
        m_ufl = inner(p, q) * self.ds(self.Gamma_c_id)
        m_form = form(m_ufl)

        M = assemble_matrix(m_form)
        M.assemble()

        is_c = PETSc.IS().createGeneral(self.dofs_c_W, comm=comm)
        self.Mcc = M.createSubMatrix(is_c, is_c)
        self.is_c = is_c

        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(self.Mcc)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setUp()
        self.ksp_Mcc = ksp

        n = len(self.dofs_c_W)
        self._rhs_arr = np.zeros(n, dtype=PETSc.ScalarType)
        self._sol_arr = np.zeros(n, dtype=PETSc.ScalarType)
        self._rhs = PETSc.Vec().createWithArray(self._rhs_arr, comm=comm)
        self._sol = PETSc.Vec().createWithArray(self._sol_arr, comm=comm)

    def fc_to_tc(self, fc):
        self._rhs_arr[:] = fc  # Remeber to change to -fc
        self.ksp_Mcc.solve(self._rhs, self._sol)
        return self._sol_arr

    def build_Sc(self):
        self.problem._A.zeroEntries()
        assemble_matrix_mat(self.problem._A, self.problem._a, bcs=self.problem.bcs)
        self.problem._A.assemble()

        with self.problem._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self.problem._b, self.problem._L)

        apply_lifting(self.problem._b, [self.problem._a], bcs=[self.problem.bcs])
        self.problem._b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        for bc in self.problem.bcs:
            bc.set(self.problem._b.array_w)

        self.Sc = Sc(
            self.problem.A, self.problem.b, self.mesh.topology.dim, self.Gamma_c_dofs
        ).by_sampling()

        # if self.save_matrix:
        #     self.Sc.save()

    def g(self):
        pts = self.mesh.geometry.x[self.Gamma_c_dofs].reshape(
            -1, self.mesh.topology.dim
        )
        return self.indenter.gap(pts)

    def solve(self, max_iter=1000, tol=1e-6, pfactor=1e12, p0=None, *args, **kwargs):
        g = self.g()
        if self.solver == "ccg":
            lcp_solver = CCG(
                Sc=self.Sc,
                err_type="displacement",
                g=g,
                max_iter=max_iter,
                tol=tol,
                *args,
                **kwargs,
            )
            self.fc, _, _ = lcp_solver.solve()
        elif self.solver == "lemke":
            self.fc, _, _ = lemkelcp(self.Sc, g, maxIter=max_iter)

    def apply_contact_forces(self):
        self.tc.x.array[:] = 0.0
        if self.fc is None:
            raise ValueError(
                "Contact forces have not been computed. Call solve() first."
            )
        tc_vals = self.fc_to_tc(self.fc)
        self.tc.x.array[self.dofs_c_V] = tc_vals
        self.tc.x.scatter_forward()

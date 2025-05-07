import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import typing
import ufl
from cofebem.fem.backends.base import FEMWraper
from cofebem.fem.fem import FEM


class Fenics(FEMWraper):
    def __init__(self, fem: FEM):
        self.mesh = mesh
        self.V = function_space
        self.comm = MPI.COMM_WORLD

        # Placeholders for forms and solution
        self.a_form = None
        self.L_form = None
        self.bcs = []
        self.solution = fem.Function(self.V)

        # Assembled matrix and vector
        self.A = None
        self.b = None

    def assemble_system(self, bilinear_form, linear_form):
        # Store forms
        self.a_form = bilinear_form
        self.L_form = linear_form

        # Assemble matrix
        self.A = fem.petsc.assemble_matrix(self.a_form, self.bcs)
        self.A.assemble()

        # Assemble RHS vector
        self.b = fem.petsc.assemble_vector(self.L_form)
        fem.petsc.apply_lifting(self.b, [self.a_form], [self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b, self.bcs)

    def apply_boundary_conditions(self, bcs):
        self.bcs = bcs

    def solve(self):
        # Create linear solver and solve
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(self.A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.solve(self.b, self.solution.vector)

    def get_solution(self):
        return self.solution

    def reset(self):
        # Reset solution and forms
        self.solution.vector.set(0.0)
        self.a_form = None
        self.L_form = None
        self.A = None
        self.b = None
        self.bcs = []

    @property
    def function_space(self):
        return self.V

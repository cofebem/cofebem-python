import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from tqdm import tqdm

from cofebem.utils.linalg.schur_complement import schur_complement


class Sc:
    def __init__(self, A, b, tdim, Ic, full_Sc=False, single_direction=2):
        self.A = A
        self.b = b
        self.comm = MPI.COMM_WORLD
        self.tdim = tdim
        self.Ic = Ic
        self.full_Sc = full_Sc
        self.single_direction = single_direction
        self.dense = None

    def by_sampling(self):
        f_magnitude = 1e9

        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(self.A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setFromOptions()
        solver.setUp()

        b = self.b.copy()
        uh = PETSc.Vec().createMPI(b.getSize(), comm=self.comm)

        n_c = len(self.Ic)

        if self.full_Sc:
            full_dofs = np.array(
                [
                    vertex * self.tdim + comp
                    for vertex in self.Ic
                    for comp in range(self.tdim)
                ],
                dtype=np.int32,
            )
            Sc = np.zeros((self.tdim * n_c, self.tdim * n_c), dtype=PETSc.ScalarType)
            for i, dof_applied in enumerate(
                tqdm(full_dofs, desc="Computing Contact Compliance Matrix", unit="it")
            ):
                b.set(0)
                b.setValue(
                    dof_applied,
                    f_magnitude,
                )
                b.assemble()

                solver.solve(b, uh)

                Sc[i, :] = uh.array[full_dofs] / f_magnitude

        else:
            selected_dofs = self.Ic * self.tdim + self.single_direction
            Sc = np.zeros((n_c, n_c), dtype=PETSc.ScalarType)
            for i, dof_applied in enumerate(
                tqdm(
                    selected_dofs, desc="Computing Contact Compliance Matrix", unit="it"
                )
            ):
                b.set(0)
                b.setValue(
                    dof_applied,
                    f_magnitude,
                )
                b.assemble()

                solver.solve(b, uh)

                Sc[i, :] = uh.array[selected_dofs] / f_magnitude

        self.dense = Sc
        return self.dense

    def by_schur(self):
        K = self.problem.A.convert("dense").getDenseArray()

        # Partition the global matrix into blocks
        all_dofs = np.arange(K.shape[0])
        uc_dofs = np.asarray(self.Ic)
        uv_dofs = np.setdiff1d(all_dofs, uc_dofs)

        Kvv = K[np.ix_(uv_dofs, uv_dofs)]
        Kvc = K[np.ix_(uv_dofs, uc_dofs)]
        Kcv = K[np.ix_(uc_dofs, uv_dofs)]
        Kcc = K[np.ix_(uc_dofs, uc_dofs)]

        Sc = np.linalg.inv(schur_complement(Kvv, Kvc, Kcv, Kcc))

        self.dense = Sc
        return self.dense

    def save(self, file="Sc.npy"):
        np.save(file, self.dense)
        print(f"Contact compliance matrix successfully saved to {file}")

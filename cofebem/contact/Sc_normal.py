import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm


class Sc_normal:
    def __init__(self, A: PETSc.Mat, b: PETSc.Vec, Ic, normals, tdim=3, f0=1e9):
        self.A = A
        self.comm = MPI.COMM_WORLD
        self.tdim = int(tdim)
        self.f0 = float(f0)

        self.Ic = np.asarray(Ic, dtype=np.int64)
        self.nc = int(self.Ic.size)

        normals = np.asarray(normals, dtype=float)
        if normals.shape != (self.nc, self.tdim):
            raise ValueError(f"normals must have shape ({self.nc}, {self.tdim}).")

        nrm = np.linalg.norm(normals, axis=1)
        if np.any(nrm == 0.0):
            raise ValueError("Some normals have zero norm.")
        self.n = normals / nrm[:, None]

        self.rhs = b.duplicate()
        self.rhs.set(0.0)
        self.u = b.duplicate()

        self.cdofs = np.stack(
            [self.Ic * self.tdim + c for c in range(self.tdim)], axis=1
        ).astype(np.int32)
        self.cdofs_flat = self.cdofs.reshape(-1).astype(np.int32)

        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.setFromOptions()
        self.ksp.setUp()

        self._uc = PETSc.Vec().createSeq(self.tdim * self.nc, comm=PETSc.COMM_SELF)
        is_from = PETSc.IS().createGeneral(self.cdofs_flat, comm=self.comm)
        is_to = PETSc.IS().createStride(
            self.tdim * self.nc, first=0, step=1, comm=PETSc.COMM_SELF
        )
        self._scat = PETSc.Scatter().create(self.u, is_from, self._uc, is_to)

        self.Snn = None

    def _set_nload(self, j: int):
        dofs = self.cdofs[j]  # (3,)
        vals = self.f0 * self.n[j]  # (3,)

        self.rhs.set(0.0)
        self.rhs.setValues(dofs, vals, addv=PETSc.InsertMode.INSERT_VALUES)
        self.rhs.assemble()

    def _get_uc_xyz(self) -> np.ndarray:
        self._uc.set(0.0)
        self._scat.scatter(
            self.u,
            self._uc,
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )
        return self._uc.getArray(readonly=True).copy()

    def _proj_n(self, uc_xyz: np.ndarray) -> np.ndarray:
        U = uc_xyz.reshape(self.nc, self.tdim)
        return np.einsum("ij,ij->i", self.n, U)

    def sample_n(self, show=True) -> np.ndarray:
        S = np.zeros((self.nc, self.nc), dtype=PETSc.ScalarType)

        it = range(self.nc)
        if show:
            it = tqdm(it, desc="Sampling Snn (normal)", unit="it")

        for j in it:
            self._set_nload(j)
            self.ksp.solve(self.rhs, self.u)

            uc_xyz = self._get_uc_xyz()
            u_n = self._proj_n(uc_xyz)

            S[:, j] = u_n / self.f0

        self.Snn = S
        return S

    def save(self, file="Snn.npy"):
        if self.Snn is None:
            raise RuntimeError("Nothing to save: run sample_n() first.")
        np.save(file, self.Snn)
        if self.comm.rank == 0:
            print(f"Saved Snn to {file}")

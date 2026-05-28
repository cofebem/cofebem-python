import numpy as np
import mfem.ser as mfem
from scipy.sparse import csr_matrix


def mfem_sparse_to_scipy(A_mfem):
    I = np.array(A_mfem.GetIArray(), copy=True)
    J = np.array(A_mfem.GetJArray(), copy=True)
    data = np.array(A_mfem.GetDataArray(), copy=True)

    return csr_matrix((data, J, I), shape=(A_mfem.Height(), A_mfem.Width()))


def main():
    order = 1
    nx = ny = nz = 2

    E = 1.0
    nu = 0.3

    mu = E / (2.0 * (1.0 + nu))
    lamb = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    mesh = mfem.Mesh.MakeCartesian3D(nx, ny, nz, mfem.Element.HEXAHEDRON, 1.0, 1.0, 1.0)

    dim = mesh.Dimension()

    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec, dim)

    print("Number of FE unknowns:", fespace.GetVSize())
    print("Number of true unknowns:", fespace.GetTrueVSize())

    ess_bdr = mfem.intArray([1] * mesh.bdr_attributes.Max())
    ess_tdof_list = mfem.intArray()
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    f = mfem.VectorArrayCoefficient(dim)
    f.Set(0, mfem.ConstantCoefficient(0.0))
    f.Set(1, mfem.ConstantCoefficient(0.0))
    f.Set(2, mfem.ConstantCoefficient(-1.0))

    b = mfem.LinearForm(fespace)
    b.Assemble()

    lambda_coef = mfem.ConstantCoefficient(lamb)
    mu_coef = mfem.ConstantCoefficient(mu)

    a = mfem.BilinearForm(fespace)

    a.Assemble()
    a.Finalize()

    K_mfem = a.SpMat()
    K = mfem_sparse_to_scipy(K_mfem)

    print("Raw stiffness matrix K:")
    print("  shape =", K.shape)
    print("  nnz   =", K.nnz)

    A = mfem.OperatorPtr()
    X = mfem.Vector()
    B = mfem.Vector()

    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

    A_mfem = mfem.OperatorHandle2SparseMatrix(A)
    A_bc = mfem_sparse_to_scipy(A_mfem)

    print("Constrained system matrix A after Dirichlet elimination:")
    print("  shape =", A_bc.shape)
    print("  nnz   =", A_bc.nnz)

    # -----------------------------
    # Solve A X = B
    # -----------------------------
    smoother = mfem.GSSmoother(A_mfem)

    mfem.PCG(
        A_mfem,
        smoother,
        B,
        X,
        1,  # print iterations
        500,  # max iterations
        1e-12,  # relative tolerance
        0.0,  # absolute tolerance
    )

    a.RecoverFEMSolution(X, b, x)

    x.Save("u.gf")
    mesh.Print("mesh.mesh")

    print("Solution saved to u.gf")
    print("Mesh saved to mesh.mesh")


if __name__ == "__main__":
    main()

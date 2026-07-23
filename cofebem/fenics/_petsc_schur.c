#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include <petsc4py/petsc4py.h>
#include <petscmat.h>

static const char *capsule_name = "cofebem.PetscSchurFactor";

static void destroy_factor(PyObject *capsule)
{
    Mat *factor = (Mat *)PyCapsule_GetPointer(capsule, capsule_name);
    if (factor != NULL) {
        MatDestroy(factor);
        PyMem_Free(factor);
    } else {
        PyErr_Clear();
    }
}

static int raise_petsc_error(PetscErrorCode ierr, const char *operation)
{
    if (!ierr) return 0;
    PyErr_Format(
        PyExc_RuntimeError,
        "%s failed with PETSc error code %d",
        operation,
        (int)ierr
    );
    return -1;
}

static Mat *get_factor(PyObject *capsule)
{
    return (Mat *)PyCapsule_GetPointer(capsule, capsule_name);
}

static PyObject *create_factor(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *py_a = NULL;
    PyObject *py_is = NULL;
    const char *factor_type = "lu";
    static char *keywords[] = {"matrix", "schur_is", "factor_type", NULL};
    Mat A = NULL;
    IS schur_is = NULL;
    Mat *factor = NULL;
    MatFactorInfo info;
    PetscErrorCode ierr = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO|s:create_factor", keywords,
            &py_a, &py_is, &factor_type)) {
        return NULL;
    }
    if (!PyObject_TypeCheck(py_a, &PyPetscMat_Type) ||
        !PyObject_TypeCheck(py_is, &PyPetscIS_Type)) {
        PyErr_SetString(
            PyExc_TypeError,
            "matrix and schur_is must be petsc4py PETSc.Mat and PETSc.IS objects"
        );
        return NULL;
    }
    A = PyPetscMat_Get(py_a);
    schur_is = PyPetscIS_Get(py_is);
    factor = (Mat *)PyMem_Calloc(1, sizeof(Mat));
    if (factor == NULL) return PyErr_NoMemory();

    MatFactorInfoInitialize(&info);
    if (strcmp(factor_type, "lu") == 0) {
        ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU, factor);
        if (!ierr) ierr = MatFactorSetSchurIS(*factor, schur_is);
        if (!ierr) ierr = MatLUFactorSymbolic(*factor, A, NULL, NULL, &info);
        if (!ierr) ierr = MatLUFactorNumeric(*factor, A, &info);
    } else if (strcmp(factor_type, "cholesky") == 0) {
        ierr = MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, factor);
        if (!ierr) ierr = MatFactorSetSchurIS(*factor, schur_is);
        if (!ierr) ierr = MatCholeskyFactorSymbolic(*factor, A, NULL, &info);
        if (!ierr) ierr = MatCholeskyFactorNumeric(*factor, A, &info);
    } else {
        PyMem_Free(factor);
        PyErr_SetString(PyExc_ValueError, "factor_type must be 'lu' or 'cholesky'");
        return NULL;
    }
    if (!ierr) ierr = MatFactorFactorizeSchurComplement(*factor);
    if (raise_petsc_error(ierr, "MUMPS selected-Schur factorization") < 0) {
        MatDestroy(factor);
        PyMem_Free(factor);
        return NULL;
    }
    return PyCapsule_New(factor, capsule_name, destroy_factor);
}

static PyObject *solve_impl(PyObject *args, PetscBool schur)
{
    PyObject *capsule = NULL;
    PyObject *py_rhs = NULL;
    PyObject *py_solution = NULL;
    Mat *factor = NULL;
    Vec rhs = NULL;
    Vec solution = NULL;
    PetscErrorCode ierr;

    if (!PyArg_ParseTuple(args, "OOO", &capsule, &py_rhs, &py_solution)) {
        return NULL;
    }
    factor = get_factor(capsule);
    if (factor == NULL) return NULL;
    if (!PyObject_TypeCheck(py_rhs, &PyPetscVec_Type) ||
        !PyObject_TypeCheck(py_solution, &PyPetscVec_Type)) {
        PyErr_SetString(
            PyExc_TypeError,
            "rhs and solution must be petsc4py PETSc.Vec objects"
        );
        return NULL;
    }
    rhs = PyPetscVec_Get(py_rhs);
    solution = PyPetscVec_Get(py_solution);
    ierr = schur
        ? MatFactorSolveSchurComplement(*factor, rhs, solution)
        : MatSolve(*factor, rhs, solution);
    if (raise_petsc_error(
            ierr,
            schur ? "selected-Schur solve" : "full factor solve") < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *solve_schur(PyObject *self, PyObject *args)
{
    return solve_impl(args, PETSC_TRUE);
}

static PyObject *solve_full(PyObject *self, PyObject *args)
{
    return solve_impl(args, PETSC_FALSE);
}

static PyMethodDef methods[] = {
    {"create_factor", (PyCFunction)create_factor, METH_VARARGS | METH_KEYWORDS,
     "Create and factor a MUMPS selected Schur complement."},
    {"solve_schur", solve_schur, METH_VARARGS,
     "Solve the selected condensed stiffness system."},
    {"solve_full", solve_full, METH_VARARGS,
     "Solve the original full system using the same MUMPS factor."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_petsc_schur",
    "Minimal petsc4py bridge for PETSc factor-Schur functions.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__petsc_schur(void)
{
    if (import_petsc4py() < 0) return NULL;
    return PyModule_Create(&module);
}

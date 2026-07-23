import os
import sys

from setuptools import Extension, setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_modules = []
if os.environ.get("COFEBEM_BUILD_PETSC_SCHUR") == "1":
    try:
        import petsc4py
    except ImportError as exc:
        raise RuntimeError(
            "COFEBEM_BUILD_PETSC_SCHUR=1 requires petsc4py in the active "
            "environment; use --no-build-isolation inside fenicsx-env"
        ) from exc
    petsc_dir = petsc4py.get_config().get("PETSC_DIR", sys.prefix)
    ext_modules.append(
        Extension(
            "cofebem.fenics._petsc_schur",
            sources=["cofebem/fenics/_petsc_schur.c"],
            include_dirs=[petsc4py.get_include(), os.path.join(petsc_dir, "include")],
            library_dirs=[os.path.join(petsc_dir, "lib")],
            libraries=["petsc"],
            runtime_library_dirs=[os.path.join(petsc_dir, "lib")],
        )
    )

setup(
    name="cofebem",
    version="0.1.0",
    author="cofebem, Yahya Boye, Vladislav Yastrebov",
    author_email="yahyaboye1998@gmail.com",
    description="This code enables to construct contact problem as an auxilary problem and solve it using BEM solver accelerated by H-matrices (hierarchical matrices)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cofebem/cofebem-python",
    packages=find_packages(),
    license="BSD-3-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
    ],
    ext_modules=ext_modules,
)

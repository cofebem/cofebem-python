from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
)

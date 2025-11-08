
# Help with FEniCSx

+ To activate:

```shell
$ conda activate fenicsx-env
```

Use new environment for fenicsx v0.10
`conda activate dolfinx-010`


+ To desactivate:

```shell
$ conda deactivate
```

+ To install
```shell 
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

+ To run solver

`python simplified_3D_elasticity_clean.py`

+ To run paraview with my presettings

`paraview --state=out_elasticity/fenicsx.pvsm`


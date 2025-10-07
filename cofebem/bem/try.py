import numpy as np
import meshio

mesh = meshio.read("fine_sphere.msh")

meshio.write("fine_sphere.vtk", mesh)

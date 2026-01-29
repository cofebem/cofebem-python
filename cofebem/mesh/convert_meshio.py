import meshio

input_file = "msh_files/hemisphere6.msh"


vtk_output = "/hemisphere6.vtk"
xdmf_output = "hertz_cube.xdmf"

mesh = meshio.read(input_file)

meshio.write(vtk_output, mesh)

# meshio.write(xdmf_output, mesh)

print(f"Converted '{input_file}' to:")
print(f"  → {vtk_output} (for ParaView)")
# print(f"  → {xdmf_output} (for FEniCSx)")
